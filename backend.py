
import asyncio
import base64
import csv
import io
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.pulsevision.signal_extraction import SignalExtractor
from src.pulsevision.tracking import PersonTracker
from src.pulsevision.pos_engine import POSEngine


class PulseVisionBackend:
    def __init__(self):
        self.extractor = None
        self.tracker = None
        self.camera = None

        self.engine = POSEngine(
            min_bpm=45,
            max_bpm=180,
            min_seconds=20,
            max_seconds=30,
        )

        self.lock = threading.RLock()
        self.running = False
        self.worker = None
        self.stop_event = threading.Event()

        self.latest_frame = None
        self.latest_data = {
            "connected": False,
            "status": "Camera stopped",
            "fps": None,
            "people": [],
            "session_seconds": 0,
        }

        self.rows = []
        self.started_at = None
        self.last_estimate_time = {}
        self.last_results = {}
        self.processing_fps = 0.0

    def start(self):
        with self.lock:
            if self.running:
                return

            extractor = SignalExtractor()
            camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(0)

            if not camera.isOpened():
                extractor.close()
                raise RuntimeError(
                    "Unable to open webcam. Close other apps using it."
                )

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.extractor = extractor
            self.tracker = PersonTracker(
                buffer_seconds=30,
                expected_fps=30,
                max_missing_seconds=10,
            )
            self.camera = camera
            self.running = True
            self.stop_event.clear()
            self.started_at = time.perf_counter()
            self.rows = []
            self.last_estimate_time = {}
            self.last_results = {}
            self.processing_fps = 0.0

            self.latest_frame = None
            self.latest_data = {
                "connected": True,
                "status": "Collecting signal",
                "fps": 0.0,
                "people": [],
                "session_seconds": 0,
            }

            self.worker = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="PulseVisionCapture",
            )
            self.worker.start()

    def stop(self):
        with self.lock:
            if not self.running:
                return

            self.running = False
            self.stop_event.set()
            camera = self.camera
            self.camera = None
            worker = self.worker

        if camera is not None:
            camera.release()

        if worker is not None and worker.is_alive():
            worker.join(timeout=3)

        with self.lock:
            if self.extractor is not None:
                self.extractor.close()
                self.extractor = None

            self.tracker = None
            self.worker = None
            self.last_results.clear()
            self.last_estimate_time.clear()
            self.latest_frame = None
            self.latest_data = {
                "connected": False,
                "status": "Camera stopped",
                "fps": None,
                "people": [],
                "session_seconds": 0,
            }

    def _capture_loop(self):
        frame_count = 0
        fps_started = time.perf_counter()
        last_publish = 0.0
        cached_frame_b64 = None

        while not self.stop_event.is_set():
            with self.lock:
                camera = self.camera
                extractor = self.extractor
                tracker = self.tracker

            if camera is None or extractor is None or tracker is None:
                break

            ok, frame = camera.read()
            if not ok:
                with self.lock:
                    self.latest_data["status"] = "Camera frame capture failed"
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            frame_time = time.perf_counter()

            try:
                detections = extractor.extract(frame)
                tracked_people = tracker.update(detections, now=frame_time)
            except Exception as exc:
                with self.lock:
                    self.latest_data["status"] = f"Pipeline error: {exc}"
                time.sleep(0.03)
                continue

            frame_count += 1
            elapsed_fps = frame_time - fps_started
            if elapsed_fps >= 1.0:
                self.processing_fps = frame_count / elapsed_fps
                frame_count = 0
                fps_started = frame_time

            people_data = []

            for face in tracked_people:
                person_id = face.get("track_id", face.get("person_id", 0))
                samples = list(face.get("buffer", []))
                result = self.last_results.get(person_id)

                sample_times = [
                    float(sample["timestamp"])
                    for sample in samples
                    if isinstance(sample, dict)
                    and isinstance(sample.get("timestamp"), (int, float))
                    and np.isfinite(sample["timestamp"])
                ]

                signal_seconds = (
                    max(sample_times) - min(sample_times)
                    if len(sample_times) >= 2
                    else 0.0
                )
                signal_seconds = max(0.0, float(signal_seconds))

                # Recompute at most once per second after enough data exists.
                if (
                    signal_seconds >= 20.0
                    and frame_time - self.last_estimate_time.get(person_id, 0)
                    >= 1.0
                ):
                    try:
                        usable_samples = [
                            sample for sample in samples
                            if isinstance(sample, dict)
                            and "timestamp" in sample
                            and "rgb" in sample
                        ]
                        estimate_times = [
                            sample["timestamp"] for sample in usable_samples
                        ]
                        rgb_values = [
                            sample["rgb"] for sample in usable_samples
                        ]

                        new_result = self.engine.estimate(
                            estimate_times,
                            rgb_values,
                            fs=30.0,
                        )

                        self.last_results[person_id] = new_result
                        self.last_estimate_time[person_id] = frame_time
                        result = new_result

                        with self.lock:
                            self.rows.append({
                                "timestamp": time.time(),
                                "person_id": person_id,
                                "signal_seconds": round(signal_seconds, 2),
                                "bpm": new_result.bpm,
                                "signal_quality": new_result.signal_quality,
                                "status": new_result.status,
                            })
                    except Exception as exc:
                        print(f"Estimator error for person {person_id}: {exc}")

                bpm = result.bpm if result is not None else None
                quality = result.signal_quality if result is not None else None
                status = (
                    result.status if result is not None
                    else "Collecting signal" if signal_seconds < 20.0
                    else "Waiting for estimate"
                )

                polygon = face.get("face_polygon")
                if polygon is not None:
                    try:
                        polygon = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
                        if len(polygon) >= 3:
                            cv2.polylines(
                                frame, [polygon], True, (105, 160, 125), 2
                            )
                    except (ValueError, TypeError):
                        pass

                regions_data = []
                for region in face.get("regions", []):
                    box = region.get("box")
                    if box is None:
                        continue

                    try:
                        box = np.asarray(box, dtype=np.int32).reshape(-1, 2)
                    except (TypeError, ValueError):
                        continue

                    if len(box) >= 3:
                        color = {
                            "forehead": (110, 170, 145),
                            "left_cheek": (180, 150, 100),
                            "right_cheek": (130, 140, 200),
                        }.get(region.get("name"), (180, 180, 180))

                        cv2.polylines(
                            frame, [box.reshape(-1, 1, 2)], True, color, 2
                        )

                    regions_data.append({
                        "name": region.get("name", "region"),
                        "box": box.tolist(),
                        "skin_fraction": float(region.get("skin_fraction", 0)),
                        "valid": bool(region.get("valid", False)),
                    })

                center_x, center_y = face.get("center", (0, 0))
                label = (
                    f"Person {person_id} | {bpm:.1f} BPM"
                    if bpm is not None and np.isfinite(bpm)
                    else f"Person {person_id} | {status}"
                )

                cv2.putText(
                    frame,
                    label,
                    (max(8, int(center_x) - 90), max(20, int(center_y) - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 235),
                    2,
                    cv2.LINE_AA,
                )

                people_data.append({
                    "id": person_id,
                    "bpm": bpm,
                    "signal_quality": quality,
                    "status": status,
                    "buffer_samples": len(samples),
                    "signal_seconds": round(signal_seconds, 2),
                    "regions": regions_data,
                })

            # JPEG encoding is still required for the browser preview.
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if ok:
                cached_frame_b64 = base64.b64encode(encoded).decode("ascii")

            session_seconds = (
                frame_time - self.started_at if self.started_at else 0.0
            )

            with self.lock:
                self.latest_data = {
                    "connected": True,
                    "status": "Experimental live processing",
                    "fps": round(self.processing_fps, 1),
                    "people": people_data,
                    "session_seconds": round(session_seconds, 1),
                }

                # Keep the most recent encoded frame for WebSocket clients.
                self.latest_frame = cached_frame_b64

    def status(self):
        with self.lock:
            return dict(self.latest_data)


backend = PulseVisionBackend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    backend.stop()


app = FastAPI(
    title="PulseVision API",
    version="0.4.0",
    lifespan=lifespan,
)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "PulseVision API",
        "camera_running": backend.running,
    }


@app.get("/api/status")
def status():
    return backend.status()


@app.post("/api/start")
def start_camera():
    try:
        backend.start()
        return {"ok": True, "status": "Camera started"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/api/stop")
def stop_camera():
    backend.stop()
    return {"ok": True, "status": "Camera stopped"}


@app.get("/api/export.csv")
def export_csv():
    with backend.lock:
        rows = list(backend.rows)

    output = io.StringIO()
    fields = [
        "timestamp",
        "person_id",
        "signal_seconds",
        "bpm",
        "signal_quality",
        "status",
    ]

    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="pulsevision_session.csv"'
        },
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            with backend.lock:
                payload = {
                    "metrics": backend.latest_data,
                    "frame": backend.latest_frame,
                }

            await websocket.send_json(payload)
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print("WebSocket error:", exc)
