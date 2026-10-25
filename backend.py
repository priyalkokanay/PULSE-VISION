
import asyncio
import base64
import csv
import io
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.pulsevision import database
from src.pulsevision.signal_extraction import SignalExtractor
from src.pulsevision.tracking import PersonTracker
from src.pulsevision.pos_engine import POSEngine


class PulseVisionBackend:
    def __init__(self):
        self.extractor = None
        self.tracker = None
        self.camera = None
        self.worker = None

        self.lock = threading.RLock()
        self.running = False
        self.stop_event = threading.Event()

        self.engine = POSEngine(
            min_bpm=45,
            max_bpm=180,
            min_seconds=20,
            max_seconds=30,
        )

        # These IDs only last for the current camera run.
        self.track_patient_ids = {}
        self.patient_sessions = {}
        self.patient_last_saved = {}
        self.last_estimate_time = {}
        self.last_results = {}

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
        self.processing_fps = 0.0

    def start(self):
        with self.lock:
            if self.running:
                return

            extractor = None
            camera = None

            try:
                # No face-recognition model is loaded here.
                extractor = SignalExtractor()

                for backend_id in (
                    cv2.CAP_DSHOW,
                    cv2.CAP_MSMF,
                    cv2.CAP_ANY,
                ):
                    candidate = cv2.VideoCapture(0, backend_id)

                    if candidate.isOpened():
                        camera = candidate
                        break

                    candidate.release()

                if camera is None or not camera.isOpened():
                    raise RuntimeError(
                        "Could not open the webcam. Close other apps "
                        "using the camera and try again."
                    )

                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.extractor = extractor
                self.camera = camera
                self.tracker = PersonTracker(
                    buffer_seconds=30,
                    expected_fps=30,
                    max_missing_seconds=10,
                )

                self.track_patient_ids.clear()
                self.patient_sessions.clear()
                self.patient_last_saved.clear()
                self.last_estimate_time.clear()
                self.last_results.clear()
                self.rows = []

                self.stop_event.clear()
                self.started_at = time.perf_counter()
                self.processing_fps = 0.0
                self.latest_frame = None
                self.latest_data = {
                    "connected": True,
                    "status": "Collecting signal",
                    "fps": 0.0,
                    "people": [],
                    "session_seconds": 0,
                }

                self.running = True
                self.worker = threading.Thread(
                    target=self._capture_loop,
                    daemon=True,
                    name="PulseVisionCapture",
                )
                self.worker.start()

            except Exception:
                self.running = False
                self.stop_event.set()

                if camera is not None:
                    camera.release()

                if extractor is not None:
                    try:
                        extractor.close()
                    except Exception:
                        pass

                self.camera = None
                self.extractor = None
                self.tracker = None
                raise

    def stop(self):
        with self.lock:
            if not self.running:
                return

            self.running = False
            self.stop_event.set()

            camera = self.camera
            worker = self.worker
            self.camera = None

        if camera is not None:
            camera.release()

        if worker is not None and worker.is_alive():
            worker.join(timeout=5)

        for session_id in list(self.patient_sessions.values()):
            try:
                database.end_session(session_id)
            except Exception as exc:
                print(f"[PulseVision] Could not close session: {exc}")

        with self.lock:
            extractor = self.extractor
            self.extractor = None
            self.tracker = None
            self.worker = None

            self.track_patient_ids.clear()
            self.patient_sessions.clear()
            self.patient_last_saved.clear()
            self.last_estimate_time.clear()
            self.last_results.clear()

            self.latest_frame = None
            self.latest_data = {
                "connected": False,
                "status": "Camera stopped",
                "fps": None,
                "people": [],
                "session_seconds": 0,
            }

        if extractor is not None:
            try:
                extractor.close()
            except Exception as exc:
                print(f"[PulseVision] Extractor shutdown warning: {exc}")

    def _create_patient_for_track(self, frame, face, track_id):
        """
        Save one patient photo per live tracking ID.

        This intentionally does not claim to recognize a person across
        separate camera sessions.
        """
        key = str(track_id)

        if key in self.track_patient_ids:
            return self.track_patient_ids[key]

        polygon = face.get("face_polygon")
        if polygon is None:
            return None

        try:
            points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
            if len(points) < 3:
                return None

            x, y, width, height = cv2.boundingRect(points)

            if width < 40 or height < 40:
                return None

            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + width + padding)
            y2 = min(frame.shape[0], y + height + padding)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            success, encoded = cv2.imencode(
                ".jpg",
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if not success:
                return None

            photo = (
                "data:image/jpeg;base64,"
                + base64.b64encode(encoded.tobytes()).decode("ascii")
            )

            patient = database.create_automatic_patient(
                face_photo=photo,
            )

            if not patient or not patient.get("patient_id"):
                raise RuntimeError(
                    "The database did not return a patient ID."
                )

            patient_id = patient["patient_id"]
            session_id = database.create_session(patient_id)

            self.track_patient_ids[key] = patient_id
            self.patient_sessions[key] = session_id
            self.patient_last_saved[key] = 0

            print(
                f"[PulseVision] Saved face photo for "
                f"{patient.get('person_label', patient.get('name', patient_id))}"
            )
            return patient_id

        except Exception as exc:
            print(f"[PulseVision] Patient auto-save failed: {exc}")
            return None

    def _save_reading(self, track_id, patient_id, result):
        if not patient_id or result is None:
            return

        try:
            bpm = getattr(result, "bpm", None)
            quality = getattr(result, "signal_quality", None)

            if bpm is None:
                return

            bpm = float(bpm)
            if not np.isfinite(bpm) or not 35 <= bpm <= 220:
                return

            if quality is not None:
                quality = float(quality)
                if not np.isfinite(quality):
                    quality = None

            key = str(track_id)
            now = time.time()

            # Save at most one reading per track every five seconds.
            if now - self.patient_last_saved.get(key, 0) < 5:
                return

            database.save_reading(
                patient_id=patient_id,
                heart_rate=bpm,
                confidence=quality,
                signal_quality=quality,
                session_id=self.patient_sessions.get(key),
                raw_data={
                    "source": "webcam",
                    "tracking_id": key,
                    "method": "experimental_rppg",
                },
            )

            self.patient_last_saved[key] = now
            print(
                f"[PulseVision] Saved {bpm:.1f} BPM "
                f"for patient {patient_id}"
            )

        except Exception as exc:
            print(f"[PulseVision] BPM database save failed: {exc}")

    def _capture_loop(self):
        frame_count = 0
        fps_started = time.perf_counter()
        cached_frame = None

        while not self.stop_event.is_set():
            camera = self.camera
            extractor = self.extractor
            tracker = self.tracker

            if camera is None or extractor is None or tracker is None:
                break

            success, frame = camera.read()

            if not success:
                with self.lock:
                    self.latest_data["status"] = "Camera frame capture failed"
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            now = time.perf_counter()

            try:
                detections = extractor.extract(frame)
                tracked_people = tracker.update(detections, now=now)
            except Exception as exc:
                with self.lock:
                    self.latest_data["status"] = f"Pipeline error: {exc}"
                print(f"[PulseVision] Detection error: {exc}")
                time.sleep(0.03)
                continue

            frame_count += 1
            elapsed = now - fps_started

            if elapsed >= 1:
                self.processing_fps = frame_count / elapsed
                frame_count = 0
                fps_started = now

            people_data = []

            for face in tracked_people:
                track_id = face.get(
                    "track_id",
                    face.get("person_id", face.get("id")),
                )
                if track_id is None:
                    continue

                patient_id = self._create_patient_for_track(
                    frame,
                    face,
                    track_id,
                )

                samples = list(face.get("buffer", []))
                timestamps = [
                    float(s["timestamp"])
                    for s in samples
                    if isinstance(s, dict)
                    and isinstance(s.get("timestamp"), (int, float))
                    and np.isfinite(s["timestamp"])
                ]

                signal_seconds = (
                    max(timestamps) - min(timestamps)
                    if len(timestamps) >= 2
                    else 0.0
                )
                signal_seconds = max(0.0, float(signal_seconds))

                result = self.last_results.get(track_id)

                if (
                    signal_seconds >= 20
                    and now - self.last_estimate_time.get(track_id, 0) >= 1
                ):
                    try:
                        usable = [
                            s for s in samples
                            if isinstance(s, dict)
                            and "timestamp" in s
                            and "rgb" in s
                        ]

                        if len(usable) >= 2:
                            result = self.engine.estimate(
                                [s["timestamp"] for s in usable],
                                [s["rgb"] for s in usable],
                                fs=30.0,
                            )

                            self.last_results[track_id] = result
                            self.last_estimate_time[track_id] = now

                            bpm = getattr(result, "bpm", None)
                            quality = getattr(
                                result, "signal_quality", None
                            )

                            with self.lock:
                                self.rows.append({
                                    "timestamp": time.time(),
                                    "person_id": track_id,
                                    "patient_id": patient_id or "",
                                    "signal_seconds": round(signal_seconds, 2),
                                    "bpm": bpm,
                                    "signal_quality": quality,
                                    "status": getattr(
                                        result,
                                        "status",
                                        "Estimate generated",
                                    ),
                                })

                            self._save_reading(
                                track_id,
                                patient_id,
                                result,
                            )

                    except Exception as exc:
                        print(f"[PulseVision] BPM estimation error: {exc}")

                bpm = getattr(result, "bpm", None) if result else None
                quality = (
                    getattr(result, "signal_quality", None)
                    if result else None
                )
                status = (
                    getattr(result, "status", "Estimate generated")
                    if result
                    else "Collecting signal"
                )

                polygon = face.get("face_polygon")
                if polygon is not None:
                    try:
                        points = np.asarray(
                            polygon, dtype=np.int32
                        ).reshape(-1, 1, 2)
                        if len(points) >= 3:
                            cv2.polylines(
                                frame,
                                [points],
                                True,
                                (105, 160, 125),
                                2,
                            )
                    except (ValueError, TypeError):
                        pass

                regions_data = []
                for region in face.get("regions", []):
                    box = region.get("box")
                    if box is None:
                        continue

                    try:
                        points = np.asarray(
                            box, dtype=np.int32
                        ).reshape(-1, 2)
                    except (ValueError, TypeError):
                        continue

                    if len(points) >= 3:
                        cv2.polylines(
                            frame,
                            [points.reshape(-1, 1, 2)],
                            True,
                            (110, 170, 145),
                            2,
                        )

                    regions_data.append({
                        "name": region.get("name", "region"),
                        "box": points.tolist(),
                        "skin_fraction": float(
                            region.get("skin_fraction", 0)
                        ),
                        "valid": bool(region.get("valid", False)),
                    })

                center_x, center_y = face.get("center", (0, 0))
                label = (
                    f"{patient_id} | {bpm:.1f} BPM"
                    if patient_id
                    and bpm is not None
                    and np.isfinite(bpm)
                    else f"{patient_id or 'Saving face...'} | {status}"
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        max(8, int(center_x) - 90),
                        max(20, int(center_y) - 15),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (245, 245, 235),
                    2,
                    cv2.LINE_AA,
                )

                people_data.append({
                    "id": track_id,
                    "patient_id": patient_id,
                    "bpm": bpm,
                    "signal_quality": quality,
                    "status": status,
                    "buffer_samples": len(samples),
                    "signal_seconds": round(signal_seconds, 2),
                    "regions": regions_data,
                })

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if success:
                cached_frame = base64.b64encode(
                    encoded.tobytes()
                ).decode("ascii")

            duration = now - self.started_at if self.started_at else 0

            with self.lock:
                self.latest_data = {
                    "connected": True,
                    "status": "Experimental live processing",
                    "fps": round(self.processing_fps, 1),
                    "people": people_data,
                    "session_seconds": round(duration, 1),
                }
                self.latest_frame = cached_frame

    def status(self):
        with self.lock:
            return dict(self.latest_data)


backend = PulseVisionBackend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    backend.stop()


app = FastAPI(title="PulseVision API", version="0.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "camera_running": backend.running,
        "face_recognition": "disabled",
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
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/stop")
def stop_camera():
    backend.stop()
    return {"ok": True, "status": "Camera stopped"}


@app.get("/api/patients")
def list_patients():
    patients = database.get_all_patients()

    # Don't expose face embeddings to the browser.
    for patient in patients:
        patient.pop("face_embedding", None)

    return {"patients": patients, "count": len(patients)}


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str):
    patient = database.get_patient(patient_id)

    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.pop("face_embedding", None)
    return patient


@app.get("/api/patients/{patient_id}/readings")
def patient_readings(patient_id: str, limit: int = 500):
    if database.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "readings": database.get_patient_readings(
            patient_id,
            limit=limit,
        )
    }


@app.get("/api/patients/{patient_id}/readings/latest")
def latest_reading(patient_id: str):
    if database.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {"reading": database.get_latest_reading(patient_id)}


@app.get("/api/patients/{patient_id}/sessions")
def patient_sessions(patient_id: str):
    if database.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {"sessions": database.get_patient_sessions(patient_id)}


@app.delete("/api/patients/clear")
def clear_database():
    if backend.running:
        raise HTTPException(
            status_code=409,
            detail="Stop monitoring before resetting the database.",
        )

    try:
        return {
            "ok": True,
            "message": "Database reset successfully",
            **database.clear_all_data(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database reset failed: {exc}",
        )


@app.get("/api/export.csv")
def export_csv():
    with backend.lock:
        rows = list(backend.rows)

    output = io.StringIO()
    fields = [
        "timestamp",
        "person_id",
        "patient_id",
        "signal_seconds",
        "bpm",
        "signal_quality",
        "status",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                'attachment; filename="pulsevision_session.csv"'
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
        print(f"[PulseVision] WebSocket error: {exc}")
