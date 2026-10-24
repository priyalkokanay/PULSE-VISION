
import csv
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pulsevision.signal_extraction import SignalExtractor
from pulsevision.tracking import PersonTracker
from pulsevision.pos_engine import POSEngine


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 0
CAMERA_API = cv2.CAP_MSMF

MAX_FACES = 8
BUFFER_SECONDS = 30
EXPECTED_FPS = 30

ESTIMATE_INTERVAL_SECONDS = 1.0
BPM_HISTORY_LENGTH = 5

OUTPUT_CSV = ROOT / "data" / "rgb_trace.csv"

REGION_NAMES = (
    "forehead",
    "left_cheek",
    "right_cheek",
)


# =========================================================
# INITIALIZATION
# =========================================================

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(CAMERA_INDEX, CAMERA_API)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions or CAMERA_INDEX."
    )

extractor = SignalExtractor(max_faces=MAX_FACES)

tracker = PersonTracker(
    buffer_seconds=BUFFER_SECONDS,
    expected_fps=EXPECTED_FPS,
    max_missing_seconds=10.0,
)

pos_engine = POSEngine(
    min_bpm=45,
    max_bpm=180,
    min_seconds=20,
    max_seconds=30,
)

last_estimate_time = defaultdict(float)
bpm_history = defaultdict(lambda: deque(maxlen=BPM_HISTORY_LENGTH))
person_results = {}

csv_file = OUTPUT_CSV.open("w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp",
    "track_id",
    "face_tilt_deg",
    "red",
    "green",
    "blue",
    "forehead_r",
    "forehead_g",
    "forehead_b",
    "forehead_skin_fraction",
    "left_cheek_r",
    "left_cheek_g",
    "left_cheek_b",
    "left_cheek_skin_fraction",
    "right_cheek_r",
    "right_cheek_g",
    "right_cheek_b",
    "right_cheek_skin_fraction",
    "valid_regions",
    "buffer_samples",
    "estimated_bpm",
    "signal_quality",
    "estimate_status",
])

print("PulseVision started. Press Q to quit.")
print(f"RGB trace CSV: {OUTPUT_CSV}")


# =========================================================
# REGION HELPERS
# =========================================================

def normalize_regions(face):
    """Normalize region measurements into a dictionary by region name."""
    raw_regions = face.get("regions", [])

    if isinstance(raw_regions, dict):
        return raw_regions

    normalized = {}

    if isinstance(raw_regions, list):
        for index, region in enumerate(raw_regions):
            if not isinstance(region, dict):
                continue

            name = region.get("name", region.get("region", ""))

            if not name and index < len(REGION_NAMES):
                name = REGION_NAMES[index]

            if name:
                normalized[name] = region

    return normalized


def region_values(face, region_name):
    """Get RGB and skin fraction for a named region."""
    region = normalize_regions(face).get(region_name, {})

    if not isinstance(region, dict):
        return ["", "", ""], ""

    rgb = region.get("rgb", region.get("mean_rgb"))

    if rgb is None:
        rgb = ["", "", ""]

    try:
        rgb = list(rgb)
    except TypeError:
        rgb = ["", "", ""]

    if len(rgb) != 3:
        rgb = ["", "", ""]

    return rgb, region.get("skin_fraction", "")


def draw_polygon(frame, polygon, color, thickness=1):
    """Draw a polygon or ROI box safely."""
    if polygon is None:
        return

    try:
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)

        if len(points) >= 3:
            cv2.polylines(
                frame,
                [points],
                isClosed=True,
                color=color,
                thickness=thickness,
            )
    except (TypeError, ValueError):
        pass


def draw_region_label(frame, points, label, color):
    """Draw a label above a facial region."""
    if points is None:
        return

    try:
        points = np.asarray(points, dtype=np.int32).reshape(-1, 2)

        if len(points) < 3:
            return

        x = max(0, int(points[:, 0].min()))
        y = max(14, int(points[:, 1].min()) - 5)

        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    except (TypeError, ValueError):
        pass


# =========================================================
# POS ESTIMATION
# =========================================================

def estimate_person(buffer):
    """Estimate pulse from one person's RGB history."""
    samples = list(buffer)

    if len(samples) < 2:
        return None, 0.0, "Collecting signal"

    timestamps = []
    rgb_values = []

    for sample in samples:
        try:
            timestamp = float(sample["timestamp"])
            rgb = np.asarray(sample["rgb"], dtype=float)

            if rgb.size != 3:
                continue

            if not np.isfinite(timestamp):
                continue

            if not np.all(np.isfinite(rgb)):
                continue

            timestamps.append(timestamp)
            rgb_values.append(rgb)

        except (KeyError, TypeError, ValueError):
            continue

    if len(rgb_values) < 2:
        return None, 0.0, "Collecting RGB samples"

    timestamps = np.asarray(timestamps, dtype=float)
    rgb_values = np.asarray(rgb_values, dtype=float)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    rgb_values = rgb_values[order]

    duration = timestamps[-1] - timestamps[0]

    if not np.isfinite(duration) or duration <= 0:
        return None, 0.0, "Invalid timestamps"

    if duration < 20:
        return None, 0.0, "Collecting signal"

    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]

    if len(intervals) == 0:
        return None, 0.0, "Invalid timestamps"

    estimated_fs = 1.0 / float(np.median(intervals))

    if not 5.0 <= estimated_fs <= 120.0:
        return None, 0.0, "Check sample timestamps"

    result = pos_engine.estimate(
        timestamps=timestamps,
        rgb=rgb_values,
        fs=EXPECTED_FPS,
    )

    return result.bpm, result.signal_quality, result.status


# =========================================================
# MAIN CAMERA LOOP
# =========================================================

frame_count = 0
start_time = time.perf_counter()

try:
    while True:
        ok, frame = camera.read()

        if not ok:
            print("Warning: webcam frame could not be read.")
            break

        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        frame_count += 1

        detections = extractor.extract(frame)
        faces = tracker.update(detections, now=now)

        for face in faces:
            track_id = face["track_id"]
            bbox = tracker._bbox(face)

            # Face bounding box.
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (60, 210, 100),
                    2,
                )

            # Face outline.
            draw_polygon(
                frame,
                face.get("face_polygon"),
                (60, 210, 100),
                1,
            )

            # Forehead and cheek boxes.
            # SignalExtractor stores corners in "box".
            regions = normalize_regions(face)

            region_styles = {
                "forehead": ((0, 220, 255), "FOREHEAD"),
                "left_cheek": ((255, 180, 0), "LEFT CHEEK"),
                "right_cheek": ((255, 0, 200), "RIGHT CHEEK"),
            }

            for region_name, region in regions.items():
                if not isinstance(region, dict):
                    continue

                color, label = region_styles.get(
                    region_name,
                    ((0, 220, 255), str(region_name).upper()),
                )

                corners = region.get("box", region.get("polygon"))

                draw_polygon(frame, corners, color, 2)
                draw_region_label(frame, corners, label, color)

            # Update pulse estimate once per second.
            if (
                now - last_estimate_time[track_id]
                >= ESTIMATE_INTERVAL_SECONDS
            ):
                bpm, quality, status = estimate_person(face["buffer"])
                last_estimate_time[track_id] = now

                if bpm is not None and np.isfinite(bpm):
                    bpm_history[track_id].append(float(bpm))

                displayed_bpm = (
                    float(np.median(list(bpm_history[track_id])))
                    if bpm_history[track_id]
                    else None
                )

                person_results[track_id] = {
                    "bpm": displayed_bpm,
                    "quality": quality,
                    "status": status,
                }

            result = person_results.get(
                track_id,
                {
                    "bpm": None,
                    "quality": 0.0,
                    "status": "Collecting signal",
                },
            )

            bpm = result["bpm"]
            bpm_text = "-- BPM" if bpm is None else f"{bpm:.1f} BPM"

            label = (
                f"Person {track_id} | {bpm_text} | "
                f"Quality: {result['quality']:.2f}"
            )

            if bbox is not None:
                label_x = max(0, int(bbox[0]))
                label_y = max(20, int(bbox[1]) - 10)
            else:
                label_x, label_y = 10, 30

            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (60, 210, 100),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                result["status"],
                (label_x, min(frame.shape[0] - 8, label_y + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 220, 120),
                1,
                cv2.LINE_AA,
            )

            # CSV logging.
            forehead_rgb, forehead_skin = region_values(face, "forehead")
            left_rgb, left_skin = region_values(face, "left_cheek")
            right_rgb, right_skin = region_values(face, "right_cheek")

            rgb = face.get("rgb")
            if rgb is None:
                rgb = ["", "", ""]

            csv_writer.writerow([
                face.get("timestamp", ""),
                track_id,
                face.get("tilt", ""),
                *list(rgb),
                *forehead_rgb,
                forehead_skin,
                *left_rgb,
                left_skin,
                *right_rgb,
                right_skin,
                face.get("valid_regions", ""),
                face.get("buffer_samples", 0),
                bpm if bpm is not None else "",
                result["quality"],
                result["status"],
            ])

        # Performance overlay.
        elapsed = max(0.001, time.perf_counter() - start_time)
        fps = frame_count / elapsed

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | People: {len(faces)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if frame_count % 30 == 0:
            csv_file.flush()

        cv2.imshow("PulseVision - Live Pulse Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    csv_file.flush()
    csv_file.close()
    camera.release()
    extractor.close()
    cv2.destroyAllWindows()
    print(f"PulseVision stopped. CSV saved to: {OUTPUT_CSV}")
