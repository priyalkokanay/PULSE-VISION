
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class SignalExtractor:
    """Fast face and skin-region extraction for PulseVision."""

    OUTLINE = [
        10, 109, 67, 103, 54, 21, 162, 127, 234,
        93, 132, 58, 172, 136, 150, 149, 176, 148,
        152, 377, 400, 378, 379, 365, 397, 288, 361,
        323, 454, 356, 389, 251, 284, 332, 297, 338,
    ]

    # Region coordinates are relative to the eye midpoint and eye distance.
    # The regions are axis-aligned to avoid rotating the entire camera frame.
    REGIONS = {
        "forehead": (-0.55, -0.90, 0.55, -0.48),
        "left_cheek": (-1.05, 0.35, -0.48, 0.78),
        "right_cheek": (0.48, 0.35, 1.05, 0.78),
    }

    def __init__(self, max_faces=2, min_skin_fraction=0.25):
        self.min_skin_fraction = min_skin_fraction
        self.last_timestamp_ms = -1

        model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "face_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe model: {model_path}"
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max(1, int(max_faces)),
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    @staticmethod
    def _skin_mask(roi):
        """Return a lightweight YCrCb skin mask for a small BGR ROI."""
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        return (
            (cr >= 125)
            & (cr <= 180)
            & (cb >= 65)
            & (cb <= 135)
        )

    @staticmethod
    def _clip_box(x1, y1, x2, y2, width, height):
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))
        return x1, y1, x2, y2

    def extract(self, frame):
        """
        Detect faces and return:
        - face outline and center
        - three sampling regions
        - per-region skin fraction and RGB mean
        - a timestamp for each detected face
        """
        if self.landmarker is None:
            raise RuntimeError("SignalExtractor is closed")

        now = time.perf_counter()
        height, width = frame.shape[:2]

        # MediaPipe expects an RGB image.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = max(
            int(now * 1000),
            self.last_timestamp_ms + 1,
        )
        self.last_timestamp_ms = timestamp_ms

        detection = self.landmarker.detect_for_video(
            image,
            timestamp_ms,
        )

        faces = []

        for landmarks in detection.face_landmarks:
            points = np.asarray(
                [(lm.x * width, lm.y * height) for lm in landmarks],
                dtype=np.float32,
            )

            # Eye centers provide a rough scale for the face sampling regions.
            left_eye = points[[33, 133]].mean(axis=0)
            right_eye = points[[362, 263]].mean(axis=0)

            eye_distance = float(
                np.linalg.norm(right_eye - left_eye)
            )

            if eye_distance < 10:
                continue

            eye_midpoint = (left_eye + right_eye) * 0.5

            tilt = float(
                np.degrees(
                    np.arctan2(
                        right_eye[1] - left_eye[1],
                        right_eye[0] - left_eye[0],
                    )
                )
            )

            outline = points[self.OUTLINE]
            face_polygon = np.round(outline).astype(np.int32)
            face_polygon[:, 0] = np.clip(
                face_polygon[:, 0], 0, width - 1
            )
            face_polygon[:, 1] = np.clip(
                face_polygon[:, 1], 0, height - 1
            )

            measurements = []
            valid_rgb_values = []

            eye_x, eye_y = eye_midpoint

            for name, (rx1, ry1, rx2, ry2) in self.REGIONS.items():
                x1, y1, x2, y2 = self._clip_box(
                    eye_x + rx1 * eye_distance,
                    eye_y + ry1 * eye_distance,
                    eye_x + rx2 * eye_distance,
                    eye_y + ry2 * eye_distance,
                    width,
                    height,
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                mask = self._skin_mask(roi)
                skin_fraction = float(mask.mean())
                valid = skin_fraction >= self.min_skin_fraction
                mean_rgb = None

                if valid:
                    pixels = roi[mask]

                    if pixels.size:
                        # OpenCV uses BGR; PulseVision stores RGB.
                        mean_rgb = pixels.mean(axis=0)[::-1].astype(
                            np.float64
                        )
                        valid_rgb_values.append(mean_rgb)
                    else:
                        valid = False

                # Axis-aligned boxes keep drawing and frontend data simple.
                box = np.asarray(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                    dtype=np.int32,
                )

                measurements.append({
                    "name": name,
                    "box": box,
                    "rgb": mean_rgb,
                    "skin_fraction": skin_fraction,
                    "valid": valid,
                })

            face_rgb = (
                np.mean(valid_rgb_values, axis=0)
                if valid_rgb_values
                else None
            )

            faces.append({
                "timestamp": now,
                "center": (
                    float(points[:, 0].mean()),
                    float(points[:, 1].mean()),
                ),
                "face_polygon": face_polygon,
                "tilt": tilt,
                "regions": measurements,
                "rgb": face_rgb,
            })

        # Temporary deterministic order; PersonTracker assigns persistent IDs.
        faces.sort(key=lambda face: face["center"][0])

        for index, face in enumerate(faces, start=1):
            face["person_id"] = index

        return faces
