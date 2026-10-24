
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class SignalExtractor:
    """Extract RGB measurements and visible ROI boxes from multiple faces."""

    OUTLINE = [
        10, 109, 67, 103, 54, 21, 162, 127, 234,
        93, 132, 58, 172, 136, 150, 149, 176, 148,
        152, 377, 400, 378, 379, 365, 397, 288, 361,
        323, 454, 356, 389, 251, 284, 332, 297, 338,
    ]

    # Region coordinates are relative to eye distance.
    REGIONS = {
        "forehead": (-0.55, -0.90, 0.55, -0.48),
        "left_cheek": (-1.05, 0.35, -0.48, 0.78),
        "right_cheek": (0.48, 0.35, 1.05, 0.78),
    }

    def __init__(self, max_faces=8, min_skin_fraction=0.25):
        self.min_skin_fraction = min_skin_fraction
        self.last_timestamp_ms = -1

        model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "face_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing face landmark model: {model_path}"
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    @staticmethod
    def _skin_mask(roi):
        """Return a simple YCrCb skin-colour mask."""
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
    def _clip_points(points, width, height):
        points = np.asarray(points, dtype=np.float32).copy()
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        return np.round(points).astype(int)

    def extract(self, frame):
        """Return face landmarks, ROI boxes, RGB values and timestamps."""
        now = time.perf_counter()
        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            points = np.array(
                [[lm.x * width, lm.y * height] for lm in landmarks],
                dtype=np.float32,
            )

            left_eye = points[[33, 133]].mean(axis=0)
            right_eye = points[[362, 263]].mean(axis=0)

            eye_distance = float(
                np.linalg.norm(right_eye - left_eye)
            )
            if eye_distance < 10:
                continue

            tilt = float(
                np.degrees(
                    np.arctan2(
                        right_eye[1] - left_eye[1],
                        right_eye[0] - left_eye[0],
                    )
                )
            )

            eye_midpoint = (left_eye + right_eye) / 2
            matrix = cv2.getRotationMatrix2D(
                tuple(eye_midpoint),
                tilt,
                1.0,
            )

            aligned = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            aligned_points = cv2.transform(
                points.reshape(1, -1, 2),
                matrix,
            )[0]

            eye_center = aligned_points[[33, 133, 362, 263]].mean(axis=0)
            ex, ey = eye_center
            inverse = cv2.invertAffineTransform(matrix)

            outline = aligned_points[self.OUTLINE]
            face_polygon = cv2.transform(
                outline.reshape(1, -1, 2),
                inverse,
            )[0]
            face_polygon = self._clip_points(
                face_polygon, width, height
            )

            measurements = []
            valid_rgb_values = []

            for name, (rx1, ry1, rx2, ry2) in self.REGIONS.items():
                x1 = int(round(ex + rx1 * eye_distance))
                y1 = int(round(ey + ry1 * eye_distance))
                x2 = int(round(ex + rx2 * eye_distance))
                y2 = int(round(ey + ry2 * eye_distance))

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                # Keep region boxes in original camera-frame coordinates.
                corners = np.array(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                    dtype=np.float32,
                )

                original_corners = cv2.transform(
                    corners.reshape(1, -1, 2),
                    inverse,
                )[0]

                box = self._clip_points(
                    original_corners, width, height
                )

                roi = aligned[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                mask = self._skin_mask(roi)
                skin_fraction = float(mask.mean())
                valid = skin_fraction >= self.min_skin_fraction
                mean_rgb = None

                if valid:
                    pixels = roi[mask]
                    if pixels.size:
                        # Convert OpenCV BGR channel order to RGB.
                        mean_rgb = pixels.mean(axis=0)[::-1].copy()
                        valid_rgb_values.append(mean_rgb)
                    else:
                        valid = False

                # Preserve boxes even if skin detection rejects the ROI.
                measurements.append({
                    "name": name,
                    "box": box,
                    "rgb": mean_rgb,
                    "skin_fraction": skin_fraction,
                    "valid": valid,
                })

            rgb = (
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
                "rgb": rgb,
                "valid_regions": len(valid_rgb_values),
            })

        # Temporary left-to-right ordering only.
        # PersonTracker should handle persistent tracking IDs.
        faces.sort(key=lambda face: face["center"][0])

        for index, face in enumerate(faces, start=1):
            face["person_id"] = index

        return faces
