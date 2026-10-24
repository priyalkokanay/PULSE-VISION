
import cv2
import mediapipe as mp
from pathlib import Path


class FaceDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "blaze_face_short_range.tflite"
            )

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face detector model not found: {model_path}"
            )

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.6,
        )

        self.detector = mp.tasks.vision.FaceDetector.create_from_options(
            options
        )

    def detect(self, frame):
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        results = self.detector.detect(mp_image)

        if not results.detections:
            return None

        best = max(
            results.detections,
            key=lambda detection: detection.categories[0].score,
        )

        box = best.bounding_box

        x = max(0, box.origin_x)
        y = max(0, box.origin_y)
        w = min(box.width, width - x)
        h = min(box.height, height - y)

        if w <= 0 or h <= 0:
            return None

        return x, y, w, h

    def close(self):
        self.detector.close()
