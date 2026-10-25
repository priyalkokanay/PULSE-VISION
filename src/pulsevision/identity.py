
from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from src.pulsevision import database


class FaceIdentity:
    """Persistent face matching backed by the PulseVision SQLite database."""

    def __init__(self, model_name: str = "buffalo_l", threshold: float = 0.42):
        self.threshold = threshold
        self.lock = threading.RLock()

        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=model_name,
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as exc:
            raise RuntimeError(
                "Face recognition could not initialize. Install InsightFace "
                "and ONNX Runtime and ensure the buffalo_l model is available. "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _normalize(vector: Any) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))

        if norm <= 1e-12:
            raise ValueError("The face embedding is empty or invalid.")

        return value / norm

    def detect_faces(self, frame: np.ndarray) -> list[Any]:
        """Return InsightFace detections for a BGR OpenCV image."""
        return self.app.get(frame)

    def embedding_for_face(self, face: Any) -> np.ndarray:
        embedding = getattr(face, "normed_embedding", None)

        if embedding is None:
            embedding = getattr(face, "embedding", None)

        if embedding is None:
            raise ValueError("The detected face has no embedding.")

        return self._normalize(embedding)

    def best_match(self, embedding: np.ndarray) -> tuple[dict | None, float]:
        """Compare a normalized embedding with saved database embeddings."""
        candidate = self._normalize(embedding)
        best_patient = None
        best_score = -1.0

        for patient in database.get_all_patients():
            saved = database.get_face_embedding(patient["patient_id"])

            if saved is None:
                continue

            try:
                saved_vector = self._normalize(saved)

                if saved_vector.shape != candidate.shape:
                    continue

                score = float(np.dot(candidate, saved_vector))

                if score > best_score:
                    best_score = score
                    best_patient = patient
            except (TypeError, ValueError):
                continue

        if best_patient is not None and best_score >= self.threshold:
            return best_patient, best_score

        return None, best_score

    def identify_or_create(
        self,
        frame: np.ndarray,
        face: Any,
    ) -> tuple[dict, np.ndarray, bool, float]:
        """
        Identify a face or create a new record.

        Returns (patient, embedding, is_new_patient, similarity_score).
        """
        embedding = self.embedding_for_face(face)
        matched, score = self.best_match(embedding)

        # Crop and store a face image when creating a new record.
        bbox = np.asarray(face.bbox, dtype=np.int32).reshape(-1)
        x1, y1, x2, y2 = bbox[:4]

        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError("Unable to crop the detected face.")

        ok, encoded = cv2.imencode(
            ".jpg",
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

        if not ok:
            raise RuntimeError("Unable to encode the face photo.")

        import base64

        photo = (
            "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode("ascii")
        )

        with self.lock:
            # Recheck after taking the lock to reduce duplicate creation
            # when two detections are processed close together.
            matched, score = self.best_match(embedding)

            if matched is not None:
                # Refresh the last-seen photo but retain the stored identity.
                database.update_patient_photo(
                    matched["patient_id"],
                    photo,
                )
                return matched, embedding, False, score

            patient = database.create_automatic_patient(
                face_photo=photo,
                face_embedding=embedding.tolist(),
            )

            return patient, embedding, True, score
