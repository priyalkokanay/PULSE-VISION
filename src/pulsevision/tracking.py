from collections import deque
import math
import time

import numpy as np


class PersonTracker:
    def __init__(
        self,
        buffer_seconds=30,
        expected_fps=30,
        max_missing_seconds=10.0,
    ):
        self.buffer_seconds = buffer_seconds
        self.expected_fps = expected_fps
        self.max_missing_seconds = max_missing_seconds
        self.tracks = {}
        self.next_track_id = 1
        self.max_samples = max(1, int(buffer_seconds * expected_fps))

    @staticmethod
    def _bbox(face):
        polygon = face.get("face_polygon")
        if polygon is None or len(polygon) == 0:
            return None

        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        return (
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        )

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def _new_track(self, face, now):
        track_id = self.next_track_id
        self.next_track_id += 1

        bbox = self._bbox(face)

        self.tracks[track_id] = {
            "bbox": bbox,
            "center": self._center(bbox) if bbox else None,
            "last_seen": now,
            "first_seen": now,
            "buffer": deque(maxlen=self.max_samples),
        }

        return track_id

    def _add_sample(self, track, face, timestamp):
        rgb = face.get("rgb")

        if rgb is None:
            return

        try:
            values = np.asarray(rgb, dtype=float).reshape(-1)

            if (
                values.size != 3
                or not np.isfinite(timestamp)
                or not np.all(np.isfinite(values))
            ):
                return

            buffer = track["buffer"]

            if buffer and timestamp <= buffer[-1]["timestamp"]:
                return

            buffer.append({
                "timestamp": float(timestamp),
                "rgb": values.tolist(),
            })

        except (TypeError, ValueError, KeyError):
            return

    def update(self, detections, now=None):
        if now is None:
            now = time.perf_counter()

        now = float(now)

        # Remove tracks that have been missing too long.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] > self.max_missing_seconds
        ]

        for track_id in expired:
            del self.tracks[track_id]

        valid_detections = []
        for face in detections:
            bbox = self._bbox(face)
            if bbox is not None:
                valid_detections.append((face, bbox))

        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(valid_detections)))
        assignments = {}

        # Match detections to existing tracks using bounding-box overlap
        # and center distance.
        candidates = []

        for index, (face, bbox) in enumerate(valid_detections):
            center = self._center(bbox)

            for track_id in unmatched_tracks:
                track = self.tracks[track_id]
                previous_bbox = track.get("bbox")
                previous_center = track.get("center")

                if previous_bbox is None or previous_center is None:
                    continue

                overlap = self._iou(previous_bbox, bbox)

                distance = math.dist(center, previous_center)
                scale = max(
                    1.0,
                    math.dist(
                        (previous_bbox[0], previous_bbox[1]),
                        (previous_bbox[2], previous_bbox[3]),
                    ),
                )

                normalized_distance = distance / scale

                if overlap >= 0.05 or normalized_distance <= 0.8:
                    score = (
                        0.7 * (1.0 - overlap)
                        + 0.3 * min(normalized_distance, 2.0)
                    )
                    candidates.append((score, track_id, index))

        candidates.sort(key=lambda item: item[0])

        for score, track_id, index in candidates:
            if (
                track_id in unmatched_tracks
                and index in unmatched_detections
            ):
                assignments[index] = track_id
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(index)

        # Create tracks for newly detected people.
        for index in sorted(unmatched_detections):
            face, bbox = valid_detections[index]
            assignments[index] = self._new_track(face, now)

        results = []

        for index, (face, bbox) in enumerate(valid_detections):
            track_id = assignments[index]
            track = self.tracks[track_id]

            track["bbox"] = bbox
            track["center"] = self._center(bbox)
            track["last_seen"] = now

            # Use one monotonic clock for both tracking and signal samples.
            self._add_sample(track, face, now)

            result = dict(face)
            result["track_id"] = track_id
            result["timestamp"] = now
            result["buffer"] = track["buffer"]
            result["buffer_samples"] = len(track["buffer"])
            result["tracking_missing_seconds"] = 0.0

            results.append(result)

        return results