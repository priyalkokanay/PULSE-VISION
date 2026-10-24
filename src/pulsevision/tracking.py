
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

        self.max_samples = max(
            1, int(buffer_seconds * expected_fps)
        )

    @staticmethod
    def _bbox(face):
        polygon = face.get("face_polygon")

        if polygon is None or len(polygon) == 0:
            return None

        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

        x1 = float(np.min(points[:, 0]))
        y1 = float(np.min(points[:, 1]))
        x2 = float(np.max(points[:, 0]))
        y2 = float(np.max(points[:, 1]))

        return x1, y1, x2, y2

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

    def _add_sample(self, track, face):
        rgb = face.get("rgb")

        if rgb is None:
            return

        try:
            values = np.asarray(rgb, dtype=float)

            if values.size == 3 and np.all(np.isfinite(values)):
                track["buffer"].append({
                    "timestamp": face.get("timestamp", time.time()),
                    "rgb": values.tolist(),
                })
        except (TypeError, ValueError):
            pass

    def update(self, detections, now=None):
        if now is None:
            now = time.perf_counter()

        # Keep temporarily missing tracks for 10 seconds.
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] > self.max_missing_seconds
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]

        detections = list(detections or [])
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        assignments = {}

        # Build possible matches between existing tracks and detections.
        candidates = []

        for track_id, track in self.tracks.items():
            old_bbox = track["bbox"]
            old_center = track["center"]

            if old_bbox is None or old_center is None:
                continue

            elapsed = max(0.0, now - track["last_seen"])

            for index, face in enumerate(detections):
                new_bbox = self._bbox(face)

                if new_bbox is None:
                    continue

                new_center = self._center(new_bbox)

                distance = math.dist(old_center, new_center)
                iou = self._iou(old_bbox, new_bbox)

                old_width = max(1.0, old_bbox[2] - old_bbox[0])
                old_height = max(1.0, old_bbox[3] - old_bbox[1])
                face_scale = max(old_width, old_height)

                # Allow more movement after a longer disappearance.
                distance_limit = face_scale * (
                    1.0 + 1.5 * elapsed
                )

                if distance <= distance_limit or iou > 0.05:
                    normalized_distance = distance / face_scale
                    score = normalized_distance - (2.0 * iou)
                    candidates.append((score, track_id, index))

        # Greedy one-to-one matching: each track and face can be used once.
        candidates.sort(key=lambda item: item[0])

        for score, track_id, index in candidates:
            if (
                track_id in unmatched_tracks
                and index in unmatched_detections
            ):
                assignments[index] = track_id
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(index)

        # Assign new IDs to detections that did not match an existing track.
        for index in sorted(unmatched_detections):
            assignments[index] = self._new_track(detections[index], now)

        results = []

        for index, face in enumerate(detections):
            track_id = assignments[index]
            track = self.tracks[track_id]

            bbox = self._bbox(face)

            if bbox is not None:
                track["bbox"] = bbox
                track["center"] = self._center(bbox)

            track["last_seen"] = now
            self._add_sample(track, face)

            result = dict(face)
            result["track_id"] = track_id
            result["buffer"] = track["buffer"]
            result["buffer_samples"] = len(track["buffer"])
            result["tracking_missing_seconds"] = 0.0

            results.append(result)

        return results
