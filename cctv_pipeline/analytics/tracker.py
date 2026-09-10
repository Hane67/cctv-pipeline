"""
Stage 4: Multi-Object Tracking with Trajectory Management and Hybrid Motion Fallback.
Combines Ultralytics ByteTrack (for semantic COCO objects) with a high-performance
Motion Blob Tracker (for foreground-segmented moving objects, synthetic test shapes,
and general surveillance motion).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from cctv_pipeline.config.settings import TrackingConfig, DetectionConfig
from cctv_pipeline.analytics.detector import CCTV_CLASS_NAMES
from cctv_pipeline.core.logger import logger


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], margin: int = 15) -> List[Tuple[int, int, int, int]]:
    """
    Merges bounding boxes that overlap or are within `margin` pixels of each other.
    Prevents single fragmented objects from producing multiple tracking IDs.
    """
    if not boxes:
        return []

    merged = list(boxes)
    has_merged = True
    while has_merged:
        has_merged = False
        new_merged = []
        skip = set()
        for i in range(len(merged)):
            if i in skip:
                continue
            x1_a, y1_a, x2_a, y2_a = merged[i]
            for j in range(i + 1, len(merged)):
                if j in skip:
                    continue
                x1_b, y1_b, x2_b, y2_b = merged[j]
                if not (x2_a + margin < x1_b or x1_a - margin > x2_b or
                        y2_a + margin < y1_b or y1_a - margin > y2_b):
                    x1_a = min(x1_a, x1_b)
                    y1_a = min(y1_a, y1_b)
                    x2_a = max(x2_a, x2_b)
                    y2_a = max(y2_a, y2_b)
                    skip.add(j)
                    has_merged = True
            new_merged.append((x1_a, y1_a, x2_a, y2_a))
        merged = new_merged
    return merged


class MotionBlobTracker:
    """
    Centroid and IoU tracker for foreground-segmented motion blobs.
    Enables tracking of any moving object (including synthetic test shapes, animals,
    or generic surveillance targets) directly from Stage 2 foreground masks.
    """

    def __init__(self, max_disappeared: int = 20, max_distance: float = 65.0, min_area: int = 40):
        self.next_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}      # track_id -> (cx, cy)
        self.boxes: Dict[int, Tuple[int, int, int, int]] = {} # track_id -> (x1, y1, x2, y2)
        self.disappeared: Dict[int, int] = {}
        self.durations: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_area = min_area

    def reset(self) -> None:
        """Resets tracking state and ID counter."""
        self.next_id = 1
        self.objects.clear()
        self.boxes.clear()
        self.disappeared.clear()
        self.durations.clear()

    def update(self, fg_mask: np.ndarray) -> List[dict]:
        """
        Extracts contours from foreground mask and updates tracked object state.
        Returns list of active track dictionaries.
        """
        # 1. Close and dilate mask slightly to solidify hollow outlines into solid blobs
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close)
        dilated_mask = cv2.dilate(closed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            raw_boxes.append((x, y, x + w, y + h))

        # Merge adjacent or split contour boxes to prevent duplicate track IDs on single objects
        input_boxes = merge_overlapping_boxes(raw_boxes, margin=15)
        input_centroids = [
            ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            for box in input_boxes
        ]

        # If no objects currently tracked, register all new detections
        if len(self.objects) == 0:
            for pt, box in zip(input_centroids, input_boxes):
                self._register(pt, box)
        elif len(input_centroids) == 0:
            # Mark all existing objects as disappeared
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
        else:
            object_ids = list(self.objects.keys())
            object_pts = list(self.objects.values())

            # Distance matrix between existing centroids and new detection centroids
            D = np.zeros((len(object_pts), len(input_centroids)), dtype=np.float32)
            for i, (ox, oy) in enumerate(object_pts):
                for j, (ix, iy) in enumerate(input_centroids):
                    D[i, j] = math.hypot(ox - ix, oy - iy)

            # Match nearest centroids (greedy assignment)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                tid = object_ids[row]
                self.objects[tid] = input_centroids[col]
                self.boxes[tid] = input_boxes[col]
                self.disappeared[tid] = 0
                self.durations[tid] += 1

                used_rows.add(row)
                used_cols.add(col)

            # Unmatched existing objects
            unused_rows = set(range(len(object_pts))) - used_rows
            for row in unused_rows:
                tid = object_ids[row]
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

            # Unmatched new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self._register(input_centroids[col], input_boxes[col])

        # Return active tracks
        active = []
        for tid, pt in self.objects.items():
            if self.disappeared[tid] == 0:
                box = self.boxes[tid]
                active.append({
                    "track_id": int(tid),
                    "bbox": box,
                    "centroid": pt,
                    "conf": 0.90,
                    "class_id": 99,
                    "class_name": "target",
                    "duration_frames": self.durations[tid],
                })
        return active

    def _register(self, centroid: Tuple[int, int], box: Tuple[int, int, int, int]) -> None:
        self.objects[self.next_id] = centroid
        self.boxes[self.next_id] = box
        self.disappeared[self.next_id] = 0
        self.durations[self.next_id] = 1
        self.next_id += 1

    def _deregister(self, tid: int) -> None:
        self.objects.pop(tid, None)
        self.boxes.pop(tid, None)
        self.disappeared.pop(tid, None)
        self.durations.pop(tid, None)


class ObjectTracker:
    """
    Unified Multi-Object Tracker supporting:
    1. Deep Learning tracking via YOLOv8 + ByteTrack (for semantic COCO classes).
    2. Motion Blob tracking via SuBSENSE foreground masks (for generic moving shapes).
    3. Hybrid strategy (YOLO prioritized with automatic motion blob fallback).
    """

    def __init__(self, tracking_config: TrackingConfig, detection_config: DetectionConfig, yolo_model=None):
        self.config = tracking_config
        self.detection_config = detection_config
        self.model = yolo_model
        # Mapping: track_id -> deque of (cx, cy) coordinates
        self.trajectories: Dict[int, deque] = {}
        self.track_durations: Dict[int, int] = {}
        self.blob_tracker = MotionBlobTracker(
            max_disappeared=15,
            max_distance=70.0,
            min_area=tracking_config.min_blob_area
        )

    def reset(self) -> None:
        """Resets all active trajectories, durations, and motion blob tracking state."""
        self.trajectories.clear()
        self.track_durations.clear()
        self.blob_tracker.reset()

    def track(
        self,
        frame_bgr: np.ndarray,
        fg_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[dict], Dict[int, List[Tuple[int, int]]]]:
        """
        Executes tracking according to configured tracking_mode:
        - 'yolo': Only YOLOv8 + ByteTrack
        - 'motion_blob': Only MotionBlobTracker on fg_mask
        - 'hybrid': Runs YOLO; if 0 detections or generic motion present, falls back to MotionBlobTracker
        """
        mode = self.config.tracking_mode.lower()
        active_tracks: List[dict] = []

        # 1. Run YOLO tracking if enabled
        if mode in ("yolo", "hybrid") and self.model is not None:
            active_tracks = self._run_yolo_track(frame_bgr)

        # 2. If in motion_blob mode OR in hybrid mode with 0 YOLO detections:
        if (mode == "motion_blob" or (mode == "hybrid" and len(active_tracks) == 0)) and fg_mask is not None:
            blob_tracks = self.blob_tracker.update(fg_mask)
            active_tracks = blob_tracks

        # Update trajectories for active tracks
        current_active_ids = set()
        for trk in active_tracks:
            tid = trk["track_id"]
            cx, cy = trk["centroid"]
            current_active_ids.add(tid)

            if tid not in self.trajectories:
                self.trajectories[tid] = deque(maxlen=self.config.max_history_length)
                self.track_durations[tid] = 0

            self.trajectories[tid].append((cx, cy))
            self.track_durations[tid] += 1
            trk["duration_frames"] = self.track_durations[tid]

        # Cleanup old trajectories
        if len(self.trajectories) > 200:
            inactive_ids = [tid for tid in self.trajectories if tid not in current_active_ids]
            for tid in inactive_ids[:50]:
                self.trajectories.pop(tid, None)
                self.track_durations.pop(tid, None)

        return active_tracks, self._get_trajectories_snapshot()

    def _run_yolo_track(self, frame_bgr: np.ndarray) -> List[dict]:
        """Runs Ultralytics YOLOv8 + ByteTrack."""
        results = self.model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.config.tracker_type,
            conf=self.detection_config.conf_threshold,
            classes=self.detection_config.classes,
            device=self.detection_config.device,
            verbose=False,
        )

        tracks = []
        if not results:
            return tracks

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return tracks

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, tid, conf, cls_id in zip(boxes_xyxy, track_ids, confs, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = CCTV_CLASS_NAMES.get(cls_id, str(cls_id))

            tracks.append({
                "track_id": int(tid),
                "bbox": (x1, y1, x2, y2),
                "centroid": (cx, cy),
                "conf": float(conf),
                "class_id": int(cls_id),
                "class_name": class_name,
            })
        return tracks

    def _get_trajectories_snapshot(self) -> Dict[int, List[Tuple[int, int]]]:
        return {tid: list(pts) for tid, pts in self.trajectories.items()}
