"""
Stage 4: Object Detection Wrapper.
Encapsulates Ultralytics YOLOv8 with class filtering, confidence thresholding, and device selection.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from cctv_pipeline.config.settings import DetectionConfig
from cctv_pipeline.core.logger import logger

CCTV_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class ObjectDetector:
    """
    YOLOv8 object detector for surveillance applications.
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model with weights: {self.config.weights} on device: {self.config.device}")
            self.model = YOLO(self.config.weights)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        """
        Runs object detection on a single frame.
        Returns a list of detected objects:
        [{"bbox": (x1, y1, x2, y2), "conf": float, "class_id": int, "class_name": str}]
        """
        if self.model is None:
            return []

        results = self.model.predict(
            source=frame_bgr,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            device=self.config.device,
            verbose=False,
        )

        detections = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_name = CCTV_CLASS_NAMES.get(cls_id, str(cls_id))
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "class_id": int(cls_id),
                "class_name": class_name,
            })

        return detections
