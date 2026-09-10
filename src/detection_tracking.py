"""
Backward-compatibility shim for Stage 4 Detection and Tracking.
Redirects imports to `cctv_pipeline.analytics`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from cctv_pipeline.analytics.detector import CCTV_CLASS_NAMES as CCTV_CLASSES
from cctv_pipeline.analytics.cdre import extract_cdre_metadata as extract_cdre


def load_detector(weights="yolov8n.pt"):
    """Loads YOLO detector."""
    return YOLO(weights)


def track_video(model, source, tracker="bytetrack.yaml", conf=0.35, classes=None,
                 save_path=None, stream=True):
    """Runs Ultralytics tracking."""
    return model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        classes=classes,
        save=save_path is not None,
        project=str(save_path.parent) if save_path else None,
        name=save_path.stem if save_path else None,
        stream=stream,
        verbose=False,
    )


__all__ = [
    "load_detector",
    "track_video",
    "CCTV_CLASSES",
    "extract_cdre",
]
