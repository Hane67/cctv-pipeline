"""Stage 4 Analytics, Detection, Tracking, and CDRE Package."""
from cctv_pipeline.analytics.detector import ObjectDetector, CCTV_CLASS_NAMES
from cctv_pipeline.analytics.tracker import ObjectTracker
from cctv_pipeline.analytics.counting import (
    TripwireCounter,
    line_segments_intersect,
)
from cctv_pipeline.analytics.heatmap import MotionHeatmap
from cctv_pipeline.analytics.cdre import extract_cdre_metadata

__all__ = [
    "ObjectDetector",
    "CCTV_CLASS_NAMES",
    "ObjectTracker",
    "TripwireCounter",
    "line_segments_intersect",
    "MotionHeatmap",
    "extract_cdre_metadata",
]
