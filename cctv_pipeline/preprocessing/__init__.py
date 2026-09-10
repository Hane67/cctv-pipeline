"""Stage 1 Adaptive Preprocessing Package."""
from cctv_pipeline.preprocessing.illumination import (
    SceneMetrics,
    analyze_scene_illumination,
    clahe_normalize,
    gamma_correct,
)
from cctv_pipeline.preprocessing.artifacts import deband_adaptive, deblock_bilateral
from cctv_pipeline.preprocessing.denoise import denoise_spatial, denoise_fast_bilateral
from cctv_pipeline.preprocessing.pipeline import AdaptivePreprocessor

__all__ = [
    "SceneMetrics",
    "analyze_scene_illumination",
    "clahe_normalize",
    "gamma_correct",
    "deband_adaptive",
    "deblock_bilateral",
    "denoise_spatial",
    "denoise_fast_bilateral",
    "AdaptivePreprocessor",
]
