"""
Backward-compatibility shim for Stage 1 Preprocessing.
Redirects imports to `cctv_pipeline.preprocessing`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cctv_pipeline.preprocessing.illumination import (
    clahe_normalize,
    gamma_correct,
    analyze_scene_illumination,
)
from cctv_pipeline.preprocessing.artifacts import (
    deband_adaptive as deband_lite,
    deblock_bilateral as deblock_lite,
)
from cctv_pipeline.preprocessing.denoise import denoise_spatial as denoise_night
from cctv_pipeline.config.settings import PreprocessingConfig
from cctv_pipeline.preprocessing.pipeline import AdaptivePreprocessor


def preprocess_frame(frame_bgr, do_denoise=False):
    """Executes the standard Stage-1 chain."""
    cfg = PreprocessingConfig(enable_denoise=do_denoise)
    preprocessor = AdaptivePreprocessor(cfg)
    clean, _ = preprocessor.process(frame_bgr)
    return clean


__all__ = [
    "clahe_normalize",
    "gamma_correct",
    "denoise_night",
    "deband_lite",
    "deblock_lite",
    "preprocess_frame",
    "analyze_scene_illumination",
]
