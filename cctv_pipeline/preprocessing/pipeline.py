"""
Stage 1: Preprocessing Pipeline Orchestrator.
Combines illumination normalization, debanding, deblocking, and denoising.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from cctv_pipeline.config.settings import PreprocessingConfig
from cctv_pipeline.preprocessing.illumination import (
    SceneMetrics,
    analyze_scene_illumination,
    clahe_normalize,
    gamma_correct,
)
from cctv_pipeline.preprocessing.artifacts import deband_adaptive, deblock_bilateral
from cctv_pipeline.preprocessing.denoise import denoise_spatial, denoise_fast_bilateral


class AdaptivePreprocessor:
    """
    Orchestrates the entire Stage-1 enhancement pipeline with dynamic auto-tuning.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.last_metrics: Optional[SceneMetrics] = None
        self.last_timings_ms: Dict[str, float] = {}

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, SceneMetrics]:
        """
        Executes Stage-1 chain:
        1. Scene Condition Analysis (Daylight, Night, Glare, High Contrast).
        2. Illumination Normalization (CLAHE + Dynamic/Configured Gamma).
        3. Compression Debanding (AdaDeband-style selective blur).
        4. Compression Deblocking (VRCNN edge-preserving bilateral stand-in).
        5. Denoising (automatically triggered in night scenes or when forced).
        """
        timings = {}
        t_start = time.perf_counter()

        # Step 1: Scene metrics & dynamic gamma calculation
        t0 = time.perf_counter()
        metrics = analyze_scene_illumination(frame_bgr)
        self.last_metrics = metrics
        timings["scene_analysis"] = (time.perf_counter() - t0) * 1000.0

        out = frame_bgr

        # Step 2: CLAHE
        if self.config.enable_clahe:
            t0 = time.perf_counter()
            out = clahe_normalize(
                out,
                clip_limit=self.config.clahe_clip_limit,
                tile_grid=self.config.clahe_tile_grid
            )
            timings["clahe"] = (time.perf_counter() - t0) * 1000.0

        # Step 3: Gamma correction
        if self.config.enable_gamma:
            t0 = time.perf_counter()
            gamma = metrics.recommended_gamma if self.config.auto_gamma else self.config.gamma_value
            out = gamma_correct(out, gamma=gamma)
            timings["gamma"] = (time.perf_counter() - t0) * 1000.0

        # Step 4: Debanding
        if self.config.enable_deband:
            t0 = time.perf_counter()
            out = deband_adaptive(
                out,
                flat_thresh=self.config.deband_flat_thresh,
                blur_ksize=self.config.deband_blur_ksize
            )
            timings["deband"] = (time.perf_counter() - t0) * 1000.0

        # Step 5: Deblocking
        if self.config.enable_deblock:
            t0 = time.perf_counter()
            out = deblock_bilateral(
                out,
                d=self.config.deblock_d,
                sigma_color=self.config.deblock_sigma_color,
                sigma_space=self.config.deblock_sigma_space
            )
            timings["deblock"] = (time.perf_counter() - t0) * 1000.0

        # Step 6: Denoising (auto-enabled if night/low-light or explicitly set)
        should_denoise = self.config.enable_denoise or (
            self.config.auto_night_denoise and metrics.scene_type == "Night/LowLight"
        )
        if should_denoise:
            t0 = time.perf_counter()
            # Use fast bilateral for high FPS performance, fallback to NLM if requested
            out = denoise_fast_bilateral(out, d=5, sigma=self.config.denoise_h)
            timings["denoise"] = (time.perf_counter() - t0) * 1000.0

        timings["total_preprocessing"] = (time.perf_counter() - t_start) * 1000.0
        self.last_timings_ms = timings

        return out, metrics
