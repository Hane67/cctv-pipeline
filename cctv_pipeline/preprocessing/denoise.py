"""
Stage 1: Low-light / High-noise spatial and temporal denoising.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise_spatial(
    frame_bgr: np.ndarray,
    h: float = 7.0,
    h_color: float = 7.0,
    template_window: int = 7,
    search_window: int = 21
) -> np.ndarray:
    """
    Non-Local Means Denoising for color frames.
    Particularly effective at suppressing sensor gain noise in CCTV night footage.
    """
    return cv2.fastNlMeansDenoisingColored(
        frame_bgr,
        None,
        h=h,
        hColor=h_color,
        templateWindowSize=template_window,
        searchWindowSize=search_window
    )


def denoise_fast_bilateral(frame_bgr: np.ndarray, d: int = 7, sigma: float = 30.0) -> np.ndarray:
    """
    Lightweight, fast bilateral filter for real-time edge-preserving noise reduction.
    """
    return cv2.bilateralFilter(frame_bgr, d=d, sigmaColor=sigma, sigmaSpace=sigma)
