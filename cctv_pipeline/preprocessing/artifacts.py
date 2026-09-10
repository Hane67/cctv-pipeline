"""
Stage 1: Debanding (AdaDeband-style) and Deblocking (VRCNN/Bilateral).
Mitigates H.264 / H.265 compression artifacts before feature extraction and segmentation.
"""
from __future__ import annotations

import cv2
import numpy as np


def deband_adaptive(frame_bgr: np.ndarray, flat_thresh: float = 8.0, blur_ksize: int = 9) -> np.ndarray:
    """
    Adaptive debanding algorithm (AdaDeband emulation):
    1. Computes local gradient magnitude using a 2nd-order Laplacian operator.
    2. Identifies flat-but-stepped regions (candidate color banding from coarse quantization).
    3. Masks true semantic edges (people, cars, obstacles) so they remain crisp.
    4. Selectively blurs candidate banding regions with a bilateral/Gaussian low-pass kernel.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    grad_mag = np.abs(grad)

    # Candidate banding: low gradient magnitude but not pitch black / flat void
    banding_mask = ((grad_mag < flat_thresh) & (gray > 8)).astype(np.uint8) * 255
    banding_mask = cv2.GaussianBlur(banding_mask, (5, 5), 0)

    # Ensure odd kernel size for blur
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)

    mask3 = (cv2.cvtColor(banding_mask, cv2.COLOR_GRAY2BGR).astype(np.float32)) / 255.0
    out = frame_bgr.astype(np.float32) * (1.0 - mask3) + blurred.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def deblock_bilateral(
    frame_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Edge-preserving deblocking filter:
    Smooths high-frequency 8x8 and 16x16 macroblock boundary discontinuities produced by
    heavy H.264/H.265 DCT quantization while preserving high-contrast object boundaries.
    """
    return cv2.bilateralFilter(frame_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
