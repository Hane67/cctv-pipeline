"""
Stage 1: Illumination normalization and scene condition analysis.
Provides dynamic CLAHE and luminance-adaptive Gamma correction.
"""
from __future__ import annotations

import cv2
import numpy as np


class SceneMetrics:
    """Stores illumination and sharpness metrics for a single video frame."""

    def __init__(
        self,
        mean_luminance: float,
        std_luminance: float,
        p10: float,
        p90: float,
        laplacian_var: float,
        scene_type: str,
        recommended_gamma: float
    ):
        self.mean_luminance = mean_luminance
        self.std_luminance = std_luminance
        self.p10 = p10
        self.p90 = p90
        self.laplacian_var = laplacian_var
        self.scene_type = scene_type
        self.recommended_gamma = recommended_gamma

    def to_dict(self) -> dict:
        return {
            "mean_luminance": round(self.mean_luminance, 2),
            "std_luminance": round(self.std_luminance, 2),
            "p10": round(self.p10, 2),
            "p90": round(self.p90, 2),
            "laplacian_var": round(self.laplacian_var, 2),
            "scene_type": self.scene_type,
            "recommended_gamma": round(self.recommended_gamma, 2),
        }


def analyze_scene_illumination(frame_bgr: np.ndarray) -> SceneMetrics:
    """
    Analyzes frame luminance distribution and sharpness to classify surveillance condition:
    - 'Night/LowLight': mean luminance < 60
    - 'OverExposed/Glare': mean luminance > 190 or p90 > 250
    - 'HighContrast': std luminance > 65 with extreme dynamic range
    - 'Normal': standard daylight illumination
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))
    p10, p90 = np.percentile(gray, [10, 90])
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Dynamic gamma calculation: target neutral midpoint around 115
    target_lum = 115.0
    safe_mean = max(mean_lum, 10.0)
    # gamma > 1.0 brightens underexposed scenes, gamma < 1.0 darkens glare
    auto_gamma = float(np.clip(np.log(safe_mean / 255.0) / np.log(target_lum / 255.0), 0.6, 2.2))

    if mean_lum < 65:
        scene_type = "Night/LowLight"
    elif mean_lum > 190 or (p90 > 250 and mean_lum > 160):
        scene_type = "OverExposed/Glare"
    elif std_lum > 65:
        scene_type = "HighContrast"
    else:
        scene_type = "Normal"

    return SceneMetrics(
        mean_luminance=mean_lum,
        std_luminance=std_lum,
        p10=p10,
        p90=p90,
        laplacian_var=lap_var,
        scene_type=scene_type,
        recommended_gamma=auto_gamma
    )


def clahe_normalize(frame_bgr: np.ndarray, clip_limit: float = 2.5, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization on the LAB color space L-channel.
    Preserves color balance while maximizing local contrast in shadowy and washed-out surveillance areas.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def gamma_correct(frame_bgr: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """
    Applies non-linear gamma curve via a precomputed 256-element Look-Up Table (LUT).
    gamma > 1.0 lifts shadow detail, gamma < 1.0 compresses high-luminance flares.
    """
    gamma = max(gamma, 0.1)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(frame_bgr, table)
