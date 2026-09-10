"""
Stage 4: Motion and Pedestrian Density Heatmap Accumulator.
Accumulates spatial occupancy over time with exponential temporal decay.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np


class MotionHeatmap:
    """
    Accumulates tracking centroids into an intensity heatmap with temporal decay.
    """

    def __init__(self, height: int, width: int, decay: float = 0.995, alpha: float = 0.4):
        self.height = height
        self.width = width
        self.decay = decay
        self.alpha = alpha
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def reset(self) -> None:
        """Resets the accumulated heatmap to zero."""
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, centroids: List[Tuple[int, int]], radius: int = 15, intensity: float = 1.0) -> None:
        """
        Decays existing heat and splats a Gaussian kernel at each object centroid.
        """
        self.heatmap *= self.decay

        for cx, cy in centroids:
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(self.heatmap, (cx, cy), radius, intensity, -1)

    def overlay_on_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Blends the colorized jet heatmap on top of the input surveillance frame.
        """
        h, w = frame_bgr.shape[:2]
        if (h, w) != (self.height, self.width):
            self.height, self.width = h, w
            self.heatmap = cv2.resize(self.heatmap, (w, h))

        if self.heatmap.max() > 0:
            norm_heat = np.clip((self.heatmap / max(self.heatmap.max(), 1.0)) * 255.0, 0, 255).astype(np.uint8)
            color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
            return cv2.addWeighted(frame_bgr, 1.0 - self.alpha, color_heat, self.alpha, 0)
        return frame_bgr
