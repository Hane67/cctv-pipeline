"""
Stage 2: Baseline background subtractors (MOG2 and KNN) for benchmarking.
"""
from __future__ import annotations

import cv2
import numpy as np

from cctv_pipeline.config.settings import SegmentationConfig
from cctv_pipeline.segmentation.subsense import clean_mask_morphology


class BaselineSegmenter:
    """
    Standardized wrapper around OpenCV MOG2 and KNN background subtractors.
    Provides identical .apply() signature to SuBSENSESegmenter.
    """

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.algorithm = config.algorithm.lower()
        self._init_model()

    def _init_model(self) -> None:
        if self.algorithm == "mog2":
            self.model = cv2.createBackgroundSubtractorMOG2(
                history=self.config.mog2_history,
                varThreshold=self.config.mog2_var_threshold,
                detectShadows=self.config.detect_shadows
            )
        elif self.algorithm == "knn":
            self.model = cv2.createBackgroundSubtractorKNN(
                history=self.config.knn_history,
                dist2Threshold=self.config.knn_dist2_threshold,
                detectShadows=self.config.detect_shadows
            )
        else:
            raise ValueError(f"Unknown baseline algorithm: {self.algorithm}")

    def reset(self) -> None:
        self._init_model()

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts foreground mask. If shadow detection is enabled, removes gray shadow pixels (127)
        and retains pure foreground (255).
        """
        raw_mask = self.model.apply(frame_bgr)
        # 127 = shadow, 255 = foreground in OpenCV subtractors
        _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        cleaned = clean_mask_morphology(
            binary_mask,
            open_k=self.config.morph_open_ksize,
            close_k=self.config.morph_close_ksize
        )
        return cleaned
