"""
Backward-compatibility shim for Stage 2 Foreground Segmentation.
Redirects imports to `cctv_pipeline.segmentation`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cctv_pipeline.segmentation.subsense import (
    compute_lbsp_fast as compute_lbsp,
    clean_mask_morphology as clean_mask,
    SuBSENSESegmenter,
)
from cctv_pipeline.segmentation.baselines import BaselineSegmenter
from cctv_pipeline.config.settings import SegmentationConfig


class SuBSENSELite(SuBSENSESegmenter):
    """Alias for backwards compatibility."""
    def __init__(self, n_samples=20, lbsp_thresh=15, init_dist_thresh=40,
                 min_dist_thresh=20, max_dist_thresh=100, match_count_req=2,
                 intensity_weight=0.05):
        cfg = SegmentationConfig(
            algorithm="subsense",
            n_samples=n_samples,
            lbsp_thresh=lbsp_thresh,
            init_dist_thresh=init_dist_thresh,
            min_dist_thresh=min_dist_thresh,
            max_dist_thresh=max_dist_thresh,
            match_count_req=match_count_req,
            intensity_weight=intensity_weight,
        )
        super().__init__(cfg)


def baseline_mog2():
    import cv2
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


def baseline_knn():
    import cv2
    return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)


__all__ = [
    "compute_lbsp",
    "clean_mask",
    "SuBSENSELite",
    "SuBSENSESegmenter",
    "baseline_mog2",
    "baseline_knn",
]
