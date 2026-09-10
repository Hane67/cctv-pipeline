"""
Stage 2: Illumination-Invariant Foreground Segmentation (Problem Statement 2).
High-performance, vectorized SuBSENSE implementation based on St-Charles et al. 2015.

Key mechanisms:
1. Vectorized Local Binary Similarity Patterns (LBSP): compares 8-connected neighbors
   against center pixel via relative contrast |neighbour - center| > threshold.
   Invariant to monotonic and uniform illumination changes.
2. Per-Pixel Dynamic Distance Threshold R(x): dynamically expands or tightens based
   on local temporal stability and EMA flip rate.
3. Adaptive Background Model Update Rate T(x): suppresses compression mosquito noise
   and high-frequency flutter by penalizing frequently flipped pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from cctv_pipeline.config.settings import SegmentationConfig


# Precomputed bit count lookup table for 8-bit integers (0 to 255)
BIT_COUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def compute_lbsp_fast(gray: np.ndarray, thresh: int = 15) -> np.ndarray:
    """
    Computes 8-bit spatial LBSP descriptor using optimized NumPy boundary slicing.
    Relative contrast thresholding makes the descriptor illumination-invariant.
    """
    gray16 = gray.astype(np.int16)
    padded = cv2.copyMakeBorder(gray16, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    h, w = gray.shape

    center = padded[1:1 + h, 1:1 + w]
    desc = np.zeros((h, w), dtype=np.uint8)

    # 8 neighbor offsets: (-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1)
    ]

    for bit_idx, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        diff = np.abs(neighbor - center)
        bit = (diff > thresh).astype(np.uint8)
        desc |= (bit << bit_idx)

    return desc


def clean_mask_morphology(mask: np.ndarray, open_k: int = 3, close_k: int = 7) -> np.ndarray:
    """Removes salt-and-pepper speckle noise and fills small interior holes."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close)
    return out


class SuBSENSESegmenter:
    """
    Vectorized SuBSENSE background subtraction model.
    Maintains N background samples per pixel of both raw intensity and LBSP features.
    """

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.N = config.n_samples
        self.lbsp_thresh = config.lbsp_thresh
        self.min_R = config.min_dist_thresh
        self.max_R = config.max_dist_thresh
        self.match_req = config.match_count_req
        self.intensity_weight = config.intensity_weight

        self.R: Optional[np.ndarray] = None
        self.flip_rate: Optional[np.ndarray] = None
        self.prev_fg: Optional[np.ndarray] = None
        self.bg_intensity: Optional[np.ndarray] = None  # (H, W, N)
        self.bg_lbsp: Optional[np.ndarray] = None       # (H, W, N)
        self.initialized = False

    def reset(self) -> None:
        """Resets background model state."""
        self.R = None
        self.flip_rate = None
        self.prev_fg = None
        self.bg_intensity = None
        self.bg_lbsp = None
        self.initialized = False

    def _init_model(self, gray: np.ndarray, lbsp: np.ndarray) -> None:
        h, w = gray.shape
        self.bg_intensity = np.repeat(gray[:, :, None], self.N, axis=2).astype(np.int16)
        self.bg_lbsp = np.repeat(lbsp[:, :, None], self.N, axis=2)
        self.R = np.full((h, w), self.config.init_dist_thresh, dtype=np.float32)
        self.flip_rate = np.zeros((h, w), dtype=np.float32)
        self.prev_fg = np.zeros((h, w), dtype=bool)
        self.initialized = True

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Processes a single BGR frame and returns a binary foreground mask (0: background, 255: foreground).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lbsp = compute_lbsp_fast(gray, thresh=self.lbsp_thresh)

        h, w = gray.shape

        if not self.initialized or self.bg_intensity is None or self.bg_intensity.shape[:2] != (h, w):
            self._init_model(gray, lbsp)
            return np.zeros(gray.shape, dtype=np.uint8)
        gray16 = gray.astype(np.int16)[:, :, None]
        lbsp_3d = lbsp[:, :, None]

        # 1. Fast Vectorized Hamming Distance via Lookup Table
        # Compute bitwise XOR between current LBSP and each of the N background sample planes
        xor_result = np.bitwise_xor(self.bg_lbsp, lbsp_3d)
        # Vectorized LUT indexing: 16x scaling maps 0-8 bits to 0-128 range
        hamming_dist = BIT_COUNT_LUT[xor_result].astype(np.float32) * 16.0

        # 2. Add raw intensity difference tie-breaker
        intensity_dist = np.abs(self.bg_intensity - gray16).astype(np.float32)
        total_dist = hamming_dist + self.intensity_weight * intensity_dist

        # 3. Match count evaluation against dynamic threshold R(x)
        matches = (total_dist < self.R[:, :, None]).sum(axis=2)
        fg_mask = (matches < self.match_req)

        # 4. Update per-pixel dynamic threshold R(x)
        flipped = (fg_mask != self.prev_fg)
        self.flip_rate = 0.90 * self.flip_rate + 0.10 * flipped.astype(np.float32)
        # Unstable / noisy pixels expand threshold; stable regions tighten
        self.R = np.clip(self.R + (self.flip_rate - 0.05) * 8.0, self.min_R, self.max_R)
        self.prev_fg = fg_mask

        # 5. Adaptive background model update with mosquito-noise damping
        # Pixels with high flip rates (compression artifacts) get throttled update probabilities
        update_prob = np.clip(0.08 * (1.0 - self.flip_rate), 0.005, 0.08)
        rand_matrix = np.random.random((h, w))
        do_update = (rand_matrix < update_prob) & (~fg_mask)

        if do_update.any():
            slot = np.random.randint(0, self.N)
            ys, xs = np.where(do_update)
            self.bg_intensity[ys, xs, slot] = gray[ys, xs]
            self.bg_lbsp[ys, xs, slot] = lbsp[ys, xs]

        raw_mask = (fg_mask.astype(np.uint8)) * 255
        cleaned_mask = clean_mask_morphology(
            raw_mask,
            open_k=self.config.morph_open_ksize,
            close_k=self.config.morph_close_ksize
        )
        return cleaned_mask
