"""
Backward-compatibility shim for Stage 3 Compression Sweep.
Redirects imports to `cctv_pipeline.compression`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cctv_pipeline.compression.encoder import encode_at_qp
from cctv_pipeline.compression.sweep import CompressionSweeper
from cctv_pipeline.compression.metrics import find_operational_knee_point as find_knee_point
from cctv_pipeline.compression.roi_coding import synthesize_roi_video


def run_qp_sweep(input_path, out_dir, qp_levels=(18, 22, 26, 30, 34, 38, 42, 46, 51)):
    sweeper = CompressionSweeper()
    return sweeper.run_sweep(input_path, out_dir, qp_levels=list(qp_levels))


def detection_retention_curve(model, encoded_by_qp, reference_qp, conf=0.35, classes=None):
    sweeper = CompressionSweeper(reference_qp=reference_qp)
    counts, retention, _ = sweeper.evaluate_retention(model, encoded_by_qp, classes=classes, conf=conf)
    return counts, retention


def roi_encode(input_path, output_path, roi_mask_path=None, roi_qp=20, bg_qp=38, codec="libx265"):
    from cctv_pipeline.segmentation.subsense import SuBSENSESegmenter
    from cctv_pipeline.config.settings import SegmentationConfig
    seg = SuBSENSESegmenter(SegmentationConfig())
    synthesize_roi_video(input_path, output_path, segmenter=seg)
    return output_path


__all__ = [
    "encode_at_qp",
    "run_qp_sweep",
    "detection_retention_curve",
    "find_knee_point",
    "roi_encode",
]
