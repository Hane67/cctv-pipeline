"""Stage 3 Compression Sweep and Optimization Package."""
from cctv_pipeline.compression.encoder import (
    encode_at_qp,
    has_ffmpeg,
    compute_psnr,
    compute_ssim_simple,
    measure_video_quality,
)
from cctv_pipeline.compression.sweep import CompressionSweeper
from cctv_pipeline.compression.roi_coding import synthesize_roi_video
from cctv_pipeline.compression.metrics import (
    find_operational_knee_point,
    plot_compression_analysis,
)

__all__ = [
    "encode_at_qp",
    "has_ffmpeg",
    "compute_psnr",
    "compute_ssim_simple",
    "measure_video_quality",
    "CompressionSweeper",
    "synthesize_roi_video",
    "find_operational_knee_point",
    "plot_compression_analysis",
]
