"""
Stage 3: Video Compression Engine.
Encodes video across constant QP levels using FFmpeg with automatic OpenCV fallback.
Computes PSNR, SSIM, and bitrate metrics.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from cctv_pipeline.core.logger import logger


def get_ffmpeg_exe() -> Optional[str]:
    """Finds ffmpeg executable in system PATH or via bundled imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def has_ffmpeg() -> bool:
    """Checks if ffmpeg executable is accessible in system PATH or bundled imageio-ffmpeg."""
    return get_ffmpeg_exe() is not None


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio between two frames."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes fast luminance SSIM index between two frames."""
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def encode_at_qp(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    qp: int,
    codec: str = "libx265"
) -> Path:
    """
    Encodes video at a constant QP level.
    If ffmpeg is available (system PATH or imageio-ffmpeg), runs libx265/libx264 with -qp parameter.
    If ffmpeg is absent, uses OpenCV VideoWriter with DCT quantization simulation.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = get_ffmpeg_exe()
    if ffmpeg_exe is not None:
        # Native FFmpeg Constant QP encoding (supports libx265, libx264)
        cmd = [
            ffmpeg_exe, "-y", "-i", str(input_path),
            "-c:v", codec,
        ]
        if codec in ("libx265", "hevc"):
            cmd.extend(["-x265-params", f"qp={qp}"])
        else:
            cmd.extend(["-qp", str(qp)])

        cmd.extend(["-an", str(out_file)])
        subprocess.run(cmd, check=True, capture_output=True)
        return out_file

    # Fallback: OpenCV simulated quantization
    logger.warning(f"ffmpeg not found. Simulating QP={qp} compression via OpenCV DCT quantization...")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use MJPG or mp4v
    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Simulated compression: quantization step Q = 2^((qp - 4)/6)
    q_factor = max(1, int(2 ** ((qp - 12) / 8.0)))
    # JPEG quality factor mapped from QP: QP 18 -> quality 95, QP 51 -> quality 10
    jpeg_quality = int(np.clip(100 - (qp - 18) * (90 / 33.0), 5, 98))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Simulate compression macroblocking & high frequency loss via JPEG quantization
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode(".jpg", frame, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        writer.write(decimg)

    cap.release()
    writer.release()
    return out_file


def measure_video_quality(
    reference_video: Union[str, Path],
    compressed_video: Union[str, Path],
    sample_frames: int = 50
) -> Dict[str, float]:
    """
    Computes average PSNR, SSIM, bitrate (kbps), and file size of a compressed video against a reference.
    """
    cap_ref = cv2.VideoCapture(str(reference_video))
    cap_cmp = cv2.VideoCapture(str(compressed_video))

    psnr_vals = []
    ssim_vals = []
    count = 0

    while count < sample_frames:
        ok1, f_ref = cap_ref.read()
        ok2, f_cmp = cap_cmp.read()
        if not ok1 or not ok2:
            break

        psnr_vals.append(compute_psnr(f_ref, f_cmp))
        ssim_vals.append(compute_ssim_simple(f_ref, f_cmp))
        count += 1

    cap_ref.release()
    cap_cmp.release()

    # File size and bitrate
    cmp_path = Path(compressed_video)
    file_bytes = cmp_path.stat().st_size if cmp_path.exists() else 0
    fps = cap_ref.get(cv2.CAP_PROP_FPS) or 25.0
    duration_s = max(count / fps, 0.1)
    bitrate_kbps = (file_bytes * 8.0) / (duration_s * 1000.0)

    return {
        "avg_psnr": round(float(np.mean(psnr_vals)) if psnr_vals else 0.0, 2),
        "avg_ssim": round(float(np.mean(ssim_vals)) if ssim_vals else 0.0, 4),
        "file_size_kb": round(file_bytes / 1024.0, 2),
        "bitrate_kbps": round(bitrate_kbps, 2),
    }
