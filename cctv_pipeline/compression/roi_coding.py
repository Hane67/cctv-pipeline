"""
Stage 3: Video Coding for Machines (VCM) Region-of-Interest (ROI) Encoding.
Allocates high bitrate / fine QP to moving foreground objects and aggressively quantizes static background.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np

from cctv_pipeline.core.logger import logger
from cctv_pipeline.compression.encoder import has_ffmpeg


def synthesize_roi_video(
    source_video: Union[str, Path],
    output_video: Union[str, Path],
    segmenter,
    roi_jpeg_qual: int = 95,
    bg_jpeg_qual: int = 20,
    max_frames: int = 200
) -> Dict[str, float]:
    """
    Simulates VCM ROI encoding:
    Uses the SuBSENSE foreground segmentation mask to composite high-fidelity foreground
    over heavily quantized background, dramatically reducing bitrate while preserving detector accuracy.
    """
    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # 1. Obtain foreground mask
        fg_mask = segmenter.apply(frame)
        # Dilate slightly so object boundaries aren't clipped by the quantization seam
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        roi_mask = cv2.dilate(fg_mask, kernel)
        roi_weight = (roi_mask.astype(np.float32) / 255.0)[:, :, None]

        # 2. Simulate high-fidelity foreground
        _, enc_roi = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), roi_jpeg_qual])
        dec_roi = cv2.imdecode(enc_roi, 1)

        # 3. Simulate low-bitrate background
        _, enc_bg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), bg_jpeg_qual])
        dec_bg = cv2.imdecode(enc_bg, 1)

        # 4. Composite
        blended = (dec_roi.astype(np.float32) * roi_weight + dec_bg.astype(np.float32) * (1.0 - roi_weight))
        writer.write(np.clip(blended, 0, 255).astype(np.uint8))
        frame_idx += 1

    cap.release()
    writer.release()

    orig_bytes = Path(source_video).stat().st_size
    roi_bytes = out_path.stat().st_size
    bandwidth_saved_pct = max(0.0, 100.0 * (1.0 - (roi_bytes / max(orig_bytes, 1))))

    return {
        "frames_encoded": frame_idx,
        "original_size_kb": round(orig_bytes / 1024.0, 2),
        "roi_size_kb": round(roi_bytes / 1024.0, 2),
        "bandwidth_saved_pct": round(bandwidth_saved_pct, 2),
    }
