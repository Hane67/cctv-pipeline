"""
Video stream capture, synthetic generator, and video file writer utilities.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

from cctv_pipeline.core.logger import logger


class VideoStreamReader:
    """
    Robust video reader supporting local files, webcams, RTSP streams, and looping.
    """

    def __init__(self, source: Union[str, int, Path], loop: bool = False, max_frames: Optional[int] = None):
        self.source = source
        self.loop = loop
        self.max_frames = max_frames
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_webcam = str(source).isdigit() or isinstance(source, int)
        self.frame_count = 0
        self._init_capture()

    def _init_capture(self) -> None:
        src = int(self.source) if self.is_webcam else str(self.source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else -1
        logger.info(f"Opened video source {self.source} ({self.width}x{self.height} @ {self.fps:.1f} FPS, {self.total_frames} frames)")

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding (frame_index, frame_bgr) tuples."""
        frame_idx = 0
        while True:
            if self.max_frames and frame_idx >= self.max_frames:
                break

            ok, frame = self.cap.read()
            if not ok:
                if self.loop and not self.is_webcam and self.total_frames > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if not ok:
                        break
                else:
                    break

            yield frame_idx, frame
            frame_idx += 1

        self.release()

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoStreamWriter:
    """
    Video writer with multi-codec fallback and directory creation.
    """

    def __init__(self, out_path: Union[str, Path], fps: float, width: int, height: int, codec: str = "mp4v"):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.out_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            # Fallback to XVID or MJPG if mp4v fails
            fallback_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            logger.warning(f"Codec {codec} failed; falling back to MJPG for {self.out_path}")
            self.writer = cv2.VideoWriter(str(self.out_path), fallback_fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.info(f"Saved output video to {self.out_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def generate_synthetic_surveillance_clip(
    output_path: Union[str, Path],
    num_frames: int = 150,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    simulate_glare: bool = True,
    simulate_shadow: bool = True
) -> Path:
    """
    Generates a synthetic CCTV surveillance test clip containing:
    1. Static street/corridor background with subtle camera sensor noise.
    2. Dynamic illumination shifts: sudden shadow passage and car headlight glare.
    3. Moving synthetic targets (pedestrians and vehicles) traversing across virtual tripwires.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoStreamWriter(out_file, fps=fps, width=width, height=height)

    # Base background: a dark roadway with sidewalk
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:height // 2, :] = [60, 50, 45]       # Distant background / wall
    bg[height // 2:height * 3 // 4, :] = [40, 40, 40]  # Roadway
    bg[height * 3 // 4:, :] = [70, 70, 70]   # Sidewalk

    # Add road markings
    cv2.line(bg, (0, height * 5 // 8), (width, height * 5 // 8), (200, 200, 200), 2)
    # Add a lamp post
    cv2.rectangle(bg, (width // 4 - 5, height // 4), (width // 4 + 5, height * 3 // 4), (100, 100, 100), -1)

    for i in range(num_frames):
        frame = bg.copy().astype(np.float32)

        # 1. Illumination variation:
        # Frames 30-70: Sudden cloud shadow (luminance drops 50%)
        # Frames 90-130: Sudden car headlight glare (luminance spikes in lower center)
        illum_factor = 1.0
        if simulate_shadow and 30 <= i <= 70:
            shadow_progress = np.sin((i - 30) / 40.0 * np.pi)
            illum_factor -= 0.55 * shadow_progress

        frame *= illum_factor

        if simulate_glare and 90 <= i <= 130:
            glare_progress = np.sin((i - 90) / 40.0 * np.pi)
            # Add circular glare gradient
            glare_center = (width // 2, height // 2 + 50)
            y, x = np.ogrid[:height, :width]
            dist = np.sqrt((x - glare_center[0])**2 + (y - glare_center[1])**2)
            glare_mask = np.clip(1.0 - dist / 180.0, 0, 1) * (180.0 * glare_progress)
            frame += glare_mask[:, :, None]

        # 2. Moving Object 1: "Pedestrian" (moving left to right along sidewalk)
        ped_x = int(-40 + (width + 80) * (i / float(num_frames)))
        ped_y = int(height * 0.78)
        if 0 <= ped_x < width:
            # Draw pedestrian body & head
            cv2.rectangle(frame, (ped_x - 12, ped_y - 45), (ped_x + 12, ped_y), (30, 80, 180), -1)
            cv2.circle(frame, (ped_x, ped_y - 52), 8, (160, 190, 230), -1)

        # 3. Moving Object 2: "Vehicle" (moving right to left along roadway)
        veh_x = int(width + 80 - (width + 160) * (i / float(num_frames)))
        veh_y = int(height * 0.58)
        if 0 <= veh_x < width:
            # Draw vehicle body
            cv2.rectangle(frame, (veh_x - 50, veh_y - 30), (veh_x + 50, veh_y + 15), (180, 40, 40), -1)
            cv2.rectangle(frame, (veh_x - 30, veh_y - 45), (veh_x + 25, veh_y - 30), (220, 120, 120), -1)
            # Wheels
            cv2.circle(frame, (veh_x - 30, veh_y + 15), 9, (20, 20, 20), -1)
            cv2.circle(frame, (veh_x + 30, veh_y + 15), 9, (20, 20, 20), -1)

        # 4. Sensor noise
        noise = np.random.normal(0, 3.5, frame.shape)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    logger.info(f"Generated synthetic surveillance video with illumination changes at {out_file}")
    return out_file
