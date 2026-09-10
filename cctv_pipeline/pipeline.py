"""
Unified Master Pipeline coordinating all 4 research stages:
Stage 1 (Adaptive Preprocessing) -> Stage 2 (Foreground Segmentation) ->
Stage 4 (Detection & Tracking & Analytics) -> Visualizer / Video Stream.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union

import cv2
import numpy as np

from cctv_pipeline.config.settings import PipelineConfig, load_config
from cctv_pipeline.core.logger import logger
from cctv_pipeline.core.video_stream import VideoStreamReader, VideoStreamWriter
from cctv_pipeline.core.visualizer import PipelineVisualizer
from cctv_pipeline.preprocessing.pipeline import AdaptivePreprocessor
from cctv_pipeline.segmentation.subsense import SuBSENSESegmenter
from cctv_pipeline.segmentation.baselines import BaselineSegmenter
from cctv_pipeline.analytics.detector import ObjectDetector
from cctv_pipeline.analytics.tracker import ObjectTracker
from cctv_pipeline.analytics.counting import TripwireCounter, resolve_tripwire_coords
from cctv_pipeline.analytics.heatmap import MotionHeatmap


class CCTVAnalyticsPipeline:
    """
    Main system coordinator executing real-time CCTV video analytics.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or load_config()

        # Initialize Stage 1
        self.preprocessor = AdaptivePreprocessor(self.config.preprocessing)

        # Initialize Stage 2
        algo = self.config.segmentation.algorithm.lower()
        if algo == "subsense":
            self.segmenter = SuBSENSESegmenter(self.config.segmentation)
        else:
            self.segmenter = BaselineSegmenter(self.config.segmentation)

        # Initialize Stage 4 (Detector + Tracker)
        self.detector = ObjectDetector(self.config.detection)
        self.tracker = ObjectTracker(
            self.config.tracking,
            self.config.detection,
            yolo_model=self.detector.model
        )

        # Initialize Analytics
        self.tripwire_counter = TripwireCounter(self.config.analytics.tripwires)
        self.heatmap: Optional[MotionHeatmap] = None

        # Initialize Visualizer
        self.visualizer = PipelineVisualizer(self.config.visualizer)

        # Telemetry State
        self.frame_idx = 0
        self.fps_rolling = 0.0
        self.last_timings_ms: Dict[str, float] = {}

    def reset(self) -> None:
        """Resets all pipeline temporal states, models, and analytics."""
        self.frame_idx = 0
        self.fps_rolling = 0.0
        self.last_timings_ms.clear()
        if hasattr(self.segmenter, "reset"):
            self.segmenter.reset()
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        if hasattr(self.tripwire_counter, "reset"):
            self.tripwire_counter.reset()
        if self.heatmap is not None:
            self.heatmap.reset()
        logger.info("CCTV Analytics Pipeline state reset.")

    def switch_segmentation_algorithm(self, algorithm: str) -> None:
        """Dynamically switches segmentation algorithm between subsense, mog2, and knn."""
        self.config.segmentation.algorithm = algorithm
        algo = algorithm.lower()
        if algo == "subsense":
            self.segmenter = SuBSENSESegmenter(self.config.segmentation)
        else:
            self.segmenter = BaselineSegmenter(self.config.segmentation)
        logger.info(f"Switched segmentation algorithm to {algorithm}")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Executes one complete cycle on a single video frame.
        Returns:
            composed_frame: The final rendered frame (quad, side-by-side, or HUD view)
            telemetry: Dict containing timings, detection counts, active tracks, tripwire counts
        """
        t_total_start = time.perf_counter()
        timings = {}
        h, w = frame_bgr.shape[:2]

        if self.heatmap is None or self.heatmap.height != h or self.heatmap.width != w:
            self.heatmap = MotionHeatmap(
                height=h,
                width=w,
                decay=self.config.analytics.heatmap_decay,
                alpha=self.config.analytics.heatmap_alpha
            )

        # Stage 1: Adaptive Preprocessing
        t0 = time.perf_counter()
        clean_frame, scene_metrics = self.preprocessor.process(frame_bgr)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Illumination-Invariant Segmentation
        t0 = time.perf_counter()
        fg_mask = self.segmenter.apply(clean_frame)
        timings["segmentation"] = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Detection & Tracking (Hybrid: YOLO + Motion Blobs)
        t0 = time.perf_counter()
        active_tracks, trajectories = self.tracker.track(clean_frame, fg_mask=fg_mask)
        timings["detection_tracking"] = (time.perf_counter() - t0) * 1000.0

        # Analytics: Line-crossing & Heatmap
        centroids = [trk["centroid"] for trk in active_tracks]
        tripwire_counts = self.tripwire_counter.update(active_tracks, trajectories, frame_size=(w, h))
        if self.config.analytics.enable_heatmap:
            self.heatmap.update(centroids)

        # Render Analytics Overlays on top of clean frame
        annotated = clean_frame.copy()

        # Optional foreground mask overlay in green
        if self.config.visualizer.show_segmentation_overlay:
            overlay_fg = np.zeros_like(annotated)
            overlay_fg[:, :, 1] = fg_mask
            cv2.addWeighted(annotated, 1.0, overlay_fg, self.config.visualizer.overlay_alpha, 0, annotated)

        # Optional heatmap overlay
        if self.config.analytics.enable_heatmap:
            annotated = self.heatmap.overlay_on_frame(annotated)

        # Draw tripwire lines (resolved to frame dimensions)
        resolved_wires = resolve_tripwire_coords(self.config.analytics.tripwires, width=w, height=h)
        annotated = self.visualizer.draw_tripwires(annotated, resolved_wires, tripwire_counts)

        # Draw tracks & trails
        annotated = self.visualizer.draw_tracking_tracks(annotated, active_tracks, trajectories)

        # Compute FPS
        elapsed_total = time.perf_counter() - t_total_start
        fps_instant = 1.0 / max(elapsed_total, 1e-4)
        self.fps_rolling = 0.90 * self.fps_rolling + 0.10 * fps_instant if self.frame_idx > 0 else fps_instant
        timings["total_pipeline"] = elapsed_total * 1000.0
        self.last_timings_ms = timings

        # Draw HUD Panel (drawn directly in 'hud' and 'side_by_side' modes to avoid shrinking in quad view)
        view_mode = self.config.visualizer.view_mode
        if view_mode in ("hud", "side_by_side"):
            annotated = self.visualizer.draw_hud(
                annotated,
                fps=self.fps_rolling,
                timings_ms=timings,
                scene_mode=scene_metrics.scene_type,
                active_tracks=len(active_tracks),
                tripwire_counts=tripwire_counts
            )

        # Multi-view Composition
        composed = self.visualizer.compose_view(
            raw_frame=frame_bgr,
            clean_frame=clean_frame,
            fg_mask=fg_mask,
            annotated_frame=annotated,
            mode=view_mode
        )

        self.frame_idx += 1

        telemetry = {
            "frame_index": self.frame_idx,
            "fps": round(self.fps_rolling, 1),
            "scene_mode": scene_metrics.scene_type,
            "recommended_gamma": scene_metrics.recommended_gamma,
            "active_tracks": len(active_tracks),
            "tripwire_counts": tripwire_counts,
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        }

        return composed, telemetry

    def run_on_video(
        self,
        source: Union[str, int, Path],
        out_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Executes pipeline on an entire video source or webcam feed.
        """
        reader = VideoStreamReader(source, max_frames=max_frames)
        writer = None
        start_time = time.perf_counter()

        logger.info(f"Starting pipeline execution on source: {source}")

        for frame_idx, frame in reader.frames():
            composed, telemetry = self.process_frame(frame)

            if out_path and writer is None:
                ch, cw = composed.shape[:2]
                writer = VideoStreamWriter(out_path, fps=reader.fps, width=cw, height=ch)

            if writer:
                writer.write(composed)

            if frame_idx % 25 == 0:
                logger.info(f"Frame {frame_idx}: FPS {telemetry['fps']} | Tracks: {telemetry['active_tracks']} | Mode: {telemetry['scene_mode']}")

        reader.release()
        if writer:
            writer.release()

        total_time = time.perf_counter() - start_time
        summary = {
            "frames_processed": self.frame_idx,
            "total_time_s": round(total_time, 2),
            "average_fps": round(self.frame_idx / max(total_time, 1e-4), 1),
            "output_path": str(out_path) if out_path else None,
            "final_tripwire_counts": self.tripwire_counter.counts,
        }
        logger.info(f"Pipeline completed: {summary}")
        return summary
