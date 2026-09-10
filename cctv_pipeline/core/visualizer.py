"""
Visualization engine: Multi-view screen composition, tracking overlays,
tripwire indicators, trajectory trails, and telemetry HUD.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cctv_pipeline.config.settings import VisualizerConfig


class PipelineVisualizer:
    """
    Renders telemetry HUD, tracking trajectories, tripwires, and composes multi-view frames.
    """

    def __init__(self, config: VisualizerConfig):
        self.config = config

    def draw_hud(
        self,
        canvas: np.ndarray,
        fps: float,
        timings_ms: Dict[str, float],
        scene_mode: str = "Normal",
        active_tracks: int = 0,
        tripwire_counts: Dict[str, int] = None
    ) -> np.ndarray:
        """Renders an informative semi-transparent telemetry HUD on the top-left."""
        if not self.config.show_hud_stats:
            return canvas

        h, w = canvas.shape[:2]
        font_scale = 0.42 if w >= 640 else 0.32
        line_h = 16 if w >= 640 else 13
        pad = 6
        box_w = min(230, int(w * 0.48))

        lines = [
            f"FPS: {fps:4.1f} | Mode: {scene_mode}",
            f"Preprocess: {timings_ms.get('preprocess', 0):4.1f} ms",
            f"Segment:    {timings_ms.get('segmentation', 0):4.1f} ms",
            f"Detect+Trk: {timings_ms.get('detection_tracking', 0):4.1f} ms",
            f"Active Targets: {active_tracks}",
        ]
        if tripwire_counts:
            for name, count in tripwire_counts.items():
                lines.append(f"{name}: {count}")

        box_h = pad * 2 + len(lines) * line_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), self.config.hud_bg_color, -1)
        cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)
        cv2.rectangle(canvas, (8, 8), (8 + box_w, 8 + box_h), (70, 70, 70), 1)

        for i, line in enumerate(lines):
            y = 8 + pad + (i + 1) * line_h - 3
            color = (0, 255, 180) if i == 0 else (220, 220, 220)
            cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        return canvas

    def draw_tracking_tracks(
        self,
        canvas: np.ndarray,
        tracks: List[dict],
        trajectories: Dict[int, List[Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Draws bounding boxes, IDs, labels, and fading trajectory trails.
        """
        # Draw trajectories first (smooth, fading, recent 15 points for currently active tracks)
        if self.config.show_trajectories:
            active_ids = {trk["track_id"] for trk in tracks}
            for tid in active_ids:
                if tid not in trajectories:
                    continue
                points = trajectories[tid]
                recent_pts = points[-15:]
                if len(recent_pts) < 2:
                    continue
                for i in range(1, len(recent_pts)):
                    p_start = recent_pts[i - 1]
                    p_end = recent_pts[i]

                    # Ignore jump lines caused by video loops or sudden occlusions
                    dist = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
                    if dist > 50:
                        continue

                    alpha = i / float(len(recent_pts))
                    thickness = 2 if alpha > 0.6 else 1
                    trail_color = (0, int(190 * alpha), int(255 * alpha))
                    cv2.line(canvas, p_start, p_end, trail_color, thickness, cv2.LINE_AA)

                # Draw glowing centroid at latest position
                cv2.circle(canvas, recent_pts[-1], 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Draw bounding boxes and labels
        if self.config.show_boxes:
            for trk in tracks:
                x1, y1, x2, y2 = trk["bbox"]
                tid = trk["track_id"]
                label = trk.get("class_name", "target")
                conf = trk.get("conf", 0.0)

                # Choose distinctive color scheme
                if "person" in label:
                    color = (255, 170, 0)   # Blue/Cyan
                elif any(v in label for v in ("car", "truck", "bus", "motorcycle")):
                    color = (50, 220, 80)    # Green
                else:
                    color = (0, 200, 255)    # Amber/Gold

                # Draw box
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                # Format label tag
                clean_name = label.capitalize()
                tag = f"{clean_name} #{tid}"
                if conf and conf < 0.99 and any(x in label for x in ("person", "car", "truck", "bus", "bicycle", "motorcycle")):
                    tag += f" ({int(conf*100)}%)"

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                badge_y = max(y1 - th - 6, 0)
                cv2.rectangle(canvas, (x1, badge_y), (x1 + tw + 6, badge_y + th + 6), color, -1)
                cv2.putText(
                    canvas, tag, (x1 + 3, badge_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA
                )

        return canvas

    def draw_tripwires(self, canvas: np.ndarray, tripwires: List[dict], counts: Dict[str, int]) -> np.ndarray:
        """Draws virtual tripwire lines across the surveillance scene."""
        for wire in tripwires:
            name = wire["name"]
            pt1 = tuple(wire["pt1"])
            pt2 = tuple(wire["pt2"])
            color = tuple(wire.get("color", (0, 255, 255)))

            # Semi-transparent dashed/solid line
            cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

            # Center count badge
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            count = counts.get(name, 0)
            tag = f"{name}: {count}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

            badge_x = mid_x - tw // 2
            badge_y = max(mid_y - 12, th + 4)
            cv2.rectangle(canvas, (badge_x - 4, badge_y - th - 2), (badge_x + tw + 4, badge_y + 4), (20, 20, 20), -1)
            cv2.rectangle(canvas, (badge_x - 4, badge_y - th - 2), (badge_x + tw + 4, badge_y + 4), color, 1)
            cv2.putText(canvas, tag, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        return canvas

    def compose_view(
        self,
        raw_frame: np.ndarray,
        clean_frame: np.ndarray,
        fg_mask: np.ndarray,
        annotated_frame: np.ndarray,
        mode: Optional[str] = None
    ) -> np.ndarray:
        """Composes the selected multi-view layout."""
        mode = mode or self.config.view_mode
        h, w = raw_frame.shape[:2]

        if mode == "hud":
            return annotated_frame

        if mode == "mask_only":
            if len(fg_mask.shape) == 2:
                return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            return fg_mask

        if mode == "side_by_side":
            combined = np.hstack([raw_frame, annotated_frame])
            cv2.putText(combined, "ORIGINAL INPUT", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined, "PROCESSED & ANALYTICS", (w + 16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            return combined

        # Default: 'quad' (4-way split screen)
        half_w, half_h = w // 2, h // 2
        top_left = cv2.resize(raw_frame, (half_w, half_h))
        top_right = cv2.resize(clean_frame, (half_w, half_h))

        mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) if len(fg_mask.shape) == 2 else fg_mask
        bottom_left = cv2.resize(mask_bgr, (half_w, half_h))
        bottom_right = cv2.resize(annotated_frame, (half_w, half_h))

        for img, title in [
            (top_left, "1. RAW INPUT"),
            (top_right, "2. ADAPTIVE PREPROCESSING"),
            (bottom_left, "3. ILLUMINATION-INVARIANT SEGMENTATION"),
            (bottom_right, "4. DETECTION, TRACKING & ANALYTICS"),
        ]:
            cv2.putText(img, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        quad = np.vstack([top_row, bottom_row])

        # Subtle grid lines
        cv2.line(quad, (half_w, 0), (half_w, h), (50, 50, 50), 1)
        cv2.line(quad, (0, half_h), (w, half_h), (50, 50, 50), 1)
        return quad
