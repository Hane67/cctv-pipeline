"""Unit tests for Stage 4 Analytics (Tripwires, Heatmap, Intersections)."""
import unittest
import numpy as np

from cctv_pipeline.config.settings import TripwireConfig
from cctv_pipeline.analytics.counting import line_segments_intersect, TripwireCounter
from cctv_pipeline.analytics.heatmap import MotionHeatmap


class TestAnalytics(unittest.TestCase):

    def test_line_segments_intersect(self):
        # Vertical segment (5, 0) to (5, 10)
        A = (5, 0)
        B = (5, 10)

        # Crossing horizontal segment (0, 5) to (10, 5)
        C = (0, 5)
        D = (10, 5)
        self.assertTrue(line_segments_intersect(A, B, C, D))

        # Parallel non-intersecting segment (2, 0) to (2, 10)
        E = (2, 0)
        F = (2, 10)
        self.assertFalse(line_segments_intersect(A, B, E, F))

    def test_tripwire_counter(self):
        wire = TripwireConfig(name="Gate1", pt1=(0, 50), pt2=(100, 50))
        counter = TripwireCounter([wire])

        # Simulate track moving from (50, 40) to (50, 60) -> crosses Gate1
        trajectories = {1: [(50, 40), (50, 60)]}
        active_tracks = [{"track_id": 1, "centroid": (50, 60)}]

        counts = counter.update(active_tracks, trajectories)
        self.assertEqual(counts["Gate1"], 1)

        # Same track continues to (50, 70) -> should not count again
        trajectories[1].append((50, 70))
        counts2 = counter.update(active_tracks, trajectories)
        self.assertEqual(counts2["Gate1"], 1)

    def test_motion_heatmap(self):
        heatmap = MotionHeatmap(height=100, width=100, decay=0.9)
        heatmap.update([(50, 50)], radius=5, intensity=2.0)
        self.assertGreater(heatmap.heatmap[50, 50], 0)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        overlaid = heatmap.overlay_on_frame(frame)
        self.assertEqual(overlaid.shape, (100, 100, 3))

        # Resolution switch test
        frame_large = np.zeros((200, 200, 3), dtype=np.uint8)
        overlaid_large = heatmap.overlay_on_frame(frame_large)
        self.assertEqual(overlaid_large.shape, (200, 200, 3))

    def test_merge_overlapping_boxes(self):
        from cctv_pipeline.analytics.tracker import merge_overlapping_boxes
        # Two boxes overlapping/close: (10, 10, 30, 30) and (25, 25, 40, 40)
        boxes = [(10, 10, 30, 30), (25, 25, 40, 40)]
        merged = merge_overlapping_boxes(boxes, margin=15)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0], (10, 10, 40, 40))

        # Two clearly separated boxes: (0, 0, 10, 10) and (100, 100, 110, 110)
        boxes_sep = [(0, 0, 10, 10), (100, 100, 110, 110)]
        merged_sep = merge_overlapping_boxes(boxes_sep, margin=15)
        self.assertEqual(len(merged_sep), 2)


if __name__ == "__main__":
    unittest.main()
