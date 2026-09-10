"""Unit tests for Stage 1 Preprocessing."""
import unittest
import numpy as np

from cctv_pipeline.config.settings import PreprocessingConfig
from cctv_pipeline.preprocessing.illumination import (
    analyze_scene_illumination,
    clahe_normalize,
    gamma_correct,
)
from cctv_pipeline.preprocessing.artifacts import deband_adaptive, deblock_bilateral
from cctv_pipeline.preprocessing.pipeline import AdaptivePreprocessor


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 100x100 3-channel test image
        self.frame = np.full((100, 100, 3), 120, dtype=np.uint8)
        # Add some variation
        self.frame[30:70, 30:70] = [200, 50, 50]

    def test_analyze_scene_illumination_normal(self):
        metrics = analyze_scene_illumination(self.frame)
        self.assertIn(metrics.scene_type, ["Normal", "HighContrast"])
        self.assertGreater(metrics.mean_luminance, 0)
        self.assertGreater(metrics.recommended_gamma, 0.5)

    def test_analyze_scene_illumination_dark(self):
        dark_frame = np.full((100, 100, 3), 25, dtype=np.uint8)
        metrics = analyze_scene_illumination(dark_frame)
        self.assertEqual(metrics.scene_type, "Night/LowLight")
        self.assertGreater(metrics.recommended_gamma, 1.0)

    def test_clahe_normalize(self):
        out = clahe_normalize(self.frame, clip_limit=2.0)
        self.assertEqual(out.shape, self.frame.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_gamma_correct(self):
        brightened = gamma_correct(self.frame, gamma=2.0)
        self.assertEqual(brightened.shape, self.frame.shape)
        self.assertGreaterEqual(np.mean(brightened), np.mean(self.frame) - 5)

    def test_deband_and_deblock(self):
        debanded = deband_adaptive(self.frame, flat_thresh=8.0)
        self.assertEqual(debanded.shape, self.frame.shape)
        deblocked = deblock_bilateral(debanded, d=3)
        self.assertEqual(deblocked.shape, self.frame.shape)

    def test_adaptive_preprocessor_pipeline(self):
        cfg = PreprocessingConfig()
        preprocessor = AdaptivePreprocessor(cfg)
        clean, metrics = preprocessor.process(self.frame)
        self.assertEqual(clean.shape, self.frame.shape)
        self.assertIsNotNone(metrics)
        self.assertIn("total_preprocessing", preprocessor.last_timings_ms)


if __name__ == "__main__":
    unittest.main()
