"""Unit tests for Stage 2 Foreground Segmentation & CDNet Metrics."""
import unittest
import numpy as np

from cctv_pipeline.config.settings import SegmentationConfig
from cctv_pipeline.segmentation.subsense import compute_lbsp_fast, SuBSENSESegmenter
from cctv_pipeline.segmentation.metrics import SegmentationMetrics, CDNetEvaluator


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.gray = np.random.randint(50, 200, (60, 60), dtype=np.uint8)
        self.frame_bgr = np.stack([self.gray]*3, axis=-1)

    def test_compute_lbsp(self):
        desc = compute_lbsp_fast(self.gray, thresh=15)
        self.assertEqual(desc.shape, self.gray.shape)
        self.assertEqual(desc.dtype, np.uint8)

    def test_subsense_segmenter(self):
        cfg = SegmentationConfig(n_samples=5)
        segmenter = SuBSENSESegmenter(cfg)
        mask1 = segmenter.apply(self.frame_bgr)
        self.assertEqual(mask1.shape, (60, 60))
        # Feed another frame to test dynamic threshold update
        mask2 = segmenter.apply(self.frame_bgr)
        self.assertEqual(mask2.shape, (60, 60))
        self.assertTrue(segmenter.initialized)

    def test_subsense_resolution_switch(self):
        cfg = SegmentationConfig(n_samples=5)
        segmenter = SuBSENSESegmenter(cfg)
        mask1 = segmenter.apply(np.zeros((60, 80, 3), dtype=np.uint8))
        self.assertEqual(mask1.shape, (60, 80))
        # Switch resolution to (120, 160)
        mask2 = segmenter.apply(np.zeros((120, 160, 3), dtype=np.uint8))
        self.assertEqual(mask2.shape, (120, 160))

    def test_cdnet_metrics_perfect_score(self):
        # 100 pixels: 50 foreground (TP), 50 background (TN)
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[:5, :] = 255
        pred = gt.copy()

        evaluator = CDNetEvaluator()
        evaluator.update(pred, gt)
        summary = evaluator.get_summary()

        self.assertEqual(summary["Precision"], 1.0)
        self.assertEqual(summary["Recall"], 1.0)
        self.assertEqual(summary["F_Measure"], 1.0)
        self.assertEqual(summary["PWC"], 0.0)

    def test_cdnet_metrics_with_fp_fn(self):
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[:5, :] = 255  # 50 positives

        pred = np.zeros((10, 10), dtype=np.uint8)
        pred[2:7, :] = 255  # 50 positives: 30 overlap (TP), 20 outside (FP), 20 missed (FN)

        evaluator = CDNetEvaluator()
        evaluator.update(pred, gt)
        summary = evaluator.get_summary()

        self.assertAlmostEqual(summary["Precision"], 30 / 50.0, places=2)
        self.assertAlmostEqual(summary["Recall"], 30 / 50.0, places=2)
        self.assertGreater(summary["PWC"], 0.0)


if __name__ == "__main__":
    unittest.main()
