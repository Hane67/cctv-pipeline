"""Unit tests for Stage 3 Compression Metrics and Knee-Point Discovery."""
import unittest

from cctv_pipeline.compression.metrics import find_operational_knee_point


class TestCompression(unittest.TestCase):

    def test_find_operational_knee_point(self):
        # Retention drops below 0.90 at QP 34
        retention = {
            18: 1.0,
            22: 0.98,
            26: 0.96,
            30: 0.92,
            34: 0.86,
            38: 0.75,
            42: 0.50,
        }
        knee = find_operational_knee_point(retention, threshold=0.90)
        self.assertEqual(knee, 34)

    def test_find_operational_knee_point_never_drops(self):
        retention = {
            18: 1.0,
            22: 0.99,
            26: 0.98,
        }
        knee = find_operational_knee_point(retention, threshold=0.90)
        self.assertEqual(knee, 26)


if __name__ == "__main__":
    unittest.main()
