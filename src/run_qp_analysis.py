"""
Backward-compatibility runner for Problem Statement 3.
Redirects to `cctv_pipeline.cli`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cctv_pipeline.cli import cli_sweep
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", default="../outputs/qp_analysis")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--reference-qp", type=int, default=18)
    ap.add_argument("--qp-levels", type=int, nargs="+", default=[18, 22, 26, 30, 34, 38, 42, 46, 51])
    ap.add_argument("--codec", default="libx265")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cli_sweep(args)
