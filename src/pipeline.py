"""
Backward-compatibility runner for full pipeline.
Redirects to `cctv_pipeline.pipeline.CCTVAnalyticsPipeline`.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cctv_pipeline.pipeline import CCTVAnalyticsPipeline
from cctv_pipeline.config.settings import load_config
import argparse


def run_pipeline(source, out_path, weights="yolov8n.pt", conf=0.35,
                  show_segmentation_overlay=True, max_frames=None):
    cfg = load_config()
    cfg.detection.weights = weights
    cfg.detection.conf_threshold = conf
    cfg.visualizer.show_segmentation_overlay = show_segmentation_overlay

    pipeline = CCTVAnalyticsPipeline(cfg)
    return pipeline.run_on_video(source, out_path=out_path, max_frames=max_frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Video path or webcam index (e.g. 0)")
    ap.add_argument("--out", default="../outputs/annotated.mp4")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    stats = run_pipeline(
        args.source, args.out, weights=args.weights, conf=args.conf,
        show_segmentation_overlay=not args.no_overlay, max_frames=args.max_frames,
    )
    print(stats)
