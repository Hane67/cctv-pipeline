"""
Unified Command-Line Interface (CLI) for CCTV Video Analytics Pipeline.
Provides commands: run, ui, sweep, evaluate, and demo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cctv_pipeline.config.settings import load_config
from cctv_pipeline.core.logger import logger
from cctv_pipeline.core.video_stream import generate_synthetic_surveillance_clip
from cctv_pipeline.pipeline import CCTVAnalyticsPipeline
from cctv_pipeline.compression.sweep import CompressionSweeper
from cctv_pipeline.compression.metrics import find_operational_knee_point, plot_compression_analysis
from cctv_pipeline.analytics.detector import ObjectDetector
from cctv_pipeline.ui.web_app import launch_web_dashboard


def cli_run(args: argparse.Namespace) -> None:
    """Executes surveillance analytics on a video stream."""
    cfg = load_config(args.config)
    if args.view:
        cfg.visualizer.view_mode = args.view
    if args.algorithm:
        cfg.segmentation.algorithm = args.algorithm

    pipeline = CCTVAnalyticsPipeline(cfg)
    summary = pipeline.run_on_video(
        source=args.source,
        out_path=args.out,
        max_frames=args.max_frames
    )
    print("\n--- Pipeline Run Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cli_ui(args: argparse.Namespace) -> None:
    """Launches the interactive Web Dashboard."""
    launch_web_dashboard(host=args.host, port=args.port, source=args.source)


def cli_sweep(args: argparse.Namespace) -> None:
    """Runs Stage 3 constant-QP compression sweep."""
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweeper = CompressionSweeper(codec=args.codec, reference_qp=args.reference_qp)
    qp_levels = args.qp_levels or cfg.compression.sweep_qp_levels

    logger.info(f"Running QP sweep across levels: {qp_levels}")
    encoded = sweeper.run_sweep(args.source, out_dir / "encoded", qp_levels=qp_levels)

    detector = ObjectDetector(cfg.detection)
    counts, retention, quality = sweeper.evaluate_retention(
        detector.model,
        encoded,
        classes=cfg.detection.classes,
        conf=cfg.detection.conf_threshold
    )
    knee = find_operational_knee_point(retention, threshold=cfg.compression.knee_retention_target)

    print("\n--- QP Compression Sweep Results ---")
    print(f"Operational Knee Point: QP {knee} (Retention drops below {int(cfg.compression.knee_retention_target*100)}%)")
    for qp in sorted(retention.keys()):
        psnr = quality.get(qp, {}).get("avg_psnr", 0.0)
        kbps = quality.get(qp, {}).get("bitrate_kbps", 0.0)
        print(f"  QP {qp:2d} | Detections: {counts[qp]:3d} | Retention: {retention[qp]*100:5.1f}% | PSNR: {psnr:4.1f} dB | Bitrate: {kbps:6.1f} kbps")

    plot_path = out_dir / "qp_retention_curve.png"
    plot_compression_analysis(retention, quality, knee, plot_path)
    print(f"\nSaved analysis plot to: {plot_path}")


def cli_demo(args: argparse.Namespace) -> None:
    """Generates synthetic surveillance clip and runs live demonstration."""
    demo_clip = Path("data/synthetic_demo.mp4")
    out_clip = Path("outputs/demo_annotated.mp4")
    print(f"1. Generating synthetic CCTV surveillance video at {demo_clip}...")
    generate_synthetic_surveillance_clip(demo_clip, num_frames=120)

    print(f"2. Executing CCTV Analytics Pipeline on generated clip...")
    cfg = load_config()
    cfg.visualizer.view_mode = "quad"
    pipeline = CCTVAnalyticsPipeline(cfg)
    summary = pipeline.run_on_video(demo_clip, out_path=out_clip)

    print("\n--- Demo Complete ---")
    print(f"Annotated video saved to: {out_clip}")
    print(f"Average FPS: {summary['average_fps']}")
    print(f"Frames processed: {summary['frames_processed']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cctv-pipeline",
        description="Robust CCTV Video Analytics Platform Under Illumination Changes and Compression Artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    p_run = subparsers.add_parser("run", help="Run end-to-end pipeline on video or webcam")
    p_run.add_argument("--source", required=True, help="Path to video file or camera index (e.g. 0)")
    p_run.add_argument("--out", default="outputs/annotated.mp4", help="Path to save annotated video")
    p_run.add_argument("--config", default=None, help="Path to custom YAML configuration file")
    p_run.add_argument("--view", choices=["quad", "side_by_side", "hud", "mask_only"], default=None, help="Display layout")
    p_run.add_argument("--algorithm", choices=["subsense", "mog2", "knn"], default=None, help="Foreground segmentation algorithm")
    p_run.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")

    # Command: ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive Web Dashboard")
    p_ui.add_argument("--source", default="data/synthetic_test.mp4", help="Default video source")
    p_ui.add_argument("--port", type=int, default=5000, help="Web server port")
    p_ui.add_argument("--host", default="0.0.0.0", help="Web server host")

    # Command: sweep
    p_sw = subparsers.add_parser("sweep", help="Run Stage 3 QP compression sweep & knee-point analysis")
    p_sw.add_argument("--source", required=True, help="Path to video clip")
    p_sw.add_argument("--out-dir", default="outputs/qp_analysis", help="Directory to save compressed clips and plots")
    p_sw.add_argument("--qp-levels", type=int, nargs="+", default=[18, 22, 26, 30, 34, 38, 42, 46, 51])
    p_sw.add_argument("--reference-qp", type=int, default=18)
    p_sw.add_argument("--codec", default="libx265")
    p_sw.add_argument("--config", default=None)

    # Command: demo
    p_demo = subparsers.add_parser("demo", help="Generate synthetic surveillance clip and run full pipeline")

    args = parser.parse_args()

    if args.command == "run":
        cli_run(args)
    elif args.command == "ui":
        cli_ui(args)
    elif args.command == "sweep":
        cli_sweep(args)
    elif args.command == "demo":
        cli_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
