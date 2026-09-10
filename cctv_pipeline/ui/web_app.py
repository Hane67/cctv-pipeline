"""
Interactive Flask Web Dashboard for the CCTV Video Analytics Pipeline.
Provides live MJPEG video streaming, dynamic parameter tuning, and telemetry metrics.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from cctv_pipeline.config.settings import PipelineConfig, load_config
from cctv_pipeline.core.logger import logger
from cctv_pipeline.core.video_stream import VideoStreamReader, generate_synthetic_surveillance_clip
from cctv_pipeline.pipeline import CCTVAnalyticsPipeline
from cctv_pipeline.compression.sweep import CompressionSweeper
from cctv_pipeline.compression.metrics import find_operational_knee_point, plot_compression_analysis


def create_app(
    source: Union[str, int, Path] = "data/sample_cctv_people.mp4",
    config: Optional[PipelineConfig] = None,
    loop: bool = True
) -> Flask:
    """Factory creating the Flask web application."""
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    cfg = config or load_config()
    pipeline = CCTVAnalyticsPipeline(cfg)

    # Fallback to synthetic if sample video does not exist
    src_path = Path(source)
    if not str(source).isdigit() and not src_path.exists():
        if Path("data/synthetic_test.mp4").exists():
            source = "data/synthetic_test.mp4"
        else:
            logger.info(f"Source video {source} not found; generating synthetic surveillance clip...")
            generate_synthetic_surveillance_clip(source, num_frames=120)

    # Global shared state
    state = {
        "source": source,
        "loop": loop,
        "latest_telemetry": {},
        "current_plot": None,
    }

    @app.route("/")
    def index():
        return render_template("index.html")

    def frame_generator():
        current_source = state["source"]
        while True:
            try:
                current_source = state["source"]
                reader = VideoStreamReader(current_source, loop=state["loop"])
                pipeline.reset()
                for _, frame in reader.frames():
                    if state["source"] != current_source:
                        break  # Hot-switch video source
                    composed, telemetry = pipeline.process_frame(frame)
                    state["latest_telemetry"] = telemetry

                    # JPEG encode
                    ok, buffer = cv2.imencode(".jpg", composed, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue

                    frame_bytes = buffer.tobytes()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                reader.release()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(1.0)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/api/telemetry")
    def api_telemetry():
        return jsonify(state.get("latest_telemetry", {}))

    @app.route("/api/settings", methods=["POST"])
    def api_settings():
        data = request.get_json() or {}

        # Video source switch
        if "source" in data:
            new_src = str(data["source"])
            new_val = int(new_src) if new_src.isdigit() else new_src
            if new_val != state["source"]:
                state["source"] = new_val
                pipeline.reset()
                logger.info(f"Switched web stream source to {state['source']}")

        # Tracking strategy
        if "tracking_mode" in data:
            new_mode = str(data["tracking_mode"])
            pipeline.config.tracking.tracking_mode = new_mode
            pipeline.tracker.config.tracking_mode = new_mode
            logger.info(f"Switched tracking mode to {new_mode}")

        # Preprocessing settings
        if "gamma" in data:
            pipeline.config.preprocessing.gamma_value = float(data["gamma"])
        if "auto_gamma" in data:
            pipeline.config.preprocessing.auto_gamma = bool(data["auto_gamma"])
        if "clahe_clip" in data:
            pipeline.config.preprocessing.clahe_clip_limit = float(data["clahe_clip"])
        if "deband_thresh" in data:
            pipeline.config.preprocessing.deband_flat_thresh = float(data["deband_thresh"])

        # View mode
        if "view_mode" in data:
            pipeline.config.visualizer.view_mode = str(data["view_mode"])

        # Heatmap toggle
        if "enable_heatmap" in data:
            pipeline.config.analytics.enable_heatmap = bool(data["enable_heatmap"])

        # Algorithm switch
        if "segmentation_algorithm" in data:
            new_algo = str(data["segmentation_algorithm"])
            if new_algo != pipeline.config.segmentation.algorithm:
                pipeline.switch_segmentation_algorithm(new_algo)

        return jsonify({"status": "success", "config": pipeline.config.model_dump()})

    @app.route("/api/run_sweep", methods=["POST"])
    def api_run_sweep():
        out_dir = Path("outputs/web_qp_analysis")
        out_dir.mkdir(parents=True, exist_ok=True)
        sweeper = CompressionSweeper(reference_qp=18)
        qp_levels = [18, 26, 34, 42, 50]

        logger.info("Executing QP sweep from web dashboard...")
        encoded = sweeper.run_sweep(state["source"], out_dir / "encoded", qp_levels=qp_levels)
        counts, retention, quality = sweeper.evaluate_retention(
            pipeline.detector.model,
            encoded,
            classes=pipeline.config.detection.classes,
            conf=pipeline.config.detection.conf_threshold
        )
        knee = find_operational_knee_point(retention, threshold=pipeline.config.compression.knee_retention_target)
        plot_path = out_dir / "qp_retention.png"
        plot_compression_analysis(retention, quality, knee, plot_path)
        state["current_plot"] = plot_path

        return jsonify({
            "status": "completed",
            "knee_point": knee,
            "plot_url": f"/outputs/{plot_path.name}"
        })

    @app.route("/api/generate_clip", methods=["POST"])
    def api_generate_clip():
        out_path = Path("data/synthetic_surveillance.mp4")
        generate_synthetic_surveillance_clip(out_path, num_frames=120)
        state["source"] = str(out_path)
        return jsonify({"status": "generated", "path": str(out_path)})

    @app.route("/outputs/<path:filename>")
    def serve_outputs(filename):
        out_dir = Path("outputs/web_qp_analysis").resolve()
        return send_from_directory(str(out_dir), filename)

    return app


def launch_web_dashboard(host: str = "0.0.0.0", port: int = 5000, source: str = "data/synthetic_test.mp4") -> None:
    """Launches web dashboard server."""
    app = create_app(source=source)
    logger.info(f"Starting CCTV Analytics Web Dashboard at http://localhost:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
