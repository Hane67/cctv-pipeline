"""
Stage 3: Multi-QP Automated Sweep & Detection Retention Analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from cctv_pipeline.core.logger import logger
from cctv_pipeline.compression.encoder import encode_at_qp, measure_video_quality


class CompressionSweeper:
    """
    Automates encoding a surveillance clip across an array of QP levels,
    measuring detection retention vs. a near-lossless reference clip.
    """

    def __init__(self, codec: str = "libx265", reference_qp: int = 18):
        self.codec = codec
        self.reference_qp = reference_qp

    def run_sweep(
        self,
        input_video: Union[str, Path],
        out_dir: Union[str, Path],
        qp_levels: List[int]
    ) -> Dict[int, Path]:
        """Encodes the source video at each specified QP level."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        encoded_paths = {}

        for qp in qp_levels:
            out_file = out_dir / f"qp_{qp}.mp4"
            logger.info(f"Encoding clip at QP={qp} -> {out_file}")
            encode_at_qp(input_video, out_file, qp=qp, codec=self.codec)
            encoded_paths[qp] = out_file

        return encoded_paths

    def evaluate_retention(
        self,
        model,
        encoded_by_qp: Dict[int, Path],
        classes: Optional[List[int]] = None,
        conf: float = 0.35
    ) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, Dict[str, float]]]:
        """
        Runs YOLOv8 detector over each compressed video file.
        Computes detection count retention relative to the reference QP video.
        """
        counts = {}
        quality_metrics = {}

        # First evaluate reference video
        ref_path = encoded_by_qp.get(self.reference_qp, next(iter(encoded_by_qp.values())))

        for qp, path in encoded_by_qp.items():
            results = model.predict(source=str(path), conf=conf, classes=classes, stream=True, verbose=False)
            total_dets = sum(len(r.boxes) for r in results)
            counts[qp] = total_dets

            # Quality metrics vs reference
            quality_metrics[qp] = measure_video_quality(ref_path, path)

        ref_count = counts.get(self.reference_qp, max(counts.values()) or 1)
        retention = {
            qp: float(c / ref_count if ref_count > 0 else 0.0)
            for qp, c in counts.items()
        }

        return counts, retention, quality_metrics
