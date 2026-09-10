"""
Stage 3: Compression Metrics & Knee-Point Discovery.
Analyzes retention degradation and plots publication-grade curves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_operational_knee_point(
    retention_by_qp: Dict[int, float],
    threshold: float = 0.90
) -> int:
    """
    Finds the operational QP ceiling (knee point):
    First QP (in ascending order) where detection retention drops below threshold (e.g. 90%).
    If retention never drops below threshold, returns the maximum QP tested.
    """
    sorted_qps = sorted(retention_by_qp.keys())
    for qp in sorted_qps:
        if retention_by_qp[qp] < threshold:
            return qp
    return max(sorted_qps)


def plot_compression_analysis(
    retention_by_qp: Dict[int, float],
    quality_by_qp: Dict[int, Dict[str, float]],
    knee_point: int,
    output_png: Union[str, Path],
    threshold: float = 0.90
) -> Path:
    """
    Generates a 3-panel figure:
    1. Detection Retention (%) vs H.264/H.265 QP (with knee-point marker).
    2. Average PSNR (dB) vs QP.
    3. Bitrate (kbps) vs QP.
    """
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    qps = sorted(retention_by_qp.keys())
    retentions = [retention_by_qp[q] * 100.0 for q in qps]
    psnrs = [quality_by_qp.get(q, {}).get("avg_psnr", 0.0) for q in qps]
    bitrates = [quality_by_qp.get(q, {}).get("bitrate_kbps", 0.0) for q in qps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    # Panel 1: Retention vs QP
    ax1 = axes[0]
    ax1.plot(qps, retentions, marker="o", color="#1f77b4", linewidth=2, label="Detection Retention")
    ax1.axhline(threshold * 100, color="crimson", linestyle="--", label=f"Operational Target ({int(threshold*100)}%)")
    ax1.axvline(knee_point, color="gray", linestyle=":", label=f"Knee Point (QP {knee_point})")
    ax1.set_xlabel("Compression Parameter (QP)", fontsize=11)
    ax1.set_ylabel("Detection Retention (%)", fontsize=11)
    ax1.set_title("Detection Accuracy vs. Compression QP", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)

    # Panel 2: PSNR vs QP
    ax2 = axes[1]
    ax2.plot(qps, psnrs, marker="s", color="#ff7f0e", linewidth=2)
    ax2.set_xlabel("Compression Parameter (QP)", fontsize=11)
    ax2.set_ylabel("Average PSNR (dB)", fontsize=11)
    ax2.set_title("Reconstruction Quality (PSNR) vs. QP", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Bitrate vs QP
    ax3 = axes[2]
    ax3.plot(qps, bitrates, marker="^", color="#2ca02c", linewidth=2)
    ax3.set_xlabel("Compression Parameter (QP)", fontsize=11)
    ax3.set_ylabel("Bitrate (kbps)", fontsize=11)
    ax3.set_title("Stream Bandwidth vs. QP", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)
    return out_file
