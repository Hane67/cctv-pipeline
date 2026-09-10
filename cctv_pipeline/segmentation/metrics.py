"""
Evaluation suite implementing the CDNet2014 benchmark standard for background subtraction.
Calculates Precision, Recall, F-measure (F1), PWC, FPR, and FNR across image sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SegmentationMetrics:
    """Quantitative benchmark metrics for foreground segmentation."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def f_measure(self) -> float:
        prec = self.precision
        rec = self.recall
        denom = prec + rec
        return float(2.0 * (prec * rec) / denom) if denom > 0 else 0.0

    @property
    def pwc(self) -> float:
        """Percentage of Wrong Classifications (PWC / Error Rate). Lower is better."""
        total = self.tp + self.fp + self.tn + self.fn
        return float(100.0 * (self.fn + self.fp) / total) if total > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate. Lower is better."""
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom > 0 else 0.0

    @property
    def fnr(self) -> float:
        """False Negative Rate. Lower is better."""
        denom = self.tp + self.fn
        return float(self.fn / denom) if denom > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F_Measure": round(self.f_measure, 4),
            "PWC": round(self.pwc, 4),
            "FPR": round(self.fpr, 4),
            "FNR": round(self.fnr, 4),
        }


class CDNetEvaluator:
    """
    Evaluates binary segmentation masks against ground-truth frames.
    Follows CDNet conventions:
    - 0: Static Background
    - 50: Shade / Shadow
    - 85: Non-critical Background
    - 170: Unknown / Outside ROI (ignored in evaluation)
    - 255: Foreground Motion
    """

    def __init__(self, ignore_unknown: bool = True):
        self.ignore_unknown = ignore_unknown
        self.total_metrics = SegmentationMetrics()
        self.frame_history: List[SegmentationMetrics] = []

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> SegmentationMetrics:
        """
        Computes confusion matrix between predicted mask and ground truth mask.
        """
        # Ensure single channel
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask[:, :, 0]
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0]

        # Binary prediction (0 or 1)
        pred_bin = (pred_mask > 127)

        # CDNet evaluation:
        # Foreground = 255
        # Background = 0 or 50 (shade) or 85
        # Unknown/ROI boundary = 170 (ignored if ignore_unknown is True)
        if self.ignore_unknown:
            valid_mask = (gt_mask != 170)
        else:
            valid_mask = np.ones_like(gt_mask, dtype=bool)

        gt_fg = (gt_mask == 255) & valid_mask
        gt_bg = (gt_mask < 128) & valid_mask

        pred_fg = pred_bin & valid_mask
        pred_bg = (~pred_bin) & valid_mask

        tp = int(np.sum(pred_fg & gt_fg))
        fp = int(np.sum(pred_fg & gt_bg))
        tn = int(np.sum(pred_bg & gt_bg))
        fn = int(np.sum(pred_bg & gt_fg))

        frame_m = SegmentationMetrics(tp=tp, fp=fp, tn=tn, fn=fn)
        self.frame_history.append(frame_m)

        self.total_metrics.tp += tp
        self.total_metrics.fp += fp
        self.total_metrics.tn += tn
        self.total_metrics.fn += fn

        return frame_m

    def get_summary(self) -> Dict[str, float]:
        """Returns overall cumulative metrics across all evaluated frames."""
        return self.total_metrics.to_dict()

    def generate_markdown_report(self, algorithm_name: str = "SuBSENSE") -> str:
        """Generates a Markdown table summary suitable for reports and papers."""
        m = self.total_metrics
        table = f"""
| Metric | {algorithm_name} Value | Description |
|---|---|---|
| **F-Measure (F1)** | **{m.f_measure:.4f}** | Harmonic mean of precision and recall (Primary Metric) |
| **Precision** | {m.precision:.4f} | True positive ratio over all predicted foreground |
| **Recall** | {m.recall:.4f} | Detection rate of actual ground-truth foreground |
| **PWC (%)** | {m.pwc:.2f}% | Percentage of Wrong Classifications (lower is better) |
| **FPR** | {m.fpr:.4f} | False Positive Rate (lower is better) |
| **FNR** | {m.fnr:.4f} | False Negative Rate (lower is better) |
| **Total Frames** | {len(self.frame_history)} | Number of sequence frames evaluated |
"""
        return table.strip()
