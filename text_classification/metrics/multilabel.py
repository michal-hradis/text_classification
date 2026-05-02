"""Multi-label classification metrics accumulator.

Computes per-class and aggregated metrics after accumulating predictions and
ground-truth labels across batches:

- **AP** (Average Precision) per class → **mAP** (macro mean over classes)
- **Precision**, **Recall**, **F1** at threshold 0.5, per class and macro
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


class MultiLabelMetrics:
    """Accumulates sigmoid scores and binary targets, then computes metrics.

    Usage::

        m = MultiLabelMetrics(["cat_A", "cat_B", "cat_C"])
        for batch in dataloader:
            m.update(logits, targets, valid_mask)
        results = m.compute()   # dict[str, float]
        m.reset()
    """

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.n_classes = len(class_names)
        self._scores: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def reset(self) -> None:
        self._scores = []
        self._targets = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Accumulate one batch of logits and binary targets.

        Args:
            logits:     (B, C) raw model outputs (before sigmoid).
            targets:    (B, C) float binary labels.
            valid_mask: (B,) bool tensor; only valid examples are accumulated.
        """
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        tgts = targets.detach().float().cpu().numpy()

        if valid_mask is not None:
            mask = valid_mask.detach().cpu().numpy().astype(bool)
            probs = probs[mask]
            tgts = tgts[mask]

        if len(probs) > 0:
            self._scores.append(probs)
            self._targets.append(tgts)

    def compute(self) -> dict[str, float]:
        """Return a flat dict of all computed metric values.

        Keys follow the pattern ``"<metric>/<class_name>"`` for per-class
        values and ``"<metric>/macro"`` / ``"mAP"`` for aggregates.
        """
        if not self._scores:
            return {}

        scores = np.concatenate(self._scores, axis=0)   # (N, C)
        targets = np.concatenate(self._targets, axis=0)  # (N, C)
        preds = (scores >= 0.5).astype(int)

        metrics: dict[str, float] = {}
        ap_values: list[float] = []

        for i, cls in enumerate(self.class_names):
            s = scores[:, i]
            t = targets[:, i].astype(int)
            p = preds[:, i]

            # AP: undefined when all labels are the same
            if 0 < t.sum() < len(t):
                ap = float(average_precision_score(t, s))
                ap_values.append(ap)
            else:
                ap = float("nan")
            metrics[f"ap/{cls}"] = ap

            metrics[f"precision/{cls}"] = float(
                precision_score(t, p, zero_division=0)
            )
            metrics[f"recall/{cls}"] = float(
                recall_score(t, p, zero_division=0)
            )
            metrics[f"f1/{cls}"] = float(
                f1_score(t, p, zero_division=0)
            )

        metrics["mAP"] = float(np.mean(ap_values)) if ap_values else float("nan")

        # Compute macro metrics only over classes that have at least one positive
        # example (same convention as AP/mAP: skip classes with no ground truth).
        prec_vals: list[float] = []
        rec_vals: list[float] = []
        f1_vals: list[float] = []
        for i in range(self.n_classes):
            t = targets[:, i].astype(int)
            p = preds[:, i]
            if t.sum() == 0:
                continue
            prec_vals.append(float(precision_score(t, p, zero_division=0)))
            rec_vals.append(float(recall_score(t, p, zero_division=0)))
            f1_vals.append(float(f1_score(t, p, zero_division=0)))

        metrics["precision/macro"] = float(np.mean(prec_vals)) if prec_vals else float("nan")
        metrics["recall/macro"] = float(np.mean(rec_vals)) if rec_vals else float("nan")
        metrics["f1/macro"] = float(np.mean(f1_vals)) if f1_vals else float("nan")

        return metrics
