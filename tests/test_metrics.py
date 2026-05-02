"""Unit tests for the metrics module."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from text_classification.metrics.multilabel import MultiLabelMetrics


class TestMultiLabelMetrics:
    CLASSES = ["cat", "dog", "bird"]

    def _make_metrics(self) -> MultiLabelMetrics:
        return MultiLabelMetrics(self.CLASSES)

    # ------------------------------------------------------------------
    # Basic sanity
    # ------------------------------------------------------------------

    def test_empty_returns_empty_dict(self):
        m = self._make_metrics()
        assert m.compute() == {}

    def test_reset_clears_state(self):
        m = self._make_metrics()
        logits = torch.zeros(4, 3)
        targets = torch.zeros(4, 3)
        m.update(logits, targets)
        m.reset()
        assert m.compute() == {}

    # ------------------------------------------------------------------
    # Perfect predictions
    # ------------------------------------------------------------------

    def test_perfect_predictions(self):
        """All predictions correct → mAP and macro-F1 should be 1.0."""
        m = self._make_metrics()
        # Large positive logit → prob ≈ 1 for class 0; large negative → prob ≈ 0 for others
        logits = torch.tensor(
            [
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
                [10.0, 10.0, -10.0],
            ]
        )
        targets = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        m.update(logits, targets)
        results = m.compute()

        assert pytest.approx(results["mAP"], abs=1e-4) == 1.0
        assert pytest.approx(results["f1/macro"], abs=1e-4) == 1.0
        assert pytest.approx(results["precision/macro"], abs=1e-4) == 1.0
        assert pytest.approx(results["recall/macro"], abs=1e-4) == 1.0

    # ------------------------------------------------------------------
    # Valid mask
    # ------------------------------------------------------------------

    def test_valid_mask_filters_examples(self):
        """Examples with valid_mask=False should not contribute to metrics."""
        m = self._make_metrics()
        # Correct prediction for example 0, wrong for example 1
        logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Mask out example 1 (the wrong one)
        valid = torch.tensor([True, False])
        m.update(logits, targets, valid)
        results = m.compute()
        # Only example 0 counts — should be perfect
        assert pytest.approx(results["f1/macro"], abs=1e-4) == 1.0

    # ------------------------------------------------------------------
    # Per-class keys present
    # ------------------------------------------------------------------

    def test_per_class_keys(self):
        m = self._make_metrics()
        logits = torch.randn(8, 3)
        targets = (torch.rand(8, 3) > 0.5).float()
        m.update(logits, targets)
        results = m.compute()
        for cls in self.CLASSES:
            assert f"ap/{cls}" in results
            assert f"precision/{cls}" in results
            assert f"recall/{cls}" in results
            assert f"f1/{cls}" in results
        assert "mAP" in results
        assert "precision/macro" in results
        assert "recall/macro" in results
        assert "f1/macro" in results

    # ------------------------------------------------------------------
    # NaN when only one class present
    # ------------------------------------------------------------------

    def test_ap_nan_when_single_class(self):
        """AP is undefined when all targets are 0 or all are 1."""
        m = self._make_metrics()
        logits = torch.randn(4, 3)
        targets = torch.zeros(4, 3)  # all-zero → AP undefined for every class
        m.update(logits, targets)
        results = m.compute()
        for cls in self.CLASSES:
            assert math.isnan(results[f"ap/{cls}"])
        assert math.isnan(results["mAP"])

    # ------------------------------------------------------------------
    # Accumulation across batches
    # ------------------------------------------------------------------

    def test_multi_batch_accumulation(self):
        """Results should be the same whether examples come in one or two batches."""
        rng = torch.Generator().manual_seed(0)
        logits_all = torch.randn(10, 3, generator=rng)
        targets_all = (torch.rand(10, 3, generator=rng) > 0.5).float()

        m_single = self._make_metrics()
        m_single.update(logits_all, targets_all)
        r_single = m_single.compute()

        m_multi = self._make_metrics()
        m_multi.update(logits_all[:5], targets_all[:5])
        m_multi.update(logits_all[5:], targets_all[5:])
        r_multi = m_multi.compute()

        for key in r_single:
            if math.isnan(r_single[key]):
                assert math.isnan(r_multi[key])
            else:
                assert pytest.approx(r_multi[key], abs=1e-5) == r_single[key]
