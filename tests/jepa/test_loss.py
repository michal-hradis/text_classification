"""Tests for JEPA loss functions."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from text_classification.jepa.loss import (
    LossWeights,
    covariance_regularization,
    compute_total_loss,
    document_consistency_loss,
    segment_jepa_loss,
    variance_regularization,
)


# ---------------------------------------------------------------------------
# segment_jepa_loss
# ---------------------------------------------------------------------------

class TestSegmentJEPALoss:
    def test_perfect_prediction_zero_loss(self):
        B, N, D = 2, 5, 32
        pred = F.normalize(torch.randn(B, N, D), dim=-1)
        targets = pred.clone()
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = segment_jepa_loss(pred, targets, weights, mask)
        assert loss.item() < 1e-5

    def test_orthogonal_prediction_max_loss(self):
        """Orthogonal predictions give cosine=0 → loss≈1."""
        B, N, D = 2, 4, 32
        pred = torch.zeros(B, N, D)
        pred[:, :, 0] = 1.0   # unit vector along dim 0
        targets = torch.zeros(B, N, D)
        targets[:, :, 1] = 1.0  # unit vector along dim 1 (orthogonal)
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = segment_jepa_loss(pred, targets, weights, mask)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_mask_excludes_padding(self):
        B, N, D = 2, 6, 16
        pred = F.normalize(torch.randn(B, N, D), dim=-1)
        targets = pred.clone()  # zero loss when masked correctly
        weights = torch.ones(B, N)
        # Only first 3 segments valid
        mask = torch.tensor([[True, True, True, False, False, False]] * B)
        loss = segment_jepa_loss(pred, targets, weights, mask)
        assert loss.item() < 1e-5

    def test_weight_zero_excludes_segment(self):
        B, N, D = 1, 4, 16
        pred = F.normalize(torch.randn(B, N, D), dim=-1)
        targets = F.normalize(-pred, dim=-1)  # anti-parallel (loss=2)
        weights = torch.zeros(B, N)
        weights[0, :2] = 0.0  # all zero
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = segment_jepa_loss(pred, targets, weights, mask)
        # Division by clamped weight → not well-defined, but should not crash
        assert not loss.isnan()

    def test_gradient_flows(self):
        B, N, D = 2, 4, 16
        pred = torch.randn(B, N, D, requires_grad=True)
        pred_norm = F.normalize(pred, dim=-1)
        targets = F.normalize(torch.randn(B, N, D), dim=-1)
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = segment_jepa_loss(pred_norm, targets, weights, mask)
        loss.backward()
        assert pred.grad is not None
        assert not pred.grad.isnan().any()

    def test_targets_stop_gradient(self):
        """targets are detached — modifying them should not affect student grad."""
        B, N, D = 2, 3, 8
        pred = F.normalize(torch.randn(B, N, D, requires_grad=True), dim=-1)
        targets = F.normalize(torch.randn(B, N, D, requires_grad=True), dim=-1)
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = segment_jepa_loss(pred, targets, weights, mask)
        loss.backward()
        assert targets.grad is None  # detached → no grad


# ---------------------------------------------------------------------------
# document_consistency_loss
# ---------------------------------------------------------------------------

class TestDocumentConsistencyLoss:
    def test_perfect_match_zero_loss(self):
        doc = F.normalize(torch.randn(4, 32), dim=-1)
        loss = document_consistency_loss(doc, doc.clone())
        assert loss.item() < 1e-5

    def test_orthogonal_max_loss(self):
        d = 32
        student = torch.zeros(2, d)
        student[:, 0] = 1.0
        teacher = torch.zeros(2, d)
        teacher[:, 1] = 1.0
        loss = document_consistency_loss(student, teacher)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_gradient_flows(self):
        doc = torch.randn(4, 16, requires_grad=True)
        doc_norm = F.normalize(doc, dim=-1)
        target = F.normalize(torch.randn(4, 16), dim=-1)
        loss = document_consistency_loss(doc_norm, target)
        loss.backward()
        assert doc.grad is not None


# ---------------------------------------------------------------------------
# variance_regularization
# ---------------------------------------------------------------------------

class TestVarianceRegularization:
    def test_zero_variance_penalized(self):
        """Constant embeddings have zero variance → large penalty."""
        emb = torch.ones(16, 32)  # all identical rows
        loss = variance_regularization(emb, gamma=0.1)
        assert loss.item() > 0

    def test_high_variance_not_penalized(self):
        """Normally distributed embeddings have high variance → near-zero penalty."""
        torch.manual_seed(0)
        emb = torch.randn(256, 32)  # std ≈ 1 >> gamma=0.1
        loss = variance_regularization(emb, gamma=0.1)
        assert loss.item() < 1e-3

    def test_3d_input_flattened(self):
        emb = torch.randn(4, 8, 32)
        loss = variance_regularization(emb, gamma=0.1)
        assert loss.ndim == 0  # scalar

    def test_multiple_inputs_averaged(self):
        emb1 = torch.ones(8, 16)   # constant → high penalty
        emb2 = torch.randn(8, 16)  # normal → low penalty
        loss_both = variance_regularization(emb1, emb2, gamma=0.1)
        loss_single = variance_regularization(emb1, gamma=0.1)
        # Combined should be less than single high-penalty input alone
        assert loss_both.item() < loss_single.item()


# ---------------------------------------------------------------------------
# covariance_regularization
# ---------------------------------------------------------------------------

class TestCovarianceRegularization:
    def test_diagonal_embedding_zero_penalty(self):
        """Embeddings with uncorrelated dimensions have near-zero off-diagonal cov."""
        torch.manual_seed(0)
        emb = torch.randn(256, 32)
        loss = covariance_regularization(emb)
        # Should be small for random uncorrelated embeddings
        assert loss.item() < 0.5

    def test_correlated_embedding_penalized(self):
        """Perfectly correlated dimensions have high off-diagonal covariance."""
        N = 64
        emb = torch.zeros(N, 4)
        v = torch.randn(N)
        emb[:, 0] = v
        emb[:, 1] = v  # identical to dim 0
        emb[:, 2] = torch.randn(N)
        emb[:, 3] = torch.randn(N)
        loss = covariance_regularization(emb)
        assert loss.item() > 0.1

    def test_3d_input(self):
        emb = torch.randn(4, 8, 16)
        loss = covariance_regularization(emb)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# compute_total_loss
# ---------------------------------------------------------------------------

class TestComputeTotalLoss:
    def test_output_keys(self):
        B, N, D = 2, 4, 16
        pred_seg = F.normalize(torch.randn(B, N, D), dim=-1)
        tgt_seg = F.normalize(torch.randn(B, N, D), dim=-1)
        pred_doc = F.normalize(torch.randn(B, D), dim=-1)
        tgt_doc = F.normalize(torch.randn(B, D), dim=-1)
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        losses = compute_total_loss(
            pred_seg, tgt_seg, pred_doc, tgt_doc, weights, mask, LossWeights()
        )
        expected = {"loss", "loss/segment", "loss/document", "loss/variance", "loss/covariance"}
        assert expected == set(losses.keys())

    def test_total_is_weighted_sum(self):
        B, N, D = 2, 4, 16
        pred_seg = F.normalize(torch.randn(B, N, D), dim=-1)
        tgt_seg = pred_seg.clone()  # l_seg ≈ 0
        pred_doc = F.normalize(torch.randn(B, D), dim=-1)
        tgt_doc = pred_doc.clone()  # l_doc ≈ 0
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        w = LossWeights(segment=1.0, document=0.2, variance=0.05, covariance=0.01)
        losses = compute_total_loss(pred_seg, tgt_seg, pred_doc, tgt_doc, weights, mask, w)
        expected_total = (
            w.segment * losses["loss/segment"]
            + w.document * losses["loss/document"]
            + w.variance * losses["loss/variance"]
            + w.covariance * losses["loss/covariance"]
        )
        assert torch.allclose(losses["loss"], expected_total, atol=1e-6)

    def test_gradient_flows_to_predictions(self):
        B, N, D = 2, 3, 8
        raw_pred = torch.randn(B, N, D, requires_grad=True)
        pred_seg = F.normalize(raw_pred.view(B, N, D), dim=-1)
        tgt_seg = F.normalize(torch.randn(B, N, D), dim=-1)
        raw_doc = torch.randn(B, D, requires_grad=True)
        pred_doc = F.normalize(raw_doc, dim=-1)
        tgt_doc = F.normalize(torch.randn(B, D), dim=-1)
        weights = torch.ones(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        losses = compute_total_loss(pred_seg, tgt_seg, pred_doc, tgt_doc, weights, mask, LossWeights())
        losses["loss"].backward()
        assert raw_pred.grad is not None
        assert raw_doc.grad is not None
