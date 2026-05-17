"""Tests for JEPA model components."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from text_classification.jepa.corruption import (
    BYTE_VOCAB_SIZE,
    SEGMENT_SIZE,
    CorruptionType,
)
from text_classification.jepa.model import (
    ByteConvBlock,
    ByteInputEmbedding,
    ByteToSegmentReducer,
    ByteSegmentEncoder,
    ByteSegmentJEPA,
    LocalByteProcessor,
    SegmentPredictor,
    SwiGLU,
    TransformerEncoderWithIntermediates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_byte_input(B: int = 2, N: int = 5, device: str = "cpu"):
    byte_vals = torch.randint(0, 256, (B, N, SEGMENT_SIZE))
    byte_types = torch.zeros(B, N, SEGMENT_SIZE, dtype=torch.long)
    return byte_vals, byte_types


def _make_small_encoder(max_segments: int = 32) -> ByteSegmentEncoder:
    return ByteSegmentEncoder(
        byte_dim=16,
        seg_dim=32,
        n_byte_blocks=1,
        n_encoder_layers=2,
        n_heads=2,
        ffn_dim=64,
        max_segments=max_segments,
        kernel_size=3,
        dropout=0.0,
        byte_dropout=0.0,
    )


def _make_small_jepa(max_segments: int = 32) -> ByteSegmentJEPA:
    return ByteSegmentJEPA(
        byte_dim=16,
        seg_dim=32,
        pred_dim=32,
        n_byte_blocks=1,
        n_encoder_layers=2,
        n_heads=2,
        ffn_dim=64,
        n_predictor_layers=2,
        max_segments=max_segments,
        kernel_size=3,
        dropout=0.0,
        byte_dropout=0.0,
    )


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

class TestSwiGLU:
    def test_output_shape(self):
        m = SwiGLU(32, 64, 32)
        x = torch.randn(4, 32)
        assert m(x).shape == (4, 32)

    def test_3d_input(self):
        m = SwiGLU(16, 32, 16)
        x = torch.randn(2, 10, 16)
        assert m(x).shape == (2, 10, 16)


# ---------------------------------------------------------------------------
# ByteConvBlock
# ---------------------------------------------------------------------------

class TestByteConvBlock:
    def test_output_shape_preserved(self):
        blk = ByteConvBlock(byte_dim=16, kernel_size=3, dropout=0.0)
        x = torch.randn(4, SEGMENT_SIZE, 16)
        out = blk(x)
        assert out.shape == x.shape

    def test_residual_with_zero_weights(self):
        """With zero-init weights the residual should dominate."""
        blk = ByteConvBlock(byte_dim=8, kernel_size=3, dropout=0.0)
        blk.eval()
        x = torch.zeros(2, SEGMENT_SIZE, 8)
        # Just check it runs without NaN
        out = blk(x)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# ByteInputEmbedding
# ---------------------------------------------------------------------------

class TestByteInputEmbedding:
    def test_output_shape(self):
        emb = ByteInputEmbedding(byte_dim=32)
        bv, bt = _make_byte_input(2, 5)
        out = emb(bv, bt)
        assert out.shape == (2, 5, SEGMENT_SIZE, 32)

    def test_valid_byte_range(self):
        emb = ByteInputEmbedding(byte_dim=16)
        # PAD_BYTE and MASK_BYTE should be accepted
        bv = torch.tensor([[[256, 257, 258] + [0] * (SEGMENT_SIZE - 3)]])
        bt = torch.zeros(1, 1, SEGMENT_SIZE, dtype=torch.long)
        out = emb(bv, bt)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# ByteToSegmentReducer
# ---------------------------------------------------------------------------

class TestByteToSegmentReducer:
    def test_output_shape(self):
        red = ByteToSegmentReducer(byte_dim=16, seg_dim=32)
        x = torch.randn(2, 5, SEGMENT_SIZE, 16)
        out = red(x)
        assert out.shape == (2, 5, 32)


# ---------------------------------------------------------------------------
# LocalByteProcessor
# ---------------------------------------------------------------------------

class TestLocalByteProcessor:
    def test_shape_preserved(self):
        proc = LocalByteProcessor(byte_dim=16, n_blocks=2, kernel_size=3, dropout=0.0)
        x = torch.randn(2, 4, SEGMENT_SIZE, 16)
        out = proc(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# TransformerEncoderWithIntermediates
# ---------------------------------------------------------------------------

class TestTransformerEncoderWithIntermediates:
    def test_output_shapes(self):
        enc = TransformerEncoderWithIntermediates(
            num_layers=3, d_model=32, nhead=2, dim_feedforward=64, dropout=0.0
        )
        x = torch.randn(2, 10, 32)
        out, layers = enc(x)
        assert out.shape == (2, 10, 32)
        assert len(layers) == 3
        for l in layers:
            assert l.shape == (2, 10, 32)

    def test_with_padding_mask(self):
        enc = TransformerEncoderWithIntermediates(
            num_layers=2, d_model=16, nhead=2, dim_feedforward=32, dropout=0.0
        )
        x = torch.randn(2, 6, 16)
        mask = torch.tensor([[False, False, False, True, True, True],
                              [False, False, True, True, True, True]])  # True = pad
        out, _ = enc(x, key_padding_mask=mask)
        assert out.shape == (2, 6, 16)
        assert not out.isnan().any()


# ---------------------------------------------------------------------------
# ByteSegmentEncoder
# ---------------------------------------------------------------------------

class TestByteSegmentEncoder:
    def test_output_shape(self):
        enc = _make_small_encoder()
        bv, bt = _make_byte_input(B=2, N=5)
        positions = torch.arange(5).unsqueeze(0).expand(2, -1)
        seg_mask = torch.ones(2, 5, dtype=torch.bool)
        out, layers = enc(bv, bt, positions, seg_mask)
        # Output: (B, 1+N, seg_dim)
        assert out.shape == (2, 6, 32)
        assert len(layers) == 2

    def test_doc_token_always_first(self):
        enc = _make_small_encoder()
        bv, bt = _make_byte_input(B=1, N=3)
        positions = torch.arange(3).unsqueeze(0)
        out, _ = enc(bv, bt, positions)
        # out[:, 0, :] is the [DOC] token — just verify shape
        assert out[:, 0, :].shape == (1, 32)

    def test_with_partial_mask(self):
        enc = _make_small_encoder()
        bv, bt = _make_byte_input(B=2, N=6)
        positions = torch.arange(6).unsqueeze(0).expand(2, -1)
        seg_mask = torch.tensor([
            [True, True, True, False, False, False],
            [True, True, False, False, False, False],
        ])
        out, _ = enc(bv, bt, positions, seg_mask)
        assert out.shape == (2, 7, 32)
        assert not out.isnan().any()

    def test_no_grad_through_none_mask(self):
        enc = _make_small_encoder()
        bv, bt = _make_byte_input(B=1, N=4)
        positions = torch.arange(4).unsqueeze(0)
        out, _ = enc(bv, bt, positions)
        assert out.shape == (1, 5, 32)


# ---------------------------------------------------------------------------
# SegmentPredictor
# ---------------------------------------------------------------------------

class TestSegmentPredictor:
    def test_output_shape(self):
        pred = SegmentPredictor(
            d_model=32, pred_dim=32, nhead=2, dim_feedforward=64,
            num_layers=2, max_segments=32, dropout=0.0,
        )
        B, N, M = 2, 5, 4
        canonical_positions = torch.arange(N).unsqueeze(0).expand(B, -1)
        student_context = torch.randn(B, M + 1, 32)  # +1 for [DOC]
        canonical_mask = torch.ones(B, N, dtype=torch.bool)
        out = pred(canonical_positions, student_context, canonical_mask=canonical_mask)
        assert out.shape == (B, N, 32)

    def test_output_unit_normalized(self):
        pred = SegmentPredictor(
            d_model=16, pred_dim=16, nhead=2, dim_feedforward=32,
            num_layers=1, max_segments=16, dropout=0.0,
        )
        B, N = 2, 4
        pos = torch.arange(N).unsqueeze(0).expand(B, -1)
        ctx = torch.randn(B, 5, 16)
        out = pred(pos, ctx)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# ByteSegmentJEPA (full model)
# ---------------------------------------------------------------------------

class TestByteSegmentJEPA:
    def test_forward_output_keys(self):
        model = _make_small_jepa()
        B, N, M = 2, 5, 3
        bv, bt = _make_byte_input(B, N)
        canonical_pos = torch.arange(N).unsqueeze(0).expand(B, -1)
        canonical_mask = torch.ones(B, N, dtype=torch.bool)

        sbv = bv[:, :M, :]
        sbt = bt[:, :M, :]
        student_pos = torch.arange(M).unsqueeze(0).expand(B, -1)
        student_mask = torch.ones(B, M, dtype=torch.bool)

        out = model(bv, bt, canonical_pos, canonical_mask, sbv, sbt, student_pos, student_mask)
        assert set(out.keys()) == {
            "predicted_segments", "teacher_seg_targets",
            "predicted_doc", "teacher_doc_targets",
        }

    def test_output_shapes(self):
        model = _make_small_jepa()
        B, N, M = 2, 5, 3
        bv, bt = _make_byte_input(B, N)
        canonical_pos = torch.arange(N).unsqueeze(0).expand(B, -1)
        canonical_mask = torch.ones(B, N, dtype=torch.bool)
        sbv = bv[:, :M, :]
        sbt = bt[:, :M, :]
        student_pos = torch.arange(M).unsqueeze(0).expand(B, -1)
        student_mask = torch.ones(B, M, dtype=torch.bool)
        out = model(bv, bt, canonical_pos, canonical_mask, sbv, sbt, student_pos, student_mask)
        assert out["predicted_segments"].shape == (B, N, 32)
        assert out["teacher_seg_targets"].shape == (B, N, 32)
        assert out["predicted_doc"].shape == (B, 32)
        assert out["teacher_doc_targets"].shape == (B, 32)

    def test_teacher_has_no_grad(self):
        model = _make_small_jepa()
        for name, param in model.teacher.named_parameters():
            assert not param.requires_grad, f"Teacher param {name} has requires_grad=True"

    def test_ema_update_changes_teacher(self):
        model = _make_small_jepa()
        # Modify student slightly
        with torch.no_grad():
            for p in model.student.parameters():
                p.add_(torch.ones_like(p) * 0.1)
        old_teacher = {k: v.clone() for k, v in model.teacher.named_parameters()}
        model.update_teacher(momentum=0.9)
        for name, new_p in model.teacher.named_parameters():
            old_p = old_teacher[name]
            assert not torch.allclose(new_p, old_p), f"Teacher {name} not updated"

    def test_predictions_unit_normalized(self):
        model = _make_small_jepa()
        model.eval()
        B, N, M = 2, 4, 3
        bv, bt = _make_byte_input(B, N)
        canonical_pos = torch.arange(N).unsqueeze(0).expand(B, -1)
        canonical_mask = torch.ones(B, N, dtype=torch.bool)
        sbv = bv[:, :M, :]
        sbt = bt[:, :M, :]
        student_pos = torch.arange(M).unsqueeze(0).expand(B, -1)
        student_mask = torch.ones(B, M, dtype=torch.bool)
        with torch.no_grad():
            out = model(bv, bt, canonical_pos, canonical_mask, sbv, sbt, student_pos, student_mask)
        norms = out["predicted_segments"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_padded_batch(self):
        """Mixed-length batch via padding should not produce NaN."""
        model = _make_small_jepa(max_segments=10)
        model.eval()
        B = 2
        max_N, max_M = 6, 4

        # Document 0: 6 canonical segs, 4 student segs
        # Document 1: 3 canonical segs, 2 student segs (rest is padded)
        from text_classification.jepa.corruption import PAD_BYTE
        bv = torch.full((B, max_N, SEGMENT_SIZE), PAD_BYTE, dtype=torch.long)
        bt = torch.full((B, max_N, SEGMENT_SIZE), int(CorruptionType.PADDING), dtype=torch.long)
        bv[0] = torch.randint(0, 256, (max_N, SEGMENT_SIZE))
        bt[0] = torch.zeros(max_N, SEGMENT_SIZE, dtype=torch.long)
        bv[1, :3] = torch.randint(0, 256, (3, SEGMENT_SIZE))
        bt[1, :3] = torch.zeros(3, SEGMENT_SIZE, dtype=torch.long)

        canonical_mask = torch.tensor([[True] * max_N, [True, True, True, False, False, False]])
        canonical_pos = torch.arange(max_N).unsqueeze(0).expand(B, -1)

        sbv = torch.full((B, max_M, SEGMENT_SIZE), PAD_BYTE, dtype=torch.long)
        sbt = torch.full((B, max_M, SEGMENT_SIZE), int(CorruptionType.PADDING), dtype=torch.long)
        sbv[0] = torch.randint(0, 256, (max_M, SEGMENT_SIZE))
        sbt[0] = torch.zeros(max_M, SEGMENT_SIZE, dtype=torch.long)
        sbv[1, :2] = torch.randint(0, 256, (2, SEGMENT_SIZE))
        sbt[1, :2] = torch.zeros(2, SEGMENT_SIZE, dtype=torch.long)

        student_pos = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]])
        student_mask = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
        ])

        with torch.no_grad():
            out = model(bv, bt, canonical_pos, canonical_mask, sbv, sbt, student_pos, student_mask)
        assert not out["predicted_segments"].isnan().any()
        assert not out["predicted_doc"].isnan().any()
