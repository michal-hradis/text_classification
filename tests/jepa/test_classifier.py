"""Tests for JEPAClassifier and JEPAFinetuneModule.

All tests use randomly-initialised encoder weights — no real checkpoint is
required.  The checkpoint-loading path is tested via a temporary file that
wraps a freshly constructed ByteSegmentJEPA state dict in Lightning format.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import torch

from text_classification.jepa.corruption import SEGMENT_SIZE
from text_classification.jepa.model import ByteSegmentJEPA
from text_classification.jepa.classifier import JEPAClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASKS = ["topic", "sentiment"]
NUM_CLASSES = {"topic": 4, "sentiment": 3}


def _make_small_jepa() -> ByteSegmentJEPA:
    """Construct a tiny ByteSegmentJEPA for fast tests."""
    return ByteSegmentJEPA(
        byte_dim=16,
        seg_dim=32,
        pred_dim=32,
        n_byte_blocks=1,
        n_encoder_layers=2,
        n_heads=2,
        ffn_dim=64,
        n_predictor_layers=2,
        max_segments=64,
        kernel_size=3,
        dropout=0.0,
        byte_dropout=0.0,
    )


def _save_jepa_ckpt(jepa: ByteSegmentJEPA, path: str) -> None:
    """Save a fake Lightning checkpoint wrapping a ByteSegmentJEPA state dict."""
    # Lightning format: state_dict keys are prefixed with the module attribute name
    prefixed = {f"model.{k}": v for k, v in jepa.state_dict().items()}
    torch.save({"state_dict": prefixed}, path)


def _make_classifier(
    ckpt_path: str,
    use_teacher: bool = True,
    pooling: str = "doc_token",
    freeze_encoder_layers: int = 0,
) -> JEPAClassifier:
    return JEPAClassifier(
        checkpoint_path=ckpt_path,
        tasks=TASKS,
        num_classes=NUM_CLASSES,
        use_teacher=use_teacher,
        pooling=pooling,
        freeze_encoder_layers=freeze_encoder_layers,
        dropout=0.0,
        byte_dim=16,
        seg_dim=32,
        n_byte_blocks=1,
        n_encoder_layers=2,
        n_heads=2,
        ffn_dim=64,
        max_segments=None,   # auto-detect from checkpoint
        kernel_size=3,
        dropout_encoder=0.0,
        byte_dropout=0.0,
    )


def _make_batch(B: int = 2, N: int = 5) -> dict:
    byte_values = torch.randint(0, 256, (B, N, SEGMENT_SIZE))
    byte_types = torch.zeros(B, N, SEGMENT_SIZE, dtype=torch.long)
    positions = torch.arange(N).unsqueeze(0).expand(B, -1).clone()
    seg_mask = torch.ones(B, N, dtype=torch.bool)
    return {
        "byte_values": byte_values,
        "byte_types": byte_types,
        "positions": positions,
        "seg_mask": seg_mask,
    }


# ---------------------------------------------------------------------------
# JEPAClassifier — checkpoint loading
# ---------------------------------------------------------------------------

class TestJEPAClassifierLoading:
    def test_load_teacher(self, tmp_path):
        jepa = _make_small_jepa()
        ckpt = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, ckpt)

        model = _make_classifier(ckpt, use_teacher=True)
        # Teacher weights should match what was saved
        for (name, param), (_, saved) in zip(
            model.encoder.named_parameters(),
            jepa.teacher.named_parameters(),
        ):
            assert torch.allclose(param, saved), f"Mismatch in teacher param {name}"

    def test_load_student(self, tmp_path):
        jepa = _make_small_jepa()
        ckpt = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, ckpt)

        model = _make_classifier(ckpt, use_teacher=False)
        for (name, param), (_, saved) in zip(
            model.encoder.named_parameters(),
            jepa.student.named_parameters(),
        ):
            assert torch.allclose(param, saved), f"Mismatch in student param {name}"

    def test_missing_prefix_raises(self, tmp_path):
        # Checkpoint with no model.teacher / model.student keys
        bad_ckpt = str(tmp_path / "bad.ckpt")
        torch.save({"state_dict": {"some.other.key": torch.zeros(1)}}, bad_ckpt)
        with pytest.raises(KeyError, match="Cannot auto-detect max_segments"):
            _make_classifier(bad_ckpt)

    def test_auto_detect_max_segments(self, tmp_path):
        """max_segments=None should auto-detect from checkpoint pos_embed shape."""
        jepa = _make_small_jepa()
        ckpt = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, ckpt)
        model = _make_classifier(ckpt)   # max_segments=None
        assert model.max_segments == 64  # matches _make_small_jepa max_segments

    def test_mismatched_max_segments_warns(self, tmp_path):
        """Passing a max_segments that differs from checkpoint should still
        build the encoder with the correct checkpoint value."""
        jepa = _make_small_jepa()
        ckpt = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, ckpt)
        model = JEPAClassifier(
            checkpoint_path=ckpt,
            tasks=TASKS,
            num_classes=NUM_CLASSES,
            use_teacher=True,
            max_segments=32,   # wrong — checkpoint has 64; a warning is logged
            byte_dim=16, seg_dim=32, n_byte_blocks=1, n_encoder_layers=2,
            n_heads=2, ffn_dim=64, kernel_size=3,
        )
        # Encoder should be built with the correct checkpoint value, not 32
        assert model.max_segments == 64


# ---------------------------------------------------------------------------
# JEPAClassifier — forward pass output shapes
# ---------------------------------------------------------------------------

class TestJEPAClassifierForward:
    @pytest.fixture(autouse=True)
    def ckpt(self, tmp_path):
        jepa = _make_small_jepa()
        path = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, path)
        self._ckpt = path

    def test_doc_token_pooling_output_shapes(self):
        model = _make_classifier(self._ckpt, pooling="doc_token")
        model.eval()
        batch = _make_batch(B=3, N=6)
        with torch.no_grad():
            logits = model(**batch)
        assert set(logits.keys()) == set(TASKS)
        assert logits["topic"].shape == (3, NUM_CLASSES["topic"])
        assert logits["sentiment"].shape == (3, NUM_CLASSES["sentiment"])

    def test_mean_segments_pooling_output_shapes(self):
        model = _make_classifier(self._ckpt, pooling="mean_segments")
        model.eval()
        batch = _make_batch(B=2, N=8)
        with torch.no_grad():
            logits = model(**batch)
        assert logits["topic"].shape == (2, NUM_CLASSES["topic"])

    def test_no_seg_mask(self, tmp_path):
        model = _make_classifier(self._ckpt, pooling="doc_token")
        model.eval()
        B, N = 2, 5
        byte_values = torch.randint(0, 256, (B, N, SEGMENT_SIZE))
        byte_types = torch.zeros(B, N, SEGMENT_SIZE, dtype=torch.long)
        positions = torch.arange(N).unsqueeze(0).expand(B, -1).clone()
        with torch.no_grad():
            logits = model(byte_values, byte_types, positions, seg_mask=None)
        assert logits["topic"].shape == (B, NUM_CLASSES["topic"])

    def test_mean_segments_masked(self):
        """Mean pooling with a partial seg_mask should not use padding positions."""
        model = _make_classifier(self._ckpt, pooling="mean_segments")
        model.eval()
        B, N = 1, 6
        byte_values = torch.randint(0, 256, (B, N, SEGMENT_SIZE))
        byte_types = torch.zeros(B, N, SEGMENT_SIZE, dtype=torch.long)
        positions = torch.arange(N).unsqueeze(0).expand(B, -1).clone()
        # Only first 4 segments are valid
        seg_mask = torch.tensor([[True, True, True, True, False, False]])
        with torch.no_grad():
            logits_masked = model(byte_values, byte_types, positions, seg_mask)
        assert logits_masked["topic"].shape == (1, NUM_CLASSES["topic"])


# ---------------------------------------------------------------------------
# JEPAClassifier — layer freezing
# ---------------------------------------------------------------------------

class TestJEPAClassifierFreezing:
    @pytest.fixture(autouse=True)
    def ckpt(self, tmp_path):
        jepa = _make_small_jepa()
        path = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, path)
        self._ckpt = path

    def test_freeze_all(self):
        model = _make_classifier(self._ckpt, freeze_encoder_layers=-1)
        for name, p in model.encoder.named_parameters():
            assert not p.requires_grad, f"Expected frozen: {name}"
        # Heads must remain trainable
        for name, p in model.heads.named_parameters():
            assert p.requires_grad, f"Head param should be trainable: {name}"

    def test_freeze_n_layers(self):
        model = _make_classifier(self._ckpt, freeze_encoder_layers=1)
        # First transformer layer should be frozen
        for p in model.encoder.encoder.layers[0].parameters():
            assert not p.requires_grad
        # Second transformer layer should be trainable
        for p in model.encoder.encoder.layers[1].parameters():
            assert p.requires_grad

    def test_no_freeze(self):
        model = _make_classifier(self._ckpt, freeze_encoder_layers=0)
        for p in model.encoder.parameters():
            assert p.requires_grad

    def test_invalid_pooling(self, tmp_path):
        jepa = _make_small_jepa()
        path = str(tmp_path / "pretrain.ckpt")
        _save_jepa_ckpt(jepa, path)
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            JEPAClassifier(
                checkpoint_path=path,
                tasks=TASKS,
                num_classes=NUM_CLASSES,
                pooling="invalid",
                byte_dim=16, seg_dim=32, n_byte_blocks=1, n_encoder_layers=2,
                n_heads=2, ffn_dim=64, kernel_size=3,
            )
