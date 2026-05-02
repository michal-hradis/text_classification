"""Unit tests for the model module."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from text_classification.models.classifier import ClassificationHead, TransformerClassifier


# ---------------------------------------------------------------------------
# ClassificationHead
# ---------------------------------------------------------------------------

class TestClassificationHead:
    def test_output_shape(self):
        head = ClassificationHead(hidden_size=64, num_classes=5)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 5)

    def test_dropout_zero_deterministic(self):
        """With dropout=0 the output should be the same in train and eval."""
        head = ClassificationHead(hidden_size=16, num_classes=3, dropout=0.0)
        x = torch.randn(2, 16)
        head.train()
        out_train = head(x)
        head.eval()
        out_eval = head(x)
        assert torch.allclose(out_train, out_eval)


# ---------------------------------------------------------------------------
# Fake encoder for testing without downloading real weights
# ---------------------------------------------------------------------------

class _FakeOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    """Minimal BERT-like encoder that just returns random hidden states."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dummy = nn.Linear(1, 1)  # so named_parameters() is non-empty

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> _FakeOutput:
        B, T = input_ids.shape
        return _FakeOutput(torch.randn(B, T, self.hidden_size))


# ---------------------------------------------------------------------------
# TransformerClassifier
# ---------------------------------------------------------------------------

TASKS = ["task_a", "task_b"]
NUM_CLASSES = {"task_a": 4, "task_b": 2}
HIDDEN = 64


@pytest.fixture()
def classifier() -> TransformerClassifier:
    return TransformerClassifier.from_custom_encoder(
        encoder=_FakeEncoder(HIDDEN),
        hidden_size=HIDDEN,
        tasks=TASKS,
        num_classes=NUM_CLASSES,
    )


class TestTransformerClassifier:
    def test_forward_returns_all_tasks(self, classifier):
        B, T = 3, 16
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        out = classifier(input_ids, attention_mask)
        assert set(out.keys()) == set(TASKS)

    def test_output_shapes(self, classifier):
        B, T = 5, 12
        input_ids = torch.randint(0, 1000, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        out = classifier(input_ids, attention_mask)
        assert out["task_a"].shape == (B, NUM_CLASSES["task_a"])
        assert out["task_b"].shape == (B, NUM_CLASSES["task_b"])

    def test_token_type_ids_forwarded(self, classifier):
        """Passing token_type_ids should not raise an error (they are accepted via **kwargs)."""
        B, T = 2, 8
        input_ids = torch.randint(0, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        token_type_ids = torch.zeros(B, T, dtype=torch.long)
        out = classifier(input_ids, attention_mask, token_type_ids)
        assert "task_a" in out

    def test_cls_pooling(self, classifier):
        """CLS pooling should use position 0 of last_hidden_state."""
        classifier.pooling = "cls"
        B, T = 2, 6
        input_ids = torch.zeros(B, T, dtype=torch.long)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        # Just ensure no error and correct shape
        out = classifier(input_ids, attention_mask)
        assert out["task_a"].shape == (B, NUM_CLASSES["task_a"])

    def test_mean_pooling(self, classifier):
        classifier.pooling = "mean"
        B, T = 2, 6
        input_ids = torch.zeros(B, T, dtype=torch.long)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        out = classifier(input_ids, attention_mask)
        assert out["task_a"].shape == (B, NUM_CLASSES["task_a"])

    def test_invalid_pooling_raises(self, classifier):
        classifier.pooling = "invalid"
        B, T = 2, 6
        input_ids = torch.zeros(B, T, dtype=torch.long)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        with pytest.raises(ValueError, match="Unknown pooling"):
            classifier(input_ids, attention_mask)

    def test_eval_mode_no_grad(self, classifier):
        """In eval mode, forward should work without tracking gradients."""
        classifier.eval()
        B, T = 2, 8
        input_ids = torch.zeros(B, T, dtype=torch.long)
        attention_mask = torch.ones(B, T, dtype=torch.long)
        with torch.no_grad():
            out = classifier(input_ids, attention_mask)
        assert out["task_a"].requires_grad is False
