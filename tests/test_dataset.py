"""Unit tests for the dataset module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from text_classification.data.dataset import (
    TextClassificationDataset,
    collate_fn,
)

# Use a tiny, fast tokenizer that doesn't require downloading a real model.
# bert-base-uncased is small enough and is widely available.
TOKENIZER_NAME = "bert-base-uncased"
TASKS = ["topic", "sentiment"]
CLASS_LISTS = {
    "topic": ["politics", "economy", "culture"],
    "sentiment": ["positive", "negative", "neutral"],
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "aaa",
            "document": "doc1",
            "text": "Hello world, this is a test.",
            "topic": {"classes": ["politics", "culture"], "reason": "test"},
            "sentiment": {"classes": ["positive"], "reason": "test"},
        },
        {
            "id": "bbb",
            "document": "doc1",
            "text": "Another document with missing sentiment.",
            "topic": {"classes": ["economy"], "reason": "test"},
            # sentiment is missing → should be masked
        },
        {
            "id": "ccc",
            "document": "doc2",
            "text": "No annotations at all.",
        },
    ]
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, rows)
    return p


class TestTextClassificationDataset:
    def test_len(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        assert len(ds) == 3

    def test_label_encoding(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        item = ds[0]
        # "politics" = index 0, "culture" = index 2
        topic_label = item["labels"]["topic"]
        assert topic_label[0] == 1.0
        assert topic_label[1] == 0.0
        assert topic_label[2] == 1.0
        # sentiment valid
        assert item["valid_masks"]["sentiment"].item() is True

    def test_missing_gt_creates_zero_label_and_invalid_mask(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        item = ds[1]  # missing sentiment
        assert item["valid_masks"]["sentiment"].item() is False
        assert item["labels"]["sentiment"].sum().item() == 0.0

    def test_no_annotations(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        item = ds[2]  # no annotations at all
        for task in TASKS:
            assert item["valid_masks"][task].item() is False

    def test_tensor_shapes(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        item = ds[0]
        seq_len = item["input_ids"].size(0)
        assert item["attention_mask"].shape == torch.Size([seq_len])
        assert item["labels"]["topic"].shape == torch.Size([3])
        assert item["labels"]["sentiment"].shape == torch.Size([3])

    def test_max_length_truncation(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(
            sample_jsonl, tokenizer, TASKS, CLASS_LISTS, max_length=8
        )
        item = ds[0]
        assert item["input_ids"].size(0) <= 8

    def test_num_tokens_positive(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        assert ds[0]["num_tokens"] > 0


class TestCollateFn:
    def test_output_keys(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        batch = collate_fn([ds[0], ds[1]])
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "valid_masks" in batch
        assert "num_tokens" in batch

    def test_padding(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        batch = collate_fn([ds[0], ds[1]])
        B, T = batch["input_ids"].shape
        assert B == 2
        # All sequences in the batch must have the same length
        assert batch["attention_mask"].shape == (B, T)

    def test_stacked_labels(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        batch = collate_fn([ds[0], ds[1]])
        assert batch["labels"]["topic"].shape == (2, 3)
        assert batch["valid_masks"]["topic"].shape == (2,)

    def test_num_tokens_sum(self, sample_jsonl, tokenizer):
        ds = TextClassificationDataset(sample_jsonl, tokenizer, TASKS, CLASS_LISTS)
        items = [ds[0], ds[1]]
        expected = items[0]["num_tokens"] + items[1]["num_tokens"]
        batch = collate_fn(items)
        assert batch["num_tokens"] == expected
