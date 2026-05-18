"""Tests for JEPAClassificationDataset and jepa_collate_fn."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from text_classification.jepa.corruption import SEGMENT_SIZE, CorruptionType, PAD_BYTE
from text_classification.jepa.finetune_data import (
    JEPAClassificationDataset,
    JEPAClassificationDataModule,
    jepa_collate_fn,
)
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASKS = ["topic", "sentiment"]
CLASS_LISTS = {
    "topic": ["politics", "sport", "culture"],
    "sentiment": ["positive", "negative"],
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


SAMPLE_ROWS = [
    {
        "id": "a1",
        "document": "doc1",
        "text": "Hello world, this is a test.",
        "topic": {"classes": ["politics"], "reason": ""},
        "sentiment": {"classes": ["positive"], "reason": ""},
    },
    {
        "id": "a2",
        "document": "doc1",
        "text": "Another example with no sentiment annotation.",
        "topic": {"classes": ["sport", "culture"], "reason": ""},
        "sentiment": None,
    },
    {
        "id": "a3",
        "document": "doc2",
        "text": "Short.",
        "topic": None,
        "sentiment": {"classes": ["negative"], "reason": ""},
    },
]


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestJEPAClassificationDataset:
    @pytest.fixture
    def jsonl_path(self, tmp_path) -> Path:
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, SAMPLE_ROWS)
        return p

    def test_len(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        assert len(ds) == 3

    def test_getitem_segments_type(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        item = ds[0]
        segs = item["segments"]
        assert isinstance(segs, list)
        assert all(isinstance(s, bytes) and len(s) == SEGMENT_SIZE for s in segs)

    def test_getitem_max_segments(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=2)
        item = ds[0]
        assert len(item["segments"]) <= 2

    def test_labels_present(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        item = ds[0]
        assert item["valid_masks"]["topic"].item() is True
        assert item["valid_masks"]["sentiment"].item() is True
        # "politics" is index 0
        assert item["labels"]["topic"][0].item() == 1.0
        assert item["labels"]["topic"][1].item() == 0.0

    def test_missing_annotation_masked(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        item = ds[1]  # sentiment=None
        assert item["valid_masks"]["sentiment"].item() is False
        assert item["labels"]["sentiment"].sum().item() == 0.0

    def test_multilabel(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        item = ds[1]  # topic: [sport, culture]
        assert item["labels"]["topic"][1].item() == 1.0   # sport
        assert item["labels"]["topic"][2].item() == 1.0   # culture
        assert item["labels"]["topic"][0].item() == 0.0   # politics

    def test_num_bytes(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        item = ds[0]
        assert item["num_bytes"] == len(item["segments"]) * SEGMENT_SIZE


# ---------------------------------------------------------------------------
# Collate function tests
# ---------------------------------------------------------------------------

class TestJEPACollateFn:
    @pytest.fixture
    def jsonl_path(self, tmp_path) -> Path:
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, SAMPLE_ROWS)
        return p

    def test_output_keys(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        batch = jepa_collate_fn([ds[0], ds[1]])
        expected_keys = {
            "byte_values", "byte_types", "positions", "seg_mask",
            "labels", "valid_masks", "ids", "documents", "texts", "num_bytes",
        }
        assert expected_keys.issubset(batch.keys())

    def test_tensor_shapes(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        items = [ds[i] for i in range(3)]
        batch = jepa_collate_fn(items)

        B = 3
        N = batch["byte_values"].shape[1]  # padded to longest

        assert batch["byte_values"].shape == (B, N, SEGMENT_SIZE)
        assert batch["byte_types"].shape == (B, N, SEGMENT_SIZE)
        assert batch["positions"].shape == (B, N)
        assert batch["seg_mask"].shape == (B, N)

    def test_padding_values(self, jsonl_path):
        """Padding segments should have PAD_BYTE values and PADDING type."""
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        # Use one very short and one longer example to force padding
        items = [ds[2], ds[0]]   # "Short." vs longer text
        batch = jepa_collate_fn(items)

        short_n = len(ds[2]["segments"])
        pad_mask = ~batch["seg_mask"][0]  # False = padding for first example (short one)
        if pad_mask.any():
            assert (batch["byte_values"][0][pad_mask] == PAD_BYTE).all()
            assert (batch["byte_types"][0][pad_mask] == int(CorruptionType.PADDING)).all()

    def test_real_segments_clean_type(self, jsonl_path):
        """Real (non-padding) segments should have CLEAN byte types."""
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        items = [ds[0]]
        batch = jepa_collate_fn(items)

        seg_mask = batch["seg_mask"][0]  # (N,) bool
        real_types = batch["byte_types"][0][seg_mask]  # (n_real, SEGMENT_SIZE)
        assert (real_types == int(CorruptionType.CLEAN)).all()

    def test_positions_sequential(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        batch = jepa_collate_fn([ds[0]])
        N = batch["positions"].shape[1]
        expected = torch.arange(N).unsqueeze(0)
        assert (batch["positions"] == expected).all()

    def test_labels_stacked(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        items = [ds[0], ds[1]]
        batch = jepa_collate_fn(items)
        assert batch["labels"]["topic"].shape == (2, len(CLASS_LISTS["topic"]))
        assert batch["valid_masks"]["topic"].shape == (2,)

    def test_num_bytes_sum(self, jsonl_path):
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        items = [ds[0], ds[1]]
        expected_bytes = items[0]["num_bytes"] + items[1]["num_bytes"]
        batch = jepa_collate_fn(items)
        assert batch["num_bytes"] == expected_bytes

    def test_byte_values_range(self, jsonl_path):
        """Real byte values must be in 0-255; padding segments use PAD_BYTE=256."""
        ds = JEPAClassificationDataset(jsonl_path, TASKS, CLASS_LISTS, max_segments=32)
        batch = jepa_collate_fn([ds[i] for i in range(3)])
        # Real bytes
        real = batch["byte_values"][batch["seg_mask"]]
        assert (real >= 0).all() and (real <= 255).all()


# ---------------------------------------------------------------------------
# DataModule tests
# ---------------------------------------------------------------------------

class TestJEPAClassificationDataModule:
    def _make_cfg(self, train_path: str, val_path: str) -> object:
        return OmegaConf.create({
            "data": {
                "train": train_path,
                "val": [{"path": val_path, "name": "val"}],
                "max_segments": 16,
                "num_workers": 0,
            },
            "training": {"batch_size": 2, "val_batch_size": 2},
        })

    def test_setup_and_dataloaders(self, tmp_path):
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, SAMPLE_ROWS)
        cfg = self._make_cfg(str(p), str(p))
        dm = JEPAClassificationDataModule(cfg, TASKS, CLASS_LISTS)
        dm.setup("fit")

        assert len(dm.val_dataset_names) == 1
        train_dl = dm.train_dataloader()
        val_dls = dm.val_dataloader()
        assert len(val_dls) == 1

        # Check one train batch
        batch = next(iter(train_dl))
        assert "byte_values" in batch
        assert batch["byte_values"].dtype == torch.long
