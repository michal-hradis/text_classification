"""Dataset and DataModule for finetuning JEPA models on text classification.

Mirrors :mod:`text_classification.data.dataset` but replaces the HuggingFace
tokenizer with the JEPA byte-segment encoder so that batches contain byte
tensors instead of token-id tensors.

Input JSONL format is identical to the standard pipeline (same ``text``,
``id``, ``document``, and per-task ground-truth fields).  The ``text`` field
is converted to UTF-8 bytes and split into fixed-size canonical segments using
:func:`text_to_canonical_segments`.

Batch keys produced by the collate function:

``byte_values``   ``LongTensor(B, N, SEGMENT_SIZE)`` — raw byte values.
                  Real bytes are in 0-255; padding segments use ``PAD_BYTE=256``.
``byte_types``    ``LongTensor(B, N, SEGMENT_SIZE)`` — per-byte
                  :class:`CorruptionType`.  All ``CLEAN=0`` for real bytes,
                  ``PADDING=6`` for padding segments.
``positions``     ``LongTensor(B, N)`` — canonical segment indices 0 … N-1.
``seg_mask``      ``BoolTensor(B, N)`` — ``True`` for valid segments.
``num_bytes``     ``int`` — total real bytes in the batch (for throughput logging).
``labels``        ``dict[task, FloatTensor(B, C)]`` — multi-hot ground truth.
``valid_masks``   ``dict[task, BoolTensor(B)]`` — annotation presence flag.
``ids``           ``list[str]``
``documents``     ``list[str]``
``texts``         ``list[str]``
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from omegaconf import DictConfig

from text_classification.jepa.corruption import (
    SEGMENT_SIZE,
    PAD_BYTE,
    CorruptionType,
    text_to_canonical_segments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JEPAClassificationDataset(Dataset):
    """Multi-task multi-label classification dataset using JEPA byte segments.

    All JSONL loading and ground-truth encoding is identical to
    :class:`~text_classification.data.dataset.TextClassificationDataset`; the
    only difference is that the text is encoded as a list of fixed-size byte
    segments rather than subword token ids.

    Args:
        path:         Path to a JSONL file.
        tasks:        Ordered list of task names.
        class_lists:  Mapping from task name to ordered list of class names.
        max_segments: Maximum number of 16-byte segments per example.  Texts
                      longer than ``max_segments × SEGMENT_SIZE`` bytes are
                      truncated from the right (first ``max_segments`` segments
                      are kept).
    """

    def __init__(
        self,
        path: str | Path,
        tasks: list[str],
        class_lists: dict[str, list[str]],
        max_segments: int = 512,
    ) -> None:
        self.path = Path(path)
        self.tasks = tasks
        self.class_lists = class_lists
        self.max_segments = max_segments
        self._class_to_idx: dict[str, dict[str, int]] = {
            task: {cls: i for i, cls in enumerate(classes)}
            for task, classes in class_lists.items()
        }
        self.examples = self._load()

    def _load(self) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        logger.info("Loaded %d examples from %s", len(examples), self.path)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        text: str = example["text"]

        # Convert to JEPA byte segments (list of bytes objects, each SEGMENT_SIZE long)
        segments: list[bytes] = text_to_canonical_segments(text, self.max_segments)

        labels: dict[str, torch.Tensor] = {}
        valid_masks: dict[str, torch.Tensor] = {}

        for task in self.tasks:
            n_classes = len(self.class_lists[task])
            c2i = self._class_to_idx[task]
            gt = example.get(task)

            if gt is not None:
                label_vec = torch.zeros(n_classes, dtype=torch.float32)
                for cls in gt.get("classes", []):
                    if cls in c2i:
                        label_vec[c2i[cls]] = 1.0
                labels[task] = label_vec
                valid_masks[task] = torch.tensor(True)
            else:
                labels[task] = torch.zeros(n_classes, dtype=torch.float32)
                valid_masks[task] = torch.tensor(False)

        return {
            "segments": segments,      # list[bytes], each SEGMENT_SIZE bytes
            "labels": labels,
            "valid_masks": valid_masks,
            "id": example.get("id", ""),
            "document": example.get("document", ""),
            "text": text,
            "num_bytes": len(segments) * SEGMENT_SIZE,
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def jepa_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of JEPA classification examples into a padded batch.

    Variable-length segment lists are padded to the longest sequence in the
    batch.  Padding segments are filled with ``PAD_BYTE`` and given
    ``CorruptionType.PADDING`` byte-type labels.
    """
    max_segs = max(len(item["segments"]) for item in batch)

    pad_byte_val = PAD_BYTE
    pad_type_val = int(CorruptionType.PADDING)
    clean_type_val = int(CorruptionType.CLEAN)

    byte_values_list: list[torch.Tensor] = []
    byte_types_list: list[torch.Tensor] = []
    positions_list: list[torch.Tensor] = []
    seg_mask_list: list[torch.Tensor] = []

    for item in batch:
        segs = item["segments"]
        n = len(segs)
        pad = max_segs - n

        # Real segments: byte values (0-255) + CLEAN types
        real_vals = torch.tensor(
            [[b for b in seg] for seg in segs], dtype=torch.long
        )                                                                # (n, SEGMENT_SIZE)
        real_types = torch.full((n, SEGMENT_SIZE), clean_type_val, dtype=torch.long)

        if pad > 0:
            pad_vals = torch.full((pad, SEGMENT_SIZE), pad_byte_val, dtype=torch.long)
            pad_types = torch.full((pad, SEGMENT_SIZE), pad_type_val, dtype=torch.long)
            bv = torch.cat([real_vals, pad_vals], dim=0)                # (max_segs, S)
            bt = torch.cat([real_types, pad_types], dim=0)
        else:
            bv = real_vals
            bt = real_types

        positions = torch.arange(max_segs, dtype=torch.long)
        mask = torch.cat([
            torch.ones(n, dtype=torch.bool),
            torch.zeros(pad, dtype=torch.bool),
        ])

        byte_values_list.append(bv)
        byte_types_list.append(bt)
        positions_list.append(positions)
        seg_mask_list.append(mask)

    tasks = list(batch[0]["labels"].keys())
    labels = {task: torch.stack([item["labels"][task] for item in batch]) for task in tasks}
    valid_masks = {
        task: torch.stack([item["valid_masks"][task] for item in batch]) for task in tasks
    }

    return {
        "byte_values": torch.stack(byte_values_list),    # (B, N, SEGMENT_SIZE)
        "byte_types": torch.stack(byte_types_list),      # (B, N, SEGMENT_SIZE)
        "positions": torch.stack(positions_list),         # (B, N)
        "seg_mask": torch.stack(seg_mask_list),           # (B, N)
        "labels": labels,
        "valid_masks": valid_masks,
        "ids": [item["id"] for item in batch],
        "documents": [item["document"] for item in batch],
        "texts": [item["text"] for item in batch],
        "num_bytes": sum(item["num_bytes"] for item in batch),
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class JEPAClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for JEPA-based multi-task text classification.

    Identical to :class:`~text_classification.data.dataset.TextClassificationDataModule`
    except that it uses :class:`JEPAClassificationDataset` and
    :func:`jepa_collate_fn` instead of tokenizer-based equivalents.

    Config keys consumed:
        data.train (str):                  path to training JSONL.
        data.val (list[{path, name}]|str): validation JSONL path(s).
        data.max_segments (int):           max byte segments per example.
        data.num_workers (int):            DataLoader worker count.
        training.batch_size (int):         training batch size.
        training.val_batch_size (int):     validation batch size.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tasks: list[str],
        class_lists: dict[str, list[str]],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tasks = tasks
        self.class_lists = class_lists
        self._val_datasets: dict[str, JEPAClassificationDataset] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        max_segs = self.cfg.data.get("max_segments", 512)
        if stage in ("fit", None):
            self.train_dataset = JEPAClassificationDataset(
                path=self.cfg.data.train,
                tasks=self.tasks,
                class_lists=self.class_lists,
                max_segments=max_segs,
            )

        val_configs = self.cfg.data.get("val", [])
        if isinstance(val_configs, str):
            val_configs = [{"path": val_configs, "name": "val"}]

        self._val_datasets = {}
        for i, vc in enumerate(val_configs):
            name = vc.get("name", f"val_{i}")
            self._val_datasets[name] = JEPAClassificationDataset(
                path=vc["path"],
                tasks=self.tasks,
                class_lists=self.class_lists,
                max_segments=max_segs,
            )

    def train_dataloader(self) -> DataLoader:
        num_workers = self.cfg.data.get("num_workers", 4)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=jepa_collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    def val_dataloader(self) -> list[DataLoader]:
        val_bs = self.cfg.training.get("val_batch_size", self.cfg.training.batch_size)
        num_workers = self.cfg.data.get("num_workers", 4)
        return [
            DataLoader(
                ds,
                batch_size=val_bs,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=jepa_collate_fn,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )
            for ds in self._val_datasets.values()
        ]

    @property
    def val_dataset_names(self) -> list[str]:
        return list(self._val_datasets.keys())
