"""Dataset loading and processing for multi-label text classification.

Each JSONL row is an example with at minimum:
    - text (str): document text segment
    - id (str): UUID of the text sample
    - document (str): UUID of the source document
    - <task_name> (dict | None): ground truth for a classification task with
        keys ``classes`` (list[str]) and ``reason`` (str).  Missing when not
        annotated.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

import pytorch_lightning as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """Multi-task multi-label text classification dataset loaded from a JSONL file.

    Handles missing ground truth via a per-task validity mask so that examples
    without annotations for a task are ignored in the loss for that task.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        tasks: list[str],
        class_lists: dict[str, list[str]],
        max_length: int = 512,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.class_lists = class_lists
        self.max_length = max_length
        # Pre-build reverse lookup: task -> {class_name: index}
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

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

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

        token_type_ids = encoding.get("token_type_ids")
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "token_type_ids": (
                torch.tensor(token_type_ids, dtype=torch.long)
                if token_type_ids is not None
                else None
            ),
            "labels": labels,
            "valid_masks": valid_masks,
            "id": example.get("id", ""),
            "document": example.get("document", ""),
            "text": text,
            "num_tokens": len(encoding["input_ids"]),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of examples into a padded batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    def _pad(t: torch.Tensor, pad_val: int = 0) -> torch.Tensor:
        return F.pad(t, (0, max_len - t.size(0)), value=pad_val)

    input_ids = torch.stack([_pad(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([_pad(item["attention_mask"]) for item in batch])

    has_tti = batch[0]["token_type_ids"] is not None
    token_type_ids: Optional[torch.Tensor] = (
        torch.stack([_pad(item["token_type_ids"]) for item in batch])
        if has_tti
        else None
    )

    tasks = list(batch[0]["labels"].keys())
    labels = {task: torch.stack([item["labels"][task] for item in batch]) for task in tasks}
    valid_masks = {
        task: torch.stack([item["valid_masks"][task] for item in batch]) for task in tasks
    }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
        "valid_masks": valid_masks,
        "ids": [item["id"] for item in batch],
        "documents": [item["document"] for item in batch],
        "texts": [item["text"] for item in batch],
        "num_tokens": sum(item["num_tokens"] for item in batch),
    }


class TextClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-task multi-label text classification.

    Config keys consumed:
        data.train (str): path to training JSONL
        data.val (list[{path, name}] | str): validation JSONL path(s)
        data.max_length (int): tokenizer max length
        data.num_workers (int): DataLoader worker count
        training.batch_size (int): training batch size
        training.val_batch_size (int): validation batch size
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
        tasks: list[str],
        class_lists: dict[str, list[str]],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tasks = tasks
        self.class_lists = class_lists
        self._val_datasets: dict[str, TextClassificationDataset] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            if stage != "validate":
                self.train_dataset = TextClassificationDataset(
                    path=self.cfg.data.train,
                    tokenizer=self.tokenizer,
                    tasks=self.tasks,
                    class_lists=self.class_lists,
                    max_length=self.cfg.data.max_length,
                )

            val_configs = self.cfg.data.get("val", [])
            # Normalise: accept a plain string path as a single val set
            if isinstance(val_configs, str):
                val_configs = [{"path": val_configs, "name": "val"}]

            self._val_datasets = {}
            for i, vc in enumerate(val_configs):
                name = vc.get("name", f"val_{i}")
                self._val_datasets[name] = TextClassificationDataset(
                    path=vc["path"],
                    tokenizer=self.tokenizer,
                    tasks=self.tasks,
                    class_lists=self.class_lists,
                    max_length=self.cfg.data.max_length,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.get("num_workers", 4),
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        val_bs = self.cfg.training.get("val_batch_size", self.cfg.training.batch_size)
        return [
            DataLoader(
                ds,
                batch_size=val_bs,
                shuffle=False,
                num_workers=self.cfg.data.get("num_workers", 4),
                collate_fn=collate_fn,
                pin_memory=True,
            )
            for ds in self._val_datasets.values()
        ]

    @property
    def val_dataset_names(self) -> list[str]:
        return list(self._val_datasets.keys())
