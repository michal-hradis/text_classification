"""PyTorch Lightning module for finetuning a pretrained JEPA encoder on
multi-task multi-label text classification.

Subclasses :class:`~text_classification.training.lightning_module.TextClassificationModule`
and overrides only the parts that differ for JEPA:

1. ``__init__`` — creates :class:`~text_classification.jepa.classifier.JEPAClassifier`
   instead of ``TransformerClassifier``.
2. ``forward`` — routes through the JEPA encoder interface
   ``(byte_values, byte_types, positions, seg_mask)``.
3. ``training_step`` / ``validation_step`` — extract JEPA-specific batch keys
   and use ``num_bytes`` for throughput tracking.

All other methods are inherited unchanged:

- ``_compute_loss`` — masked multi-task BCE loss.
- ``on_before_optimizer_step`` — gradient-norm logging.
- ``on_validation_epoch_start`` / ``on_validation_epoch_end`` — metric reset
  and logging.
- ``configure_optimizers`` — AdamW + LR scheduler.
"""
from __future__ import annotations

import time
import logging
from typing import Any, Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from text_classification.jepa.classifier import JEPAClassifier
from text_classification.metrics.multilabel import MultiLabelMetrics
from text_classification.training.lightning_module import TextClassificationModule
from text_classification.utils.optimizers import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class JEPAFinetuneModule(TextClassificationModule):
    """Lightning module that wraps :class:`JEPAClassifier` for finetuning.

    Args:
        cfg:               Full OmegaConf config (see ``configs/jepa_finetune_base.yaml``).
        tasks:             Ordered list of task names.
        class_lists:       Mapping from task name to ordered list of class names.
        val_dataset_names: Names assigned to each validation DataLoader.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tasks: list[str],
        class_lists: dict[str, list[str]],
        val_dataset_names: Optional[list[str]] = None,
    ) -> None:
        # Skip TextClassificationModule.__init__ (which creates TransformerClassifier)
        # and initialise the Lightning base directly, then set up everything ourselves.
        pl.LightningModule.__init__(self)

        self.cfg = cfg
        self.tasks = tasks
        self.class_lists = class_lists
        self.val_dataset_names: list[str] = val_dataset_names or ["val"]
        torch.set_float32_matmul_precision("medium")

        model_cfg = cfg.model
        # max_segments is auto-detected from the checkpoint; passing the config
        # value only so that a warning is emitted when it doesn't match.
        cfg_max_segments = cfg.data.get("max_segments", None)
        self.model = JEPAClassifier(
            checkpoint_path=str(model_cfg.checkpoint_path),
            tasks=tasks,
            num_classes={t: len(class_lists[t]) for t in tasks},
            use_teacher=bool(model_cfg.get("use_teacher", True)),
            pooling=str(model_cfg.get("pooling", "doc_token")),
            freeze_encoder_layers=int(model_cfg.get("freeze_encoder_layers", 0)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            # Architecture params — must match the pretrained checkpoint
            byte_dim=int(model_cfg.get("byte_dim", 256)),
            seg_dim=int(model_cfg.get("seg_dim", 512)),
            n_byte_blocks=int(model_cfg.get("n_byte_blocks", 4)),
            n_encoder_layers=int(model_cfg.get("n_encoder_layers", 12)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            ffn_dim=int(model_cfg.get("ffn_dim", 2048)),
            max_segments=int(cfg_max_segments) if cfg_max_segments is not None else None,
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            dropout_encoder=float(model_cfg.get("dropout_encoder", 0.1)),
            byte_dropout=float(model_cfg.get("byte_dropout", 0.05)),
        )
        # Warn if the dataset would generate more segments than the encoder supports
        ckpt_max_segs = self.model.max_segments
        if cfg_max_segments is not None and int(cfg_max_segments) > ckpt_max_segs:
            logger.warning(
                "data.max_segments=%d exceeds the encoder's checkpoint max_segments=%d. "
                "Input will be silently truncated to %d segments by the encoder's pos_embed. "
                "Set data.max_segments: %d in your config to silence this warning.",
                cfg_max_segments, ckpt_max_segs, ckpt_max_segs, ckpt_max_segs,
            )
        if model_cfg.get("compile", False):
            self.model = torch.compile(self.model)

        # Per-dataset per-task metric accumulators (inherited by other methods)
        self._val_metrics: dict[str, dict[str, MultiLabelMetrics]] = {
            ds: {task: MultiLabelMetrics(class_lists[task]) for task in tasks}
            for ds in self.val_dataset_names
        }

        # Confident-error buffer: (ds, task) -> list[dict]
        self._confident_errors: dict[str, list[dict[str, Any]]] = {}

        # Throughput counters
        self._total_tokens: int = 0
        self._val_tokens: int = 0
        self._val_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        byte_values: torch.Tensor,
        byte_types: torch.Tensor,
        positions: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return self.model(byte_values, byte_types, positions, seg_mask)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["byte_values"],
            batch["byte_types"],
            batch["positions"],
            batch["seg_mask"],
        )
        loss = self._compute_loss(logits, batch["labels"], batch["valid_masks"])

        self._total_tokens += batch["num_bytes"]

        n_batches = max(1, self.trainer.num_training_batches)
        epoch_frac = self.trainer.current_epoch + batch_idx / n_batches

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/epoch", epoch_frac, on_step=True, on_epoch=False)
        self.log("train/tokens_total", float(self._total_tokens), on_step=True, on_epoch=False)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        ds_name = (
            self.val_dataset_names[dataloader_idx]
            if dataloader_idx < len(self.val_dataset_names)
            else f"val_{dataloader_idx}"
        )

        logits = self(
            batch["byte_values"],
            batch["byte_types"],
            batch["positions"],
            batch["seg_mask"],
        )
        self._val_tokens += batch["num_bytes"]

        loss = self._compute_loss(logits, batch["labels"], batch["valid_masks"])
        self.log(
            f"val/{ds_name}/loss",
            loss,
            add_dataloader_idx=False,
            on_step=False,
            on_epoch=True,
        )

        for task in self.tasks:
            self._val_metrics[ds_name][task].update(
                logits[task],
                batch["labels"][task],
                batch["valid_masks"][task],
            )
            self._collect_confident_errors(
                ds_name,
                task,
                logits[task],
                batch["labels"][task],
                batch["valid_masks"][task],
                batch.get("texts", []),
            )
