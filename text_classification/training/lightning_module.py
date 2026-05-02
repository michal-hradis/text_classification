"""PyTorch Lightning module for multi-task multi-label text classification.

Training is driven by **number of optimizer updates** (``global_step``), not
epochs.  Key features:

- bf16-mixed precision
- Multi-GPU via Lightning ``strategy`` setting
- Gradient clipping / normalization (delegated to ``Trainer``)
- Per-step LR scheduler support
- Multiple validation datasets tracked independently
- Logs: loss, mAP, per-class metrics, gradient norm, LR, token throughput,
  epoch fraction, inference speed
- Most-confident validation errors printed to stdout for qualitative analysis
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig

from text_classification.models.classifier import TransformerClassifier
from text_classification.metrics.multilabel import MultiLabelMetrics

logger = logging.getLogger(__name__)

# Maximum number of confident errors kept per (dataset, task) pair per val run
_MAX_CONFIDENT_ERRORS = 20
# Minimum per-class probability error to qualify as a "confident error"
_CONFIDENT_ERROR_THRESHOLD = 0.5


class TextClassificationModule(pl.LightningModule):
    """Lightning module that wraps ``TransformerClassifier`` for training.

    Args:
        cfg: Full OmegaConf config (see ``configs/base.yaml``).
        tasks: Ordered list of task names.
        class_lists: Mapping from task name to ordered list of class names.
        val_dataset_names: Names assigned to each validation DataLoader.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tasks: list[str],
        class_lists: dict[str, list[str]],
        val_dataset_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tasks = tasks
        self.class_lists = class_lists
        self.val_dataset_names: list[str] = val_dataset_names or ["val"]

        self.model = TransformerClassifier(
            model_name_or_path=cfg.model.name_or_path,
            tasks=tasks,
            num_classes={t: len(class_lists[t]) for t in tasks},
            dropout=cfg.model.get("dropout", 0.1),
            pooling=cfg.model.get("pooling", "cls"),
            freeze_encoder_layers=cfg.model.get("freeze_encoder_layers", 0),
        )

        # Per-dataset per-task metric accumulators
        self._val_metrics: dict[str, dict[str, MultiLabelMetrics]] = {
            ds: {task: MultiLabelMetrics(class_lists[task]) for task in tasks}
            for ds in self.val_dataset_names
        }

        # Confident errors buffer: (ds, task) -> list of error dicts
        self._confident_errors: dict[str, list[dict[str, Any]]] = {}

        # Token counters
        self._total_tokens: int = 0
        self._val_tokens: int = 0
        self._val_start_time: float = 0.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return self.model(input_ids, attention_mask, token_type_ids)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        valid_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Masked multi-task BCE loss.

        Each task's loss is computed only over examples with valid GT, then
        the task losses are averaged uniformly.
        """
        total = torch.zeros(1, device=self.device)
        n_valid_tasks = 0

        for task in self.tasks:
            mask = valid_masks[task]  # (B,) bool
            if not mask.any():
                continue
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[task][mask], labels[task][mask], reduction="mean"
            )
            total = total + loss
            n_valid_tasks += 1

        if n_valid_tasks > 0:
            total = total / n_valid_tasks
        return total.squeeze()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
        )
        loss = self._compute_loss(logits, batch["labels"], batch["valid_masks"])

        self._total_tokens += batch["num_tokens"]

        # Epoch fraction (useful when max_steps drives training)
        n_batches = max(1, self.trainer.num_training_batches)
        epoch_frac = self.trainer.current_epoch + batch_idx / n_batches

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/epoch", epoch_frac, on_step=True, on_epoch=False)
        self.log("train/tokens_total", float(self._total_tokens), on_step=True, on_epoch=False)

        return loss

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Log gradient norm before each optimizer step."""
        sq_sum = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        self.log("train/grad_norm", sq_sum ** 0.5, on_step=True, on_epoch=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_start_time = time.perf_counter()
        self._val_tokens = 0
        self._confident_errors = {}
        for ds_metrics in self._val_metrics.values():
            for m in ds_metrics.values():
                m.reset()

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
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
        )
        self._val_tokens += batch["num_tokens"]

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

    def _collect_confident_errors(
        self,
        ds_name: str,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
        texts: list[str],
    ) -> None:
        key = f"{ds_name}/{task}"
        if key not in self._confident_errors:
            self._confident_errors[key] = []

        if not valid_mask.any():
            return

        buf = self._confident_errors[key]
        if len(buf) >= _MAX_CONFIDENT_ERRORS:
            return

        probs = torch.sigmoid(logits).detach().float().cpu()
        tgts = labels.detach().float().cpu()
        valid_idx = valid_mask.nonzero(as_tuple=True)[0].tolist()

        for i in valid_idx:
            error_mag = float((probs[i] - tgts[i]).abs().max())
            if error_mag >= _CONFIDENT_ERROR_THRESHOLD:
                buf.append(
                    {
                        "error": error_mag,
                        "probs": probs[i].numpy().tolist(),
                        "target": tgts[i].numpy().tolist(),
                        "text": texts[i][:300] if i < len(texts) else "",
                    }
                )
            if len(buf) >= _MAX_CONFIDENT_ERRORS:
                break

    def on_validation_epoch_end(self) -> None:
        elapsed = time.perf_counter() - self._val_start_time
        speed = self._val_tokens / max(elapsed, 1e-9)
        self.log("val/inference_tokens_per_sec", speed)

        # Log per-dataset per-task metrics
        for ds_name, ds_metrics in self._val_metrics.items():
            for task, metric in ds_metrics.items():
                results = metric.compute()
                for name, value in results.items():
                    if value != value:  # skip NaN
                        continue
                    self.log(
                        f"val/{ds_name}/{task}/{name}",
                        value,
                        add_dataloader_idx=False,
                        on_step=False,
                        on_epoch=True,
                    )

        # Print most confident errors for qualitative analysis
        for key, errors in self._confident_errors.items():
            if not errors:
                continue
            top = sorted(errors, key=lambda e: -e["error"])[:5]
            logger.info("Most confident errors — %s (step=%d):", key, self.global_step)
            for e in top:
                cls_names = self.class_lists[key.split("/", 1)[1]]
                pred_classes = [
                    cls_names[i]
                    for i, p in enumerate(e["probs"])
                    if p >= 0.5
                ]
                true_classes = [
                    cls_names[i]
                    for i, t in enumerate(e["target"])
                    if t >= 0.5
                ]
                logger.info(
                    "  err=%.3f | pred=%s | true=%s | text=%r",
                    e["error"],
                    pred_classes,
                    true_classes,
                    e["text"],
                )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        opt_cfg = self.cfg.optimizer
        sched_cfg = self.cfg.get("scheduler", None)

        # Weight-decay group split (exclude bias and LayerNorm weights)
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opt_cfg.get("weight_decay", 0.01),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
        )

        if sched_cfg is None:
            return optimizer

        scheduler = self._build_scheduler(optimizer, sched_cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": sched_cfg.get("interval", "step"),
                "frequency": sched_cfg.get("frequency", 1),
                "monitor": sched_cfg.get("monitor", "val/loss"),
            },
        }

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, cfg: DictConfig
    ) -> Any:
        from torch.optim import lr_scheduler

        name: str = cfg.name

        if name == "cosine":
            total = cfg.get("T_max", self.trainer.max_steps if self.trainer.max_steps > 0 else 10_000)
            return lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total, eta_min=cfg.get("eta_min", 0.0)
            )

        if name == "linear_warmup_cosine":
            from transformers import get_cosine_schedule_with_warmup

            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.warmup_steps,
                num_training_steps=cfg.total_steps,
            )

        if name == "linear_warmup":
            from transformers import get_linear_schedule_with_warmup

            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.warmup_steps,
                num_training_steps=cfg.total_steps,
            )

        if name == "constant_warmup":
            from transformers import get_constant_schedule_with_warmup

            return get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.warmup_steps
            )

        if name == "reduce_on_plateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.get("mode", "min"),
                factor=cfg.get("factor", 0.5),
                patience=cfg.get("patience", 5),
            )

        if name == "one_cycle":
            total = cfg.get("total_steps", self.trainer.max_steps)
            return lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.get("max_lr", self.cfg.optimizer.lr),
                total_steps=total,
                pct_start=cfg.get("pct_start", 0.3),
            )

        raise ValueError(f"Unknown scheduler: {name!r}")
