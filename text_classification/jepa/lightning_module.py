"""PyTorch Lightning module for JEPA byte-segment pretraining.

Key features:
- bf16-mixed precision ready
- Multi-GPU compatible
- EMA teacher update with cosine momentum schedule
- Per-corruption-type validation metrics
- Collapse monitoring (embedding std, cosine diversity)
- Curriculum stage tracking
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from text_classification.jepa.model import ByteSegmentJEPA
from text_classification.jepa.loss import LossWeights, compute_total_loss
from text_classification.jepa.curriculum import CurriculumScheduler, CURRICULUM_STAGES
from text_classification.utils.optimizers import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA momentum schedule
# ---------------------------------------------------------------------------

def _ema_momentum(
    step: int,
    max_steps: int,
    m_start: float = 0.996,
    m_end: float = 0.9999,
) -> float:
    """Cosine-annealed EMA momentum: ``m_start`` → ``m_end`` (spec §12)."""
    if max_steps <= 0:
        return m_end
    progress = min(step / max_steps, 1.0)
    return m_end - (m_end - m_start) * (1.0 + math.cos(math.pi * progress)) / 2.0


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class JEPAPretrainingModule(pl.LightningModule):
    """Lightning module wrapping :class:`ByteSegmentJEPA` for pretraining.

    Args:
        cfg:               Full OmegaConf config (see ``configs/jepa_base.yaml``).
        val_dataset_names: Names for each validation dataloader.  Defaults to
                           ``["val"]``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        val_dataset_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.val_dataset_names: list[str] = val_dataset_names or ["val"]
        torch.set_float32_matmul_precision("medium")

        # Build model
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        self.model = ByteSegmentJEPA(
            byte_dim=int(model_cfg.get("byte_dim", 256)),
            seg_dim=int(model_cfg.get("seg_dim", 512)),
            pred_dim=int(model_cfg.get("pred_dim", 512)),
            n_byte_blocks=int(model_cfg.get("n_byte_blocks", 4)),
            n_encoder_layers=int(model_cfg.get("n_encoder_layers", 12)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            ffn_dim=int(model_cfg.get("ffn_dim", 2048)),
            n_predictor_layers=int(model_cfg.get("n_predictor_layers", 4)),
            max_segments=int(data_cfg.get("max_segments", 2048)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            byte_dropout=float(model_cfg.get("byte_dropout", 0.05)),
            ema_momentum=float(model_cfg.get("ema_momentum", 0.996)),
        )

        if model_cfg.get("compile", False):
            self.model = torch.compile(self.model)

        # Loss weights
        loss_cfg = cfg.get("loss", {})
        self.loss_weights = LossWeights(
            segment=float(loss_cfg.get("segment", 1.0)),
            document=float(loss_cfg.get("document", 0.2)),
            variance=float(loss_cfg.get("variance", 0.05)),
            covariance=float(loss_cfg.get("covariance", 0.01)),
        )

        # EMA momentum parameters
        ema_cfg = cfg.get("ema", {})
        self.ema_m_start = float(ema_cfg.get("momentum_start", 0.996))
        self.ema_m_end = float(ema_cfg.get("momentum_end", 0.9999))

        # Optional curriculum tracker (informational only here; the DataModule
        # controls actual data selection).
        self.curriculum = CurriculumScheduler(stages=CURRICULUM_STAGES)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(
            clean_byte_values=batch["clean_byte_values"],
            clean_byte_types=batch["clean_byte_types"],
            canonical_positions=batch["canonical_positions"],
            canonical_mask=batch["canonical_mask"],
            student_bytes=batch["student_bytes"],
            student_byte_types=batch["student_byte_types"],
            student_positions=batch["student_positions"],
            student_mask=batch["student_mask"],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch)
        losses = compute_total_loss(
            predicted_segments=outputs["predicted_segments"],
            teacher_seg_targets=outputs["teacher_seg_targets"],
            predicted_doc=outputs["predicted_doc"],
            teacher_doc_targets=outputs["teacher_doc_targets"],
            segment_loss_weights=batch["segment_loss_weights"],
            canonical_mask=batch["canonical_mask"],
            weights=self.loss_weights,
        )

        # EMA teacher update (after the backward pass via on_before_optimizer_step
        # would be cleaner, but updating here is simpler and practically equivalent)
        max_steps = self.trainer.max_steps if self.trainer is not None else 10_000
        momentum = _ema_momentum(self.global_step, max_steps, self.ema_m_start, self.ema_m_end)
        self.model.update_teacher(momentum)

        self.log("train/loss", losses["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss_seg", losses["loss/segment"], on_step=True, on_epoch=False)
        self.log("train/loss_doc", losses["loss/document"], on_step=True, on_epoch=False)
        self.log("train/loss_var", losses["loss/variance"], on_step=True, on_epoch=False)
        self.log("train/loss_cov", losses["loss/covariance"], on_step=True, on_epoch=False)
        self.log("train/ema_momentum", momentum, on_step=True, on_epoch=False)

        # log the gradient norm separately for encoder and predictor to monitor training stability
        encoder_grad_norm = torch.nn.utils.get_total_norm(self.model.student.parameters(), norm_type=2.0)
        predictor_grad_norm = torch.nn.utils.get_total_norm(self.model.predictor.parameters(), norm_type=2.0)
        self.log("train/encoder_grad_norm", encoder_grad_norm, on_step=True, on_epoch=False)
        self.log("train/predictor_grad_norm", predictor_grad_norm, on_step=True, on_epoch=False)    

        return losses["loss"]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        outputs = self(batch)
        losses = compute_total_loss(
            predicted_segments=outputs["predicted_segments"],
            teacher_seg_targets=outputs["teacher_seg_targets"],
            predicted_doc=outputs["predicted_doc"],
            teacher_doc_targets=outputs["teacher_doc_targets"],
            segment_loss_weights=batch["segment_loss_weights"],
            canonical_mask=batch["canonical_mask"],
            weights=self.loss_weights,
        )

        ds = (
            self.val_dataset_names[dataloader_idx]
            if dataloader_idx < len(self.val_dataset_names)
            else f"val_{dataloader_idx}"
        )
        prefix = f"val/{ds}"

        self.log(
            f"{prefix}/loss", losses["loss"],
            on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False,
        )
        self.log(
            f"{prefix}/loss_seg", losses["loss/segment"],
            on_step=False, on_epoch=True, add_dataloader_idx=False,
        )
        self.log(
            f"{prefix}/loss_doc", losses["loss/document"],
            on_step=False, on_epoch=True, add_dataloader_idx=False,
        )
        self.log(
            f"{prefix}/loss_var", losses["loss/variance"],
            on_step=False, on_epoch=True, add_dataloader_idx=False,
        )

        # --- Cosine similarity metrics (spec §13.1) ---
        valid = batch["canonical_mask"]
        cos_seg = (outputs["predicted_segments"] * outputs["teacher_seg_targets"]).sum(dim=-1)
        if valid.any():
            mean_cos_seg = cos_seg[valid].mean()
        else:
            mean_cos_seg = cos_seg.new_tensor(0.0)
        self.log(
            f"{prefix}/cosine_seg", mean_cos_seg,
            on_step=False, on_epoch=True, add_dataloader_idx=False,
        )

        cos_doc = (outputs["predicted_doc"] * outputs["teacher_doc_targets"]).sum(dim=-1).mean()
        self.log(
            f"{prefix}/cosine_doc", cos_doc,
            on_step=False, on_epoch=True, add_dataloader_idx=False,
        )

        # --- Collapse monitoring: embedding std (spec §13.2) ---
        pred_flat = outputs["predicted_segments"][valid]  # (valid_segs, pred_dim)
        if pred_flat.shape[0] > 1:
            emb_std = pred_flat.std(dim=0).mean()
            self.log(
                f"{prefix}/embedding_std", emb_std,
                on_step=False, on_epoch=True, add_dataloader_idx=False,
            )

    def on_validation_epoch_end(self) -> None:
        """Advance curriculum if primary validation loss plateaus."""
        primary_key = f"val/{self.val_dataset_names[0]}/loss"
        val_loss = self.trainer.callback_metrics.get(primary_key)
        if val_loss is not None:
            advanced = self.curriculum.report_val_loss(float(val_loss))
            if advanced:
                logger.info(
                    "Curriculum advanced to stage %d (%s)",
                    self.curriculum.stage_idx,
                    self.curriculum.stage_name,
                )
        self.log(
            "curriculum/stage_idx",
            float(self.curriculum.stage_idx),
            on_step=False, on_epoch=True,
        )

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Teacher parameters have requires_grad=False and are skipped by
        # build_optimizer automatically.
        opt = build_optimizer(self.model.named_parameters(), self.cfg.optimizer)
        sched_dict = build_scheduler(opt, self.cfg.scheduler, max_steps=self.cfg.training.max_steps)
        return [opt], [sched_dict]
