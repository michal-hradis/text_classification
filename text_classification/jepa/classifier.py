"""JEPA-based multi-task multi-label text classifier.

Wraps a pretrained :class:`ByteSegmentEncoder` (loaded from a JEPA pretraining
checkpoint) and adds per-task linear classification heads on top of either the
``[DOC]`` token representation or a masked mean of segment representations.

Typical usage::

    model = JEPAClassifier(
        checkpoint_path="checkpoints/jepa/last.ckpt",
        tasks=["topic", "sentiment"],
        num_classes={"topic": 6, "sentiment": 3},
        use_teacher=True,
        pooling="doc_token",
    )
    logits = model(byte_values, byte_types, positions, seg_mask)
    # logits: {"topic": Tensor(B, 6), "sentiment": Tensor(B, 3)}
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from text_classification.jepa.corruption import SEGMENT_SIZE, CorruptionType
from text_classification.jepa.model import ByteSegmentEncoder
from text_classification.models.classifier import ClassificationHead

logger = logging.getLogger(__name__)


class JEPAClassifier(nn.Module):
    """Multi-task classifier built on top of a pretrained JEPA byte-segment encoder.

    The encoder is initialised from a JEPA pretraining checkpoint.  Either the
    EMA **teacher** encoder (recommended — higher quality) or the **student**
    encoder can be used.

    Args:
        checkpoint_path:      Path to a PyTorch Lightning ``.ckpt`` file produced
                              by :class:`JEPAPretrainingModule`.
        tasks:                Ordered list of task names.
        num_classes:          Mapping from task name to number of output classes.
        use_teacher:          If ``True`` (default), load the EMA teacher encoder;
                              otherwise load the gradient-trained student encoder.
        pooling:              ``"doc_token"`` — use the ``[DOC]`` token at position 0
                              (analogous to ``[CLS]`` pooling).
                              ``"mean_segments"`` — compute a masked mean over all
                              valid segment representations.
        freeze_encoder_layers: ``0`` — finetune entire encoder.
                              ``-1`` — freeze entire encoder (linear probe).
                              ``N > 0`` — freeze byte-embedding / local-processor /
                              reducer stages plus the first ``N`` Transformer layers.
        dropout:              Dropout applied in the classification heads.
        byte_dim:             Byte embedding / local-processor hidden dimension.
                              **Must match the pretrained checkpoint.**
        seg_dim:              Segment vector / Transformer model dimension.
                              **Must match the pretrained checkpoint.**
        n_byte_blocks:        Conv1D/SwiGLU blocks in the local processor.
                              **Must match the pretrained checkpoint.**
        n_encoder_layers:     Number of Transformer encoder layers.
                              **Must match the pretrained checkpoint.**
        n_heads:              Number of attention heads.
                              **Must match the pretrained checkpoint.**
        ffn_dim:              Transformer FFN hidden dimension.
                              **Must match the pretrained checkpoint.**
        max_segments:         Maximum number of canonical segments per document.
        kernel_size:          Conv1D kernel size.
                              **Must match the pretrained checkpoint.**
        dropout_encoder:      Dropout applied inside the encoder (Transformer +
                              FFN layers).  Defaults to ``0.1``.
        byte_dropout:         Dropout applied inside the local byte processor.
                              Defaults to ``0.05``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tasks: list[str],
        num_classes: dict[str, int],
        use_teacher: bool = True,
        pooling: str = "doc_token",
        freeze_encoder_layers: int = 0,
        dropout: float = 0.1,
        # Architecture hyperparameters (must match pretrained checkpoint)
        byte_dim: int = 256,
        seg_dim: int = 512,
        n_byte_blocks: int = 4,
        n_encoder_layers: int = 12,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        max_segments: Optional[int] = None,
        kernel_size: int = 5,
        dropout_encoder: float = 0.1,
        byte_dropout: float = 0.05,
        # ModernBERT-style architecture parameters
        n_additional_tokens: int = 0,
        local_window_size: int = 0,
        global_attention_every_n: int = 1,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        if pooling not in ("doc_token", "mean_segments"):
            raise ValueError(f"Unknown pooling strategy {pooling!r}. Use 'doc_token' or 'mean_segments'.")

        self.tasks = tasks
        self.pooling = pooling
        self.seg_dim = seg_dim
        self.n_additional_tokens = n_additional_tokens

        # Auto-detect max_segments from checkpoint.
        ckpt_max_segments = self._peek_max_segments(checkpoint_path, use_teacher)
        if max_segments is not None and max_segments != ckpt_max_segments:
            logger.warning(
                "Configured max_segments=%d differs from checkpoint max_segments=%d. "
                "Using checkpoint value to construct the encoder.",
                max_segments,
                ckpt_max_segments,
            )
        self.max_segments: int = ckpt_max_segments

        self.encoder = ByteSegmentEncoder(
            byte_dim=byte_dim,
            seg_dim=seg_dim,
            n_byte_blocks=n_byte_blocks,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            max_segments=self.max_segments,
            kernel_size=kernel_size,
            dropout=dropout_encoder,
            byte_dropout=byte_dropout,
            n_additional_tokens=n_additional_tokens,
            local_window_size=local_window_size,
            global_attention_every_n=global_attention_every_n,
            activation=activation,
        )
        self._load_encoder_weights(checkpoint_path, use_teacher)

        self.heads = nn.ModuleDict(
            {task: ClassificationHead(seg_dim, num_classes[task], dropout) for task in tasks}
        )

        if freeze_encoder_layers != 0:
            self._freeze_encoder(n_encoder_layers, freeze_encoder_layers)

    # ------------------------------------------------------------------
    # Checkpoint introspection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
        """Load and normalise the raw state dict from a checkpoint file."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_sd: dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)
        # Strip torch.compile wrapper prefix if present
        return {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in raw_sd.items()
        }

    @staticmethod
    def _peek_max_segments(checkpoint_path: str, use_teacher: bool) -> int:
        """Read ``max_segments`` from the checkpoint.

        Supports both the new architecture (``_max_segments`` buffer) and the
        legacy architecture (``pos_embed.weight`` table).
        """
        raw_sd = JEPAClassifier._load_raw_state_dict(checkpoint_path)
        branch = "teacher" if use_teacher else "student"
        # New architecture: scalar buffer registered in ByteSegmentEncoder.
        new_key = f"model.{branch}._max_segments"
        if new_key in raw_sd:
            max_segments = int(raw_sd[new_key].item())
            logger.info("Auto-detected max_segments=%d from checkpoint.", max_segments)
            return max_segments
        # Legacy architecture: pos_embed embedding table.
        old_key = f"model.{branch}.pos_embed.weight"
        if old_key in raw_sd:
            max_segments = raw_sd[old_key].shape[0]
            logger.info(
                "Auto-detected max_segments=%d from checkpoint (legacy pos_embed).",
                max_segments,
            )
            return max_segments
        raise KeyError(
            f"Cannot auto-detect max_segments: neither {new_key!r} nor {old_key!r} "
            f"found in checkpoint."
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_encoder_weights(self, checkpoint_path: str, use_teacher: bool) -> None:
        """Load encoder weights from a JEPA Lightning checkpoint.

        Lightning saves ``state_dict`` with module-path-prefixed keys, e.g.
        ``model.teacher.encoder.layers.0.norm1.weight``.  We strip the
        ``model.teacher.`` or ``model.student.`` prefix and load into
        :attr:`encoder`.
        """
        logger.info(
            "Loading JEPA %s encoder from %s",
            "teacher" if use_teacher else "student",
            checkpoint_path,
        )
        raw_sd = self._load_raw_state_dict(checkpoint_path)

        prefix = "model.teacher." if use_teacher else "model.student."
        encoder_sd = {
            k[len(prefix):]: v
            for k, v in raw_sd.items()
            if k.startswith(prefix)
        }
        if not encoder_sd:
            available = sorted({k.split(".")[0] for k in raw_sd})
            raise KeyError(
                f"No keys with prefix {prefix!r} found in checkpoint. "
                f"Top-level keys: {available}"
            )

        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=True)
        if missing:
            raise RuntimeError(f"Missing keys when loading encoder: {missing}")
        logger.info(
            "Loaded %d parameter tensors into %s encoder.",
            len(encoder_sd),
            "teacher" if use_teacher else "student",
        )

    # ------------------------------------------------------------------
    # Layer freezing
    # ------------------------------------------------------------------

    def _freeze_encoder(self, n_encoder_layers: int, freeze_encoder_layers: int) -> None:
        """Freeze encoder parameters.

        ``freeze_encoder_layers == -1``: freeze entire encoder.
        ``freeze_encoder_layers == N > 0``: freeze the byte-embedding, local
        processor, reducer, and positional-embedding stages plus the first
        ``N`` Transformer encoder layers.
        """
        if freeze_encoder_layers == -1:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            logger.info("Froze entire JEPA encoder (linear-probe mode).")
            return

        # Freeze pre-Transformer stages
        pre_transformer_modules: list[nn.Module] = [
            self.encoder.byte_embed,
            self.encoder.local_proc,
            self.encoder.reducer,
        ]
        for mod in pre_transformer_modules:
            for p in mod.parameters():
                p.requires_grad_(False)
        # Static token embeddings (nn.Parameters, not modules)
        self.encoder.doc_embed.requires_grad_(False)
        if getattr(self.encoder, "additional_token_embeds", None) is not None:
            self.encoder.additional_token_embeds.requires_grad_(False)

        # Freeze first N Transformer layers
        n_to_freeze = min(freeze_encoder_layers, n_encoder_layers)
        for i in range(n_to_freeze):
            for p in self.encoder.encoder.layers[i].parameters():
                p.requires_grad_(False)

        logger.info(
            "Froze pre-Transformer encoder stages + first %d Transformer layer(s).",
            n_to_freeze,
        )

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
        """Run the encoder and all classification heads.

        Args:
            byte_values:  ``(B, N, SEGMENT_SIZE)`` long — raw byte values (0-255
                          for real bytes, ``PAD_BYTE=256`` for padding segments).
            byte_types:   ``(B, N, SEGMENT_SIZE)`` long — per-byte
                          :class:`CorruptionType` labels (all ``CLEAN=0`` for
                          unmodified inference input; ``PADDING=6`` for padding).
            positions:    ``(B, N)`` long — canonical segment indices (0, 1, …).
            seg_mask:     ``(B, N)`` bool — ``True`` for valid segments,
                          ``False`` for batch-padding segments.  ``None``
                          treats all segments as valid.

        Returns:
            Dict mapping task name → raw logits of shape
            ``(B, num_classes[task])``.
        """
        # encoder_out: (B, n_static+N, seg_dim), layer_outputs: list
        encoder_out, _ = self.encoder(byte_values, byte_types, positions, seg_mask)
        n_static = self.encoder.n_static  # 1 + n_additional_tokens

        if self.pooling == "doc_token":
            pooled = encoder_out[:, 0, :]                          # (B, seg_dim)
        else:  # mean_segments
            segs = encoder_out[:, n_static:, :]                    # (B, N, seg_dim)
            if seg_mask is not None:
                mask = seg_mask.float().unsqueeze(-1)              # (B, N, 1)
                pooled = (segs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            else:
                pooled = segs.mean(1)                              # (B, seg_dim)

        return {task: head(pooled) for task, head in self.heads.items()}
