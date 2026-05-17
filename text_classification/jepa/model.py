"""Byte-Segment JEPA model components.

Architecture (spec §6):

    UTF-8 bytes
    → ByteInputEmbedding (value + offset + corruption-type)
    → LocalByteProcessor (Conv1D/SwiGLU residual blocks)
    → ByteToSegmentReducer (mean + attn + first + last → linear)
    → [DOC] prepend + positional encoding
    → TransformerEncoderWithIntermediates (student or teacher)

Student encoder (gradient) + EMA teacher encoder + cross-attention predictor
form the full :class:`ByteSegmentJEPA` model.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_classification.jepa.corruption import BYTE_VOCAB_SIZE, SEGMENT_SIZE, CorruptionType


# ---------------------------------------------------------------------------
# SwiGLU feed-forward block
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU: ``SiLU(x @ W1) * (x @ W2)`` → linear projection."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # w1 produces gate + value in one shot
        self.w1 = nn.Linear(in_features, hidden_features * 2, bias=True)
        self.w2 = nn.Linear(hidden_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(F.silu(gate) * val))


# ---------------------------------------------------------------------------
# Local byte processor
# ---------------------------------------------------------------------------

class ByteConvBlock(nn.Module):
    """Residual Conv1D + SwiGLU block (pre-LayerNorm, operates within a segment).

    Input shape: ``(batch, seq_len, byte_dim)``.
    """

    def __init__(
        self,
        byte_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(byte_dim)
        self.conv = nn.Conv1d(
            byte_dim, byte_dim, kernel_size, padding=kernel_size // 2, bias=True
        )
        self.norm2 = nn.LayerNorm(byte_dim)
        self.mlp = SwiGLU(byte_dim, byte_dim * 2, byte_dim, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1D branch
        residual = x
        h = self.norm1(x).transpose(1, 2)   # (B, byte_dim, T)
        h = self.conv(h).transpose(1, 2)     # (B, T, byte_dim)
        x = residual + self.drop(h)
        # MLP branch
        x = x + self.mlp(self.norm2(x))
        return x


class LocalByteProcessor(nn.Module):
    """Stack of :class:`ByteConvBlock` applied independently within each segment.

    Input:  ``(B, N_segs, SEGMENT_SIZE, byte_dim)``
    Output: same shape.
    """

    def __init__(
        self,
        byte_dim: int,
        n_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ByteConvBlock(byte_dim, kernel_size, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, S, D = x.shape
        x = x.view(B * N, S, D)
        for blk in self.blocks:
            x = blk(x)
        return x.view(B, N, S, D)


# ---------------------------------------------------------------------------
# Byte input embedding
# ---------------------------------------------------------------------------

class ByteInputEmbedding(nn.Module):
    """Sum of byte-value + intra-segment offset + corruption-type embeddings."""

    def __init__(self, byte_dim: int) -> None:
        super().__init__()
        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, byte_dim)
        self.offset_embed = nn.Embedding(SEGMENT_SIZE, byte_dim)
        self.corrupt_embed = nn.Embedding(len(CorruptionType), byte_dim)
        self.norm = nn.LayerNorm(byte_dim)

    def forward(
        self,
        byte_values: torch.Tensor,          # (B, N, 32)  long
        byte_corruption_types: torch.Tensor, # (B, N, 32)  long
    ) -> torch.Tensor:
        B, N, S = byte_values.shape
        offsets = (
            torch.arange(S, device=byte_values.device)
            .unsqueeze(0).unsqueeze(0)
            .expand(B, N, -1)
        )
        x = (
            self.byte_embed(byte_values)
            + self.offset_embed(offsets)
            + self.corrupt_embed(byte_corruption_types)
        )
        return self.norm(x)  # (B, N, 32, byte_dim)


# ---------------------------------------------------------------------------
# Byte-to-segment reducer
# ---------------------------------------------------------------------------

class ByteToSegmentReducer(nn.Module):
    """Reduce ``SEGMENT_SIZE`` byte states to one segment vector.

    Combines mean pooling, attention pooling, first-byte, and last-byte
    states via a linear projection.

    Input:  ``(B, N, SEGMENT_SIZE, byte_dim)``
    Output: ``(B, N, seg_dim)``
    """

    def __init__(self, byte_dim: int, seg_dim: int) -> None:
        super().__init__()
        self.attn_q = nn.Linear(byte_dim, 1, bias=True)
        self.proj = nn.Linear(byte_dim * 4, seg_dim, bias=True)
        self.norm = nn.LayerNorm(seg_dim)

    def forward(self, byte_states: torch.Tensor) -> torch.Tensor:
        # byte_states: (B, N, S, D)
        # Mean pooling
        mean_pool = byte_states.mean(dim=2)                              # (B, N, D)
        # Attention pooling
        attn_w = self.attn_q(byte_states).squeeze(-1)                   # (B, N, S)
        attn_w = F.softmax(attn_w, dim=-1)
        attn_pool = (attn_w.unsqueeze(-1) * byte_states).sum(dim=2)     # (B, N, D)
        # First / last byte
        first = byte_states[:, :, 0, :]                                  # (B, N, D)
        last = byte_states[:, :, -1, :]                                  # (B, N, D)

        combined = torch.cat([mean_pool, attn_pool, first, last], dim=-1)  # (B, N, 4D)
        return self.norm(self.proj(combined))                               # (B, N, seg_dim)


# ---------------------------------------------------------------------------
# Transformer encoder (returns all layer outputs)
# ---------------------------------------------------------------------------

class _TransformerLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer with SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (pre-LN)
        residual = x
        h = self.norm1(x)
        h, _ = self.self_attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = residual + self.drop(h)
        # FFN (pre-LN)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoderWithIntermediates(nn.Module):
    """Transformer encoder that exposes all intermediate layer outputs.

    Returns:
        ``(final_output, layer_outputs)`` where ``layer_outputs[i]`` is
        the output of layer ``i`` (before the final LayerNorm).
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        layer_outputs: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
            layer_outputs.append(x)
        return self.final_norm(x), layer_outputs


# ---------------------------------------------------------------------------
# Cross-attention predictor
# ---------------------------------------------------------------------------

class _PredictorLayer(nn.Module):
    """Pre-LN cross-attention decoder layer (self-attn + cross-attn + FFN)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        context: torch.Tensor,
        q_key_padding_mask: Optional[torch.Tensor] = None,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention on queries
        residual = q
        h = self.norm1(q)
        h, _ = self.self_attn(h, h, h, key_padding_mask=q_key_padding_mask, need_weights=False)
        q = residual + self.drop(h)
        # Cross-attention
        residual = q
        h = self.norm2(q)
        h, _ = self.cross_attn(h, context, context, key_padding_mask=ctx_key_padding_mask, need_weights=False)
        q = residual + self.drop(h)
        # FFN
        q = q + self.ffn(self.norm3(q))
        return q


class SegmentPredictor(nn.Module):
    """Cross-attention predictor: student context → canonical segment predictions.

    For every canonical position the predictor produces an L2-normalized
    embedding that is trained to match the EMA teacher's target.

    Args:
        d_model:       Model dimension.
        pred_dim:      Output (prediction) dimension.
        nhead:         Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        num_layers:    Number of predictor layers.
        max_segments:  Maximum canonical sequence length.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        pred_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        max_segments: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Embedding(max_segments, d_model)
        self.layers = nn.ModuleList(
            [_PredictorLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_dim, bias=True)

    def forward(
        self,
        canonical_positions: torch.Tensor,          # (B, N)  long
        student_context: torch.Tensor,               # (B, 1+M, d_model)
        canonical_mask: Optional[torch.Tensor] = None,          # (B, N) True=valid
        student_key_padding_mask: Optional[torch.Tensor] = None, # (B, 1+M) True=pad
    ) -> torch.Tensor:
        B, N = canonical_positions.shape
        # Build queries: shared learned embedding + per-position encoding
        q = self.target_query.expand(B, N, -1) + self.pos_embed(canonical_positions)  # (B, N, d_model)

        q_kpm = (~canonical_mask) if canonical_mask is not None else None  # True=pad

        for layer in self.layers:
            q = layer(q, student_context, q_key_padding_mask=q_kpm, ctx_key_padding_mask=student_key_padding_mask)

        q = self.norm(q)
        out = self.head(q)              # (B, N, pred_dim)
        return F.normalize(out, dim=-1) # unit sphere


# ---------------------------------------------------------------------------
# Document representation head
# ---------------------------------------------------------------------------

class _DocRepHead(nn.Module):
    """Project a state vector to a normalized document embedding."""

    def __init__(self, d_model: int, pred_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, pred_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.norm(x)), dim=-1)


# ---------------------------------------------------------------------------
# Shared encoder (student and teacher)
# ---------------------------------------------------------------------------

class ByteSegmentEncoder(nn.Module):
    """Full byte → segment → Transformer encoder pipeline.

    Bytes are first embedded (value + offset + corruption-type), processed by
    local convolutional blocks, reduced to one vector per segment, then
    passed through a Transformer encoder.  A learned ``[DOC]`` token is
    prepended to capture document-level context.

    Args:
        byte_dim:         Byte embedding / local-processor hidden dimension.
        seg_dim:          Segment vector / Transformer model dimension.
        n_byte_blocks:    Number of Conv1D/SwiGLU residual blocks.
        n_encoder_layers: Number of Transformer encoder layers.
        n_heads:          Attention heads.
        ffn_dim:          Transformer FFN hidden dimension.
        max_segments:     Maximum canonical sequence length.
        kernel_size:      Conv1D kernel size in the local byte processor.
        dropout:          Attention + FFN dropout.
        byte_dropout:     Dropout in the local byte processor.
    """

    def __init__(
        self,
        byte_dim: int,
        seg_dim: int,
        n_byte_blocks: int,
        n_encoder_layers: int,
        n_heads: int,
        ffn_dim: int,
        max_segments: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
        byte_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.byte_embed = ByteInputEmbedding(byte_dim)
        self.local_proc = LocalByteProcessor(byte_dim, n_byte_blocks, kernel_size, byte_dropout)
        self.reducer = ByteToSegmentReducer(byte_dim, seg_dim)
        self.doc_embed = nn.Parameter(torch.randn(1, 1, seg_dim) * 0.02)
        self.pos_embed = nn.Embedding(max_segments, seg_dim)
        self.encoder = TransformerEncoderWithIntermediates(
            n_encoder_layers, seg_dim, n_heads, ffn_dim, dropout
        )

    def forward(
        self,
        byte_values: torch.Tensor,       # (B, N, 32) long
        byte_types: torch.Tensor,         # (B, N, 32) long
        positions: torch.Tensor,          # (B, N) long — 0-indexed canonical ids
        seg_mask: Optional[torch.Tensor] = None,  # (B, N) bool True=valid
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Returns:
            Tuple of ``(output, layer_outputs)`` where ``output`` has shape
            ``(B, 1+N, seg_dim)`` (``[DOC]`` + segments, final-layer-normed)
            and ``layer_outputs[i]`` is the pre-norm output of layer ``i``.
        """
        B, N = byte_values.shape[:2]

        # 1. Byte embedding + local processing + segment reduction
        x = self.byte_embed(byte_values, byte_types)   # (B, N, 32, byte_dim)
        x = self.local_proc(x)                          # (B, N, 32, byte_dim)
        seg_vecs = self.reducer(x)                      # (B, N, seg_dim)

        # 2. Canonical positional encoding
        seg_vecs = seg_vecs + self.pos_embed(positions)  # (B, N, seg_dim)

        # 3. Prepend [DOC] token
        doc = self.doc_embed.expand(B, 1, -1)            # (B, 1, seg_dim)
        seq = torch.cat([doc, seg_vecs], dim=1)           # (B, 1+N, seg_dim)

        # 4. Build key-padding mask (True = PAD, PyTorch convention)
        if seg_mask is not None:
            doc_valid = seg_mask.new_ones(B, 1)           # [DOC] is always valid
            full_valid = torch.cat([doc_valid, seg_mask], dim=1)  # (B, 1+N)
            kpm = ~full_valid
        else:
            kpm = None

        # 5. Transformer
        out, layers = self.encoder(seq, key_padding_mask=kpm)
        return out, layers


# ---------------------------------------------------------------------------
# Full JEPA model
# ---------------------------------------------------------------------------

class ByteSegmentJEPA(nn.Module):
    """Complete JEPA pretraining model.

    Comprises:

    - **Student encoder** — trained by gradient descent.
    - **EMA teacher encoder** — exponential-moving-average copy of the student;
      receives the clean unmodified byte segments.
    - **Segment predictor** — cross-attention Transformer that predicts
      teacher targets for every canonical position given the student context.
    - **Teacher target projection** — projects an average of the upper teacher
      layers to the prediction space.
    - **Document representation heads** — project the ``[DOC]`` token to a
      normalized embedding for the document-level consistency loss.

    Args:
        byte_dim:               Byte embedding dimension (default 256).
        seg_dim:                Segment model dimension (default 512).
        pred_dim:               Prediction / target dimension (default 512).
        n_byte_blocks:          Conv1D/SwiGLU blocks in local processor (default 4).
        n_encoder_layers:       Student/teacher Transformer layers (default 12).
        n_heads:                Attention heads (default 8).
        ffn_dim:                Transformer FFN hidden dim (default 2048).
        n_predictor_layers:     Predictor layers (default 4).
        max_segments:           Maximum canonical segments per document (default 2048).
        kernel_size:            Conv1D kernel in local byte processor (default 5).
        dropout:                Dropout in encoder and predictor (default 0.1).
        byte_dropout:           Dropout in local byte processor (default 0.05).
        teacher_target_layers:  Indices of teacher layers to average for targets.
                                Defaults to the upper quarter of layers.
        ema_momentum:           Initial EMA momentum (default 0.996).
    """

    def __init__(
        self,
        byte_dim: int = 256,
        seg_dim: int = 512,
        pred_dim: int = 512,
        n_byte_blocks: int = 4,
        n_encoder_layers: int = 12,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        n_predictor_layers: int = 4,
        max_segments: int = 2048,
        kernel_size: int = 5,
        dropout: float = 0.1,
        byte_dropout: float = 0.05,
        teacher_target_layers: Optional[list[int]] = None,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.seg_dim = seg_dim
        self.pred_dim = pred_dim
        self.n_encoder_layers = n_encoder_layers
        self.ema_momentum = ema_momentum

        # Default upper-quarter layers for teacher targets
        if teacher_target_layers is None:
            start = max(n_encoder_layers - max(n_encoder_layers // 4, 1), 0)
            self.teacher_target_layers: list[int] = list(range(start, n_encoder_layers))
        else:
            self.teacher_target_layers = teacher_target_layers

        encoder_kwargs = dict(
            byte_dim=byte_dim,
            seg_dim=seg_dim,
            n_byte_blocks=n_byte_blocks,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            max_segments=max_segments,
            kernel_size=kernel_size,
            dropout=dropout,
            byte_dropout=byte_dropout,
        )

        # Student encoder (receives gradients)
        self.student = ByteSegmentEncoder(**encoder_kwargs)

        # Teacher encoder (EMA, frozen)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Predictor
        self.predictor = SegmentPredictor(
            d_model=seg_dim,
            pred_dim=pred_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            num_layers=n_predictor_layers,
            max_segments=max_segments,
            dropout=dropout,
        )

        # Teacher target projection (averages upper layers → pred_dim)
        self.teacher_target_proj = nn.Sequential(
            nn.LayerNorm(seg_dim),
            nn.Linear(seg_dim, pred_dim, bias=True),
        )

        # Document representation heads
        self.student_doc_head = _DocRepHead(seg_dim, pred_dim)
        # Teacher doc head starts as EMA copy → freeze separately after deepcopy
        self.teacher_doc_head = copy.deepcopy(self.student_doc_head)
        for p in self.teacher_doc_head.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Update the EMA teacher from the student with the given ``momentum``.

        Call this once per training step **after** the optimizer step.
        """
        for sp, tp in zip(self.student.parameters(), self.teacher.parameters()):
            tp.data.mul_(momentum).add_(sp.data * (1.0 - momentum))
        for sp, tp in zip(self.student_doc_head.parameters(), self.teacher_doc_head.parameters()):
            tp.data.mul_(momentum).add_(sp.data * (1.0 - momentum))

    # ------------------------------------------------------------------
    # Teacher targets
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_teacher_targets(
        self,
        clean_byte_values: torch.Tensor,    # (B, N, 32)
        clean_byte_types: torch.Tensor,     # (B, N, 32)
        canonical_positions: torch.Tensor,  # (B, N)
        canonical_mask: torch.Tensor,       # (B, N) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute L2-normalized teacher targets (no gradient).

        Returns:
            ``(seg_targets, doc_targets)`` with shapes
            ``(B, N, pred_dim)`` and ``(B, pred_dim)``.
        """
        _, layer_outputs = self.teacher(
            clean_byte_values, clean_byte_types, canonical_positions, canonical_mask
        )
        # Average upper layers; skip [DOC] token (position 0)
        upper = [layer_outputs[i][:, 1:, :] for i in self.teacher_target_layers]
        avg_upper = torch.stack(upper, dim=0).mean(dim=0)            # (B, N, seg_dim)
        seg_targets = F.normalize(self.teacher_target_proj(avg_upper), dim=-1)

        # Document target from [DOC] state of the last teacher layer
        doc_state = layer_outputs[-1][:, 0, :]                        # (B, seg_dim)
        doc_targets = self.teacher_doc_head(doc_state)                # (B, pred_dim)

        return seg_targets, doc_targets

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        clean_byte_values: torch.Tensor,    # (B, N, 32)
        clean_byte_types: torch.Tensor,     # (B, N, 32)
        canonical_positions: torch.Tensor,  # (B, N)
        canonical_mask: torch.Tensor,       # (B, N) bool True=valid
        student_bytes: torch.Tensor,        # (B, M, 32)
        student_byte_types: torch.Tensor,   # (B, M, 32)
        student_positions: torch.Tensor,    # (B, M)  canonical ids of visible segs
        student_mask: torch.Tensor,         # (B, M) bool True=valid
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Returns a dict with keys:

        - ``predicted_segments``  ``(B, N, pred_dim)`` — predictor output
        - ``teacher_seg_targets`` ``(B, N, pred_dim)`` — EMA teacher targets
        - ``predicted_doc``       ``(B, pred_dim)``    — student doc embedding
        - ``teacher_doc_targets`` ``(B, pred_dim)``    — teacher doc embedding
        """
        # 1. Teacher targets (stop-gradient)
        seg_targets, doc_targets = self._compute_teacher_targets(
            clean_byte_values, clean_byte_types, canonical_positions, canonical_mask
        )

        # 2. Student encoder
        student_out, _ = self.student(
            student_bytes, student_byte_types, student_positions, student_mask
        )
        # student_out: (B, 1+M, seg_dim) — [DOC] at position 0

        # 3. Student document embedding
        student_doc = self.student_doc_head(student_out[:, 0, :])  # (B, pred_dim)

        # 4. Build student key-padding mask for predictor cross-attention
        B = student_mask.shape[0]
        doc_valid = student_mask.new_ones(B, 1)
        full_student_valid = torch.cat([doc_valid, student_mask], dim=1)  # (B, 1+M)
        student_kpm = ~full_student_valid  # True = PAD

        # 5. Predictor
        predicted_segments = self.predictor(
            canonical_positions,
            student_out,
            canonical_mask=canonical_mask,
            student_key_padding_mask=student_kpm,
        )  # (B, N, pred_dim)

        return {
            "predicted_segments": predicted_segments,
            "teacher_seg_targets": seg_targets,
            "predicted_doc": student_doc,
            "teacher_doc_targets": doc_targets,
        }
