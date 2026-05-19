"""Byte-Segment JEPA model components — ModernBERT-style architecture.

Architecture (spec §6 + modern arch extensions):

    UTF-8 bytes
    → ByteInputEmbedding (value + offset + corruption-type)
    → LocalByteProcessor (Conv1D/SwiGLU residual blocks)
    → ByteToSegmentReducer (mean + attn + first + last → linear)
    → [DOC] + additional static tokens prepend
    → TransformerEncoderWithIntermediates
        - Pre-normalization
        - RoPE in self-attention (static tokens exempt)
        - GLU/GeGLU feed-forward
        - No linear biases in attention projections
        - Configurable local/global attention (sliding window)
        - Fast attention via torch.scaled_dot_product_attention

Student encoder (gradient) + EMA / frozen teacher encoder + cross-attention
predictor (with RoPE and static-token queries) form the full
:class:`ByteSegmentJEPA` model.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from text_classification.jepa.corruption import BYTE_VOCAB_SIZE, SEGMENT_SIZE, CorruptionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GLU feed-forward (SwiGLU or GeGLU)
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    """Gated Linear Unit feed-forward block.

    Supports two activations:
    - ``\'swiglu\'``: ``SiLU(gate) * val`` (default, aka SwiGLU)
    - ``\'geglu\'``:  ``GELU(gate) * val``

    Biases are removed from the gate/value projection (``w1``) following
    ModernBERT conventions; the output projection (``w2``) retains a bias.
    The output projection is tagged ``_is_residual_output=True`` for
    Megatron-style scaled initialization.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: str = "swiglu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in ("swiglu", "geglu"):
            raise ValueError(f"Unknown GLU activation {activation!r}. Use \'swiglu\' or \'geglu\'.")
        self.activation = activation
        self._hidden = hidden_features
        # Store the activation function at construction time so forward() has no
        # Python branch and torch.compile can trace a single graph unambiguously.
        self._act_fn = F.silu if activation == "swiglu" else F.gelu
        self.w1 = nn.Linear(in_features, hidden_features * 2, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=True)
        self.w2._is_residual_output = True  # type: ignore[attr-defined]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice with a concrete integer bound rather than chunk(2) so that
        # torch.compile / Inductor never sees a dynamic split point.
        x_proj = self.w1(x)
        gate, val = x_proj[..., :self._hidden], x_proj[..., self._hidden:]
        return self.w2(self.dropout(self._act_fn(gate) * val))


# Backward-compatible alias -- existing imports of SwiGLU continue to work.
SwiGLU = GLU


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a query/key tensor.

    Args:
        x:   ``(B, heads, S, head_dim)``
        cos: ``(B, S, head_dim)``
        sin: same shape as cos
    """
    cos = cos.unsqueeze(1)  # (B, 1, S, head_dim)
    sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables (non-persistent buffers)."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)       # (max_seq_len, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors of shape ``(B, S, head_dim)``."""
        cos = self.cos_cached[positions].to(dtype=self.cos_cached.dtype)
        sin = self.sin_cached[positions].to(dtype=self.sin_cached.dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Local attention mask helper
# ---------------------------------------------------------------------------

def _make_local_attn_mask(
    seq_len: int, n_static: int, window: int, device: torch.device
) -> torch.Tensor:
    """Boolean (seq_len, seq_len) mask: True = query allowed to attend key.

    Static tokens (rows/cols 0..n_static-1) always attend / are attended globally.
    Segment tokens use a symmetric sliding window of +/-window positions.
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    within_window = ((rows - cols).abs() <= window)
    static_row = rows < n_static
    static_col = cols < n_static
    return within_window | static_row | static_col


# ---------------------------------------------------------------------------
# Custom multi-head self-attention with RoPE and optional local window
# ---------------------------------------------------------------------------

class _SelfAttention(nn.Module):
    """Multi-head self-attention.

    - No projection biases (ModernBERT style)
    - Applies RoPE to the segment portion of the sequence
    - Supports sliding-window local attention mask
    - Uses F.scaled_dot_product_attention for fast execution
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        rotary: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj._is_residual_output = True  # type: ignore[attr-defined]

    def forward(
        self,
        x: torch.Tensor,                            # (B, S, d_model)
        positions: Optional[torch.Tensor] = None,   # (B, S-n_static) segment positions
        n_static: int = 0,
        attn_mask: Optional[torch.Tensor] = None,   # (S, S) bool True=allowed
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, S) True=pad
    ) -> torch.Tensor:
        B, S, _ = x.shape
        hd = self.head_dim
        q = self.q_proj(x).view(B, S, self.nhead, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, hd).transpose(1, 2)

        if self.rotary is not None and positions is not None and n_static < S:
            cos, sin = self.rotary.get_cos_sin(positions)
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
            q = torch.cat([q[:, :, :n_static], _apply_rope(q[:, :, n_static:], cos, sin)], dim=2)
            k = torch.cat([k[:, :, :n_static], _apply_rope(k[:, :, n_static:], cos, sin)], dim=2)

        bias: Optional[torch.Tensor] = None
        if attn_mask is not None:
            bias = x.new_zeros(1, 1, S, S).masked_fill(~attn_mask, float("-inf"))
        if key_padding_mask is not None:
            kpm_bias = x.new_zeros(B, 1, 1, S).masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            bias = kpm_bias if bias is None else bias + kpm_bias

        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=dp)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, self.d_model))


# ---------------------------------------------------------------------------
# Custom multi-head cross-attention with RoPE
# ---------------------------------------------------------------------------

class _CrossAttention(nn.Module):
    """Multi-head cross-attention with optional RoPE on Q and K."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        rotary: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.rotary = rotary
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj._is_residual_output = True  # type: ignore[attr-defined]

    def forward(
        self,
        q_input: torch.Tensor,                        # (B, S_q, d_model)
        kv_input: torch.Tensor,                       # (B, S_kv, d_model)
        q_positions: Optional[torch.Tensor] = None,   # (B, S_q - n_skip_q)
        kv_positions: Optional[torch.Tensor] = None,  # (B, S_kv - n_skip_k)
        n_skip_q: int = 0,
        n_skip_k: int = 0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S_q, _ = q_input.shape
        S_kv = kv_input.shape[1]
        hd = self.head_dim
        q = self.q_proj(q_input).view(B, S_q, self.nhead, hd).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, S_kv, self.nhead, hd).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, S_kv, self.nhead, hd).transpose(1, 2)

        if self.rotary is not None:
            if q_positions is not None and n_skip_q < S_q:
                cos, sin = self.rotary.get_cos_sin(q_positions)
                cos, sin = cos.to(q.dtype), sin.to(q.dtype)
                q = torch.cat([q[:, :, :n_skip_q], _apply_rope(q[:, :, n_skip_q:], cos, sin)], dim=2)
            if kv_positions is not None and n_skip_k < S_kv:
                cos, sin = self.rotary.get_cos_sin(kv_positions)
                cos, sin = cos.to(k.dtype), sin.to(k.dtype)
                k = torch.cat([k[:, :, :n_skip_k], _apply_rope(k[:, :, n_skip_k:], cos, sin)], dim=2)

        bias: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            bias = q_input.new_zeros(B, 1, 1, S_kv).masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=dp)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S_q, self.d_model))


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------

class _TransformerLayer(nn.Module):
    """Pre-LN Transformer encoder layer: RoPE self-attention + GLU FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        is_local: bool = False,
        local_window_size: int = 128,
        activation: str = "swiglu",
        rotary: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.self_attn = _SelfAttention(d_model, nhead, dropout, rotary)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GLU(d_model, dim_feedforward, d_model, activation, dropout)
        self.drop = nn.Dropout(dropout)
        self._is_local = is_local
        self._local_window_size = local_window_size

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        n_static: int = 0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mask: Optional[torch.Tensor] = None
        if self._is_local and self._local_window_size > 0:
            attn_mask = _make_local_attn_mask(x.shape[1], n_static, self._local_window_size, x.device)
        residual = x
        x = residual + self.drop(self.self_attn(self.norm1(x), positions, n_static, attn_mask, key_padding_mask))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Local byte processor
# ---------------------------------------------------------------------------

class ByteConvBlock(nn.Module):
    """Residual Conv1D + SwiGLU block (pre-LayerNorm, within a segment).

    Input shape: ``(batch, seq_len, byte_dim)``.
    """

    def __init__(self, byte_dim: int, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(byte_dim)
        self.conv = nn.Conv1d(byte_dim, byte_dim, kernel_size, padding=kernel_size // 2, bias=True)
        self.norm2 = nn.LayerNorm(byte_dim)
        self.mlp = GLU(byte_dim, byte_dim * 2, byte_dim, activation="swiglu", dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = residual + self.drop(h)
        return x + self.mlp(self.norm2(x))


class LocalByteProcessor(nn.Module):
    """Stack of :class:`ByteConvBlock` applied independently within each segment.

    Input:  ``(B, N_segs, SEGMENT_SIZE, byte_dim)``
    Output: same shape.
    """

    def __init__(self, byte_dim: int, n_blocks: int = 4, kernel_size: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ByteConvBlock(byte_dim, kernel_size, dropout) for _ in range(n_blocks)])

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

    def forward(self, byte_values: torch.Tensor, byte_corruption_types: torch.Tensor) -> torch.Tensor:
        B, N, S = byte_values.shape
        offsets = torch.arange(S, device=byte_values.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        return self.norm(
            self.byte_embed(byte_values) + self.offset_embed(offsets) + self.corrupt_embed(byte_corruption_types)
        )


# ---------------------------------------------------------------------------
# Byte-to-segment reducer
# ---------------------------------------------------------------------------

class ByteToSegmentReducer(nn.Module):
    """Reduce SEGMENT_SIZE byte states to one segment vector via pooling.

    Input:  ``(B, N, SEGMENT_SIZE, byte_dim)``
    Output: ``(B, N, seg_dim)``
    """

    def __init__(self, byte_dim: int, seg_dim: int) -> None:
        super().__init__()
        self.attn_q = nn.Linear(byte_dim, 1, bias=True)
        self.proj = nn.Linear(byte_dim * 4, seg_dim, bias=True)
        self.norm = nn.LayerNorm(seg_dim)

    def forward(self, byte_states: torch.Tensor) -> torch.Tensor:
        mean_pool = byte_states.mean(dim=2)
        attn_w = F.softmax(self.attn_q(byte_states).squeeze(-1), dim=-1)
        attn_pool = (attn_w.unsqueeze(-1) * byte_states).sum(dim=2)
        first, last = byte_states[:, :, 0, :], byte_states[:, :, -1, :]
        return self.norm(self.proj(torch.cat([mean_pool, attn_pool, first, last], dim=-1)))


# ---------------------------------------------------------------------------
# Transformer encoder with RoPE and configurable local/global attention
# ---------------------------------------------------------------------------

class TransformerEncoderWithIntermediates(nn.Module):
    """Transformer encoder that exposes all intermediate layer outputs.

    Key features vs. the original:
    - RoPE (static tokens exempt via ``n_static``)
    - Configurable local/global attention pattern
    - SwiGLU or GeGLU FFN
    - No attention projection biases
    - SDPA (fast attention)

    Args:
        num_layers:               Number of encoder layers.
        d_model:                  Model dimension.
        nhead:                    Attention heads.
        dim_feedforward:          GLU hidden dimension.
        dropout:                  Dropout probability.
        max_seq_len:              Max sequence length for RoPE table.
        local_window_size:        Half-width of sliding-window local attention.
                                  ``0`` -> all global.
        global_attention_every_n: Every n-th layer is global; others are local
                                  (ignored when ``local_window_size == 0``).
                                  ``1`` -> all global.
        activation:               ``\'swiglu\'`` or ``\'geglu\'``.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        local_window_size: int = 0,
        global_attention_every_n: int = 1,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        head_dim = d_model // nhead
        self.rotary = RotaryEmbedding(head_dim, max_seq_len)
        gate_n = max(global_attention_every_n, 1)
        self.layers = nn.ModuleList([
            _TransformerLayer(
                d_model, nhead, dim_feedforward, dropout,
                is_local=(local_window_size > 0 and (i + 1) % gate_n != 0),
                local_window_size=local_window_size,
                activation=activation,
                rotary=self.rotary,
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,   # (B, N_seg) canonical positions
        n_static: int = 0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns ``(final_output, layer_outputs)``."""
        B, S, _ = x.shape
        N_seg = S - n_static
        if positions is None and N_seg > 0:
            positions = torch.arange(N_seg, device=x.device).unsqueeze(0).expand(B, -1)
        layer_outputs: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, positions, n_static, key_padding_mask)
            layer_outputs.append(x)
        return self.final_norm(x), layer_outputs


# ---------------------------------------------------------------------------
# Predictor layer (cross-attention decoder with RoPE)
# ---------------------------------------------------------------------------

class _PredictorLayer(nn.Module):
    """Pre-LN cross-attention decoder: self-attn + cross-attn + GLU FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "swiglu",
        rotary: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.self_attn = _SelfAttention(d_model, nhead, dropout, rotary)
        self.cross_attn = _CrossAttention(d_model, nhead, dropout, rotary)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = GLU(d_model, dim_feedforward, d_model, activation, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        context: torch.Tensor,
        q_positions: Optional[torch.Tensor] = None,
        ctx_positions: Optional[torch.Tensor] = None,
        n_static: int = 0,
        q_key_padding_mask: Optional[torch.Tensor] = None,
        ctx_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = q + self.drop(self.self_attn(self.norm1(q), q_positions, n_static, None, q_key_padding_mask))
        q = q + self.drop(self.cross_attn(self.norm2(q), context, q_positions, ctx_positions, n_static, n_static, ctx_key_padding_mask))
        return q + self.ffn(self.norm3(q))


# ---------------------------------------------------------------------------
# Segment predictor with static-token queries
# ---------------------------------------------------------------------------

class SegmentPredictor(nn.Module):
    """Cross-attention predictor: student context -> canonical segment predictions.

    When ``n_static_tokens == 0`` (default, backward-compatible), only segment
    positions are predicted and output shape is ``(B, N, pred_dim)``.
    When ``n_static_tokens > 0`` static predictions are prepended:
    ``(B, n_static_tokens + N, pred_dim)``.

    Args:
        d_model:          Predictor model dimension.
        pred_dim:         Output embedding dimension.
        nhead:            Attention heads.
        dim_feedforward:  FFN hidden dimension.
        num_layers:       Number of predictor layers.
        max_segments:     Max sequence length (for standalone RoPE table).
        dropout:          Dropout probability.
        n_static_tokens:  Number of static token queries to prepend.
        activation:       ``\'swiglu\'`` or ``\'geglu\'``.
        rotary:           Shared :class:`RotaryEmbedding` from encoder (optional).
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
        n_static_tokens: int = 0,
        activation: str = "swiglu",
        rotary: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        if rotary is None:
            self.rotary = RotaryEmbedding(d_model // nhead, max_segments)
        else:
            self.rotary = rotary
        self.n_static_tokens = n_static_tokens
        if n_static_tokens > 0:
            self.static_queries = nn.Parameter(torch.randn(1, n_static_tokens, d_model) * 0.02)
        else:
            self.static_queries = None
        self.target_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            _PredictorLayer(d_model, nhead, dim_feedforward, dropout, activation, self.rotary)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_dim, bias=True)

    def forward(
        self,
        canonical_positions: torch.Tensor,
        student_context: torch.Tensor,
        student_positions: Optional[torch.Tensor] = None,
        canonical_mask: Optional[torch.Tensor] = None,
        student_key_padding_mask: Optional[torch.Tensor] = None,
        n_static: int = 0,
    ) -> torch.Tensor:
        B, N = canonical_positions.shape
        seg_q = self.target_query.expand(B, N, -1)
        if self.static_queries is not None:
            q = torch.cat([self.static_queries.expand(B, -1, -1), seg_q], dim=1)
            n_static_q = self.n_static_tokens
        else:
            q = seg_q
            n_static_q = 0

        q_kpm: Optional[torch.Tensor] = None
        if canonical_mask is not None:
            q_kpm = ~torch.cat([canonical_mask.new_ones(B, n_static_q), canonical_mask], dim=1)

        for layer in self.layers:
            q = layer(
                q, student_context,
                q_positions=canonical_positions,
                ctx_positions=student_positions,
                n_static=n_static_q,
                q_key_padding_mask=q_kpm,
                ctx_key_padding_mask=student_key_padding_mask,
            )
        return F.normalize(self.head(self.norm(q)), dim=-1)


# ---------------------------------------------------------------------------
# Shared encoder (student and teacher)
# ---------------------------------------------------------------------------

class ByteSegmentEncoder(nn.Module):
    """Full byte -> segment -> Transformer encoder pipeline (ModernBERT-style).

    Positional information is conveyed exclusively via RoPE; there is no
    additive absolute position embedding.

    Args:
        byte_dim:              Byte embedding / local-processor hidden dimension.
        seg_dim:               Segment / Transformer model dimension.
        n_byte_blocks:         Conv1D/SwiGLU residual blocks.
        n_encoder_layers:      Transformer encoder layers.
        n_heads:               Attention heads.
        ffn_dim:               GLU hidden dimension.
        max_segments:          Maximum canonical sequence length (for RoPE).
        kernel_size:           Conv1D kernel size.
        dropout:               Transformer dropout.
        byte_dropout:          Local byte processor dropout.
        n_additional_tokens:   Extra static tokens beyond ``[DOC]``.
        local_window_size:     Half-width of sliding-window local attention (0=global).
        global_attention_every_n: Global-attention layer period (1=all global).
        activation:            ``\'swiglu\'`` or ``\'geglu\'``.
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
        n_additional_tokens: int = 0,
        local_window_size: int = 0,
        global_attention_every_n: int = 1,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        self.n_additional_tokens = n_additional_tokens
        self.n_static = 1 + n_additional_tokens
        self.byte_embed = ByteInputEmbedding(byte_dim)
        self.local_proc = LocalByteProcessor(byte_dim, n_byte_blocks, kernel_size, byte_dropout)
        self.reducer = ByteToSegmentReducer(byte_dim, seg_dim)
        self.doc_embed = nn.Parameter(torch.randn(1, 1, seg_dim) * 0.02)
        if n_additional_tokens > 0:
            self.additional_token_embeds = nn.Parameter(
                torch.randn(1, n_additional_tokens, seg_dim) * 0.02
            )
        else:
            self.additional_token_embeds = None
        self.encoder = TransformerEncoderWithIntermediates(
            num_layers=n_encoder_layers,
            d_model=seg_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            max_seq_len=max_segments,
            local_window_size=local_window_size,
            global_attention_every_n=global_attention_every_n,
            activation=activation,
        )
        # Scalar buffer for checkpoint introspection
        self.register_buffer("_max_segments", torch.tensor(max_segments, dtype=torch.long), persistent=True)

    def forward(
        self,
        byte_values: torch.Tensor,
        byte_types: torch.Tensor,
        positions: torch.Tensor,
        seg_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns ``(output, layer_outputs)`` with output shape ``(B, n_static+N, seg_dim)``."""
        B, N = byte_values.shape[:2]
        x = self.byte_embed(byte_values, byte_types)
        x = self.local_proc(x)
        seg_vecs = self.reducer(x)                    # (B, N, seg_dim) -- no pos embed

        doc = self.doc_embed.expand(B, 1, -1)
        if self.additional_token_embeds is not None:
            extra = self.additional_token_embeds.expand(B, -1, -1)
            seq = torch.cat([doc, extra, seg_vecs], dim=1)
        else:
            seq = torch.cat([doc, seg_vecs], dim=1)

        if seg_mask is not None:
            kpm = ~torch.cat([seg_mask.new_ones(B, self.n_static), seg_mask], dim=1)
        else:
            kpm = None

        return self.encoder(seq, positions, self.n_static, key_padding_mask=kpm)


# ---------------------------------------------------------------------------
# Megatron-Core-style weight initialisation
# ---------------------------------------------------------------------------

def _apply_megatron_init(
    model: nn.Module,
    n_encoder_layers: int,
    n_predictor_layers: int,
    init_std: float = 0.02,
) -> None:
    """Apply Megatron-Core weight init rules.

    - Embeddings / most linears: N(0, 0.02)
    - Residual output projections: N(0, 0.02/sqrt(2*num_layers))
    - Biases: zero; LayerNorm weight: 1, bias: 0
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, init_std)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if getattr(module, "_is_residual_output", False):
                n_layers = n_predictor_layers if ".predictor." in f".{name}." else n_encoder_layers
                std = init_std / math.sqrt(2.0 * max(n_layers, 1))
            else:
                std = init_std
            nn.init.normal_(module.weight, 0.0, std)
    for name, param in model.named_parameters():
        if name.split(".")[-1] in ("doc_embed", "additional_token_embeds", "static_queries", "target_query"):
            nn.init.normal_(param, 0.0, init_std)


# ---------------------------------------------------------------------------
# Full JEPA model
# ---------------------------------------------------------------------------

class ByteSegmentJEPA(nn.Module):
    """Complete JEPA pretraining model (ModernBERT-style).

    Comprises student encoder + teacher encoder + segment predictor.
    The predictor now predicts ALL static tokens ([DOC] + additional) as well
    as the canonically masked segments, all via learned query embeddings + RoPE.

    New parameters vs. the original model:

    - ``n_additional_tokens``: extra static tokens beyond ``[DOC]``.
    - ``local_window_size``: sliding-window attention half-width (0=global).
    - ``global_attention_every_n``: global-attention layer period.
    - ``activation``: ``\'swiglu\'`` or ``\'geglu\'``.
    - ``teacher_mode``: ``\'ema\'`` or ``\'frozen\'``.
    - ``megatron_init``: apply Megatron-Core initialization.
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
        n_additional_tokens: int = 0,
        local_window_size: int = 0,
        global_attention_every_n: int = 1,
        activation: str = "swiglu",
        teacher_mode: str = "ema",
        megatron_init: bool = True,
    ) -> None:
        super().__init__()
        self.seg_dim = seg_dim
        self.pred_dim = pred_dim
        self.n_encoder_layers = n_encoder_layers
        self.n_predictor_layers = n_predictor_layers
        self.ema_momentum = ema_momentum
        self.n_additional_tokens = n_additional_tokens
        self.n_static = 1 + n_additional_tokens
        self.teacher_mode = teacher_mode
        if teacher_mode not in ("ema", "frozen"):
            raise ValueError(f"Unknown teacher_mode {teacher_mode!r}. Use \'ema\' or \'frozen\'.")

        if teacher_target_layers is None:
            start = max(n_encoder_layers - max(n_encoder_layers // 4, 1), 0)
            self.teacher_target_layers: list[int] = list(range(start, n_encoder_layers))
        else:
            self.teacher_target_layers = teacher_target_layers

        enc_kw: dict = dict(
            byte_dim=byte_dim, seg_dim=seg_dim, n_byte_blocks=n_byte_blocks,
            n_encoder_layers=n_encoder_layers, n_heads=n_heads, ffn_dim=ffn_dim,
            max_segments=max_segments, kernel_size=kernel_size, dropout=dropout,
            byte_dropout=byte_dropout, n_additional_tokens=n_additional_tokens,
            local_window_size=local_window_size,
            global_attention_every_n=global_attention_every_n, activation=activation,
        )
        self.student = ByteSegmentEncoder(**enc_kw)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Share RoPE table between encoder and predictor.
        shared_rotary = self.student.encoder.rotary
        self.predictor = SegmentPredictor(
            d_model=seg_dim, pred_dim=pred_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            num_layers=n_predictor_layers, max_segments=max_segments, dropout=dropout,
            n_static_tokens=self.n_static, activation=activation, rotary=shared_rotary,
        )
        self.teacher_target_proj = nn.Sequential(
            nn.LayerNorm(seg_dim),
            nn.Linear(seg_dim, pred_dim, bias=True),
        )
        if megatron_init:
            _apply_megatron_init(self, n_encoder_layers, n_predictor_layers)

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA update. No-op when ``teacher_mode == \'frozen\'``."""
        if self.teacher_mode == "frozen":
            return
        for sp, tp in zip(self.student.parameters(), self.teacher.parameters()):
            tp.data.mul_(momentum).add_(sp.data * (1.0 - momentum))

    # ------------------------------------------------------------------
    # Teacher targets
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_teacher_targets(
        self,
        clean_byte_values: torch.Tensor,
        clean_byte_types: torch.Tensor,
        canonical_positions: torch.Tensor,
        canonical_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(seg_targets, static_targets)`` -- both L2-normalised."""
        _, layer_outputs = self.teacher(
            clean_byte_values, clean_byte_types, canonical_positions, canonical_mask
        )
        upper = torch.stack([layer_outputs[i] for i in self.teacher_target_layers]).mean(0)
        static_targets = F.normalize(self.teacher_target_proj(upper[:, :self.n_static, :]), dim=-1)
        seg_targets = F.normalize(self.teacher_target_proj(upper[:, self.n_static:, :]), dim=-1)
        return seg_targets, static_targets

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        clean_byte_values: torch.Tensor,
        clean_byte_types: torch.Tensor,
        canonical_positions: torch.Tensor,
        canonical_mask: torch.Tensor,
        student_bytes: torch.Tensor,
        student_byte_types: torch.Tensor,
        student_positions: torch.Tensor,
        student_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full JEPA forward pass.

        Return keys:
        ``predicted_segments`` (B, N, pred_dim),
        ``teacher_seg_targets`` (B, N, pred_dim),
        ``predicted_doc`` (B, pred_dim),
        ``teacher_doc_targets`` (B, pred_dim),
        and optionally ``predicted_additional_static`` /
        ``teacher_additional_static_targets`` when n_additional_tokens > 0.
        """
        B = student_mask.shape[0]
        seg_targets, static_targets = self._compute_teacher_targets(
            clean_byte_values, clean_byte_types, canonical_positions, canonical_mask
        )
        student_out, _ = self.student(student_bytes, student_byte_types, student_positions, student_mask)
        student_kpm = ~torch.cat([student_mask.new_ones(B, self.n_static), student_mask], dim=1)

        predictor_out = self.predictor(
            canonical_positions, student_out,
            student_positions=student_positions,
            canonical_mask=canonical_mask,
            student_key_padding_mask=student_kpm,
            n_static=self.n_static,
        )  # (B, n_static + N, pred_dim)

        predicted_static = predictor_out[:, :self.n_static, :]
        predicted_segments = predictor_out[:, self.n_static:, :]

        result: dict[str, torch.Tensor] = {
            "predicted_segments":  predicted_segments,
            "teacher_seg_targets": seg_targets,
            "predicted_doc":       predicted_static[:, 0, :],
            "teacher_doc_targets": static_targets[:, 0, :],
        }
        if self.n_additional_tokens > 0:
            result["predicted_additional_static"] = predicted_static[:, 1:, :]
            result["teacher_additional_static_targets"] = static_targets[:, 1:, :]
        return result

    # ------------------------------------------------------------------
    # Partial checkpoint loading (gradual layer extension)
    # ------------------------------------------------------------------

    def load_partial_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretrained checkpoint into this (potentially larger) model.

        Student encoder layers present in the checkpoint are loaded; extra
        layers in the student (not in checkpoint) retain their Megatron init.
        The predictor is also loaded (strict=False).

        Teacher handling:
        - ``teacher_mode == \'ema\'``: teacher becomes deepcopy of loaded student.
        - ``teacher_mode == \'frozen\'``: teacher weights loaded from checkpoint
          (strict=False); new layers retain random init.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_sd: dict[str, torch.Tensor] = ckpt.get("state_dict", ckpt)
        raw_sd = {k.replace("_orig_mod.", "").removeprefix("model."): v for k, v in raw_sd.items()}

        def _extract(pfx: str) -> dict[str, torch.Tensor]:
            return {k[len(pfx):]: v for k, v in raw_sd.items() if k.startswith(pfx)}

        missing_s, unexpected_s = self.student.load_state_dict(_extract("student."), strict=False)
        if unexpected_s:
            logger.warning("load_partial_checkpoint: unexpected student keys: %s", unexpected_s)
        if missing_s:
            logger.info("load_partial_checkpoint: %d new student keys kept random: %s",
                        len(missing_s), missing_s[:10])

        missing_p, _ = self.predictor.load_state_dict(_extract("predictor."), strict=False)
        if missing_p:
            logger.info("load_partial_checkpoint: %d new predictor keys kept random: %s",
                        len(missing_p), missing_p[:5])

        proj_sd = _extract("teacher_target_proj.")
        if proj_sd:
            self.teacher_target_proj.load_state_dict(proj_sd, strict=False)

        if self.teacher_mode == "ema":
            self.teacher = copy.deepcopy(self.student)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            logger.info("load_partial_checkpoint: teacher re-initialised as deepcopy of student.")
        else:
            missing_t, _ = self.teacher.load_state_dict(_extract("teacher."), strict=False)
            if missing_t:
                logger.info("load_partial_checkpoint: %d frozen teacher keys absent: %s",
                            len(missing_t), missing_t[:5])
            logger.info("load_partial_checkpoint: frozen teacher loaded from checkpoint.")
