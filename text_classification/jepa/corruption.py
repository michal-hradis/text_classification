"""Student-view corruption for JEPA byte-segment pretraining.

Text is first converted to UTF-8 bytes and split into fixed-size canonical
segments of ``SEGMENT_SIZE`` bytes each.  Corruption is applied only to the
student view; the teacher always receives the clean canonical segments.

Corruption types follow Section 7 of the JEPA pretraining specification.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_SIZE: int = 16      # bytes per canonical segment
PAD_BYTE: int = 256         # padding byte (outside 0-255 range)
MASK_BYTE: int = 257        # mask sentinel
NOISE_BYTE: int = 258       # random noise sentinel
BYTE_VOCAB_SIZE: int = 259  # 0-258 inclusive


# ---------------------------------------------------------------------------
# Corruption labels
# ---------------------------------------------------------------------------

class CorruptionType(IntEnum):
    """Per-byte / per-segment corruption label used as model input feature."""
    CLEAN = 0
    LIGHTLY_CORRUPTED = 1
    HEAVILY_CORRUPTED = 2
    REPLACED = 3      # whole segment replaced with foreign content
    INSERTED = 4      # inserted distractor (no clean target)
    MISSING = 5       # absent from student sequence; tracked for loss weight only
    PADDING = 6       # padding byte / padding segment


# Loss weights by canonical segment corruption type (spec §8.1)
SEGMENT_LOSS_WEIGHTS: dict[int, float] = {
    int(CorruptionType.MISSING): 1.0,
    int(CorruptionType.REPLACED): 1.0,
    int(CorruptionType.HEAVILY_CORRUPTED): 0.7,
    int(CorruptionType.LIGHTLY_CORRUPTED): 0.4,
    int(CorruptionType.CLEAN): 0.15,
    int(CorruptionType.INSERTED): 0.0,
    int(CorruptionType.PADDING): 0.0,
}


# ---------------------------------------------------------------------------
# Corruption configuration
# ---------------------------------------------------------------------------

@dataclass
class CorruptionConfig:
    """Parameters controlling corruption severity (can be stage-dependent)."""

    # Segment-level corruptions
    missing_span_prob: float = 0.3          # P(applying at least one missing span)
    missing_span_frac_min: float = 0.05     # min fraction of segs removed per span
    missing_span_frac_max: float = 0.25     # max fraction of segs removed per span
    segment_dropout_prob: float = 0.05      # per-segment probability of dropout
    foreign_replace_frac: float = 0.05      # fraction of segs replaced with foreign
    local_reorder_prob: float = 0.0         # fraction of adjacent pairs swapped
    truncate_prob: float = 0.1              # P(truncation from start or end)
    truncate_frac_max: float = 0.2          # max fraction to truncate

    # Byte-level corruptions (within retained segments)
    byte_corrupt_prob: float = 0.05         # per-byte probability of corruption
    byte_heavy_frac: float = 0.10           # byte-frac threshold → HEAVILY_CORRUPTED


# ---------------------------------------------------------------------------
# Text → canonical segments
# ---------------------------------------------------------------------------

def text_to_canonical_segments(text: str, max_segments: int = 2048) -> list[bytes]:
    """Convert text to UTF-8 and split into fixed-size canonical segments.

    Each segment is exactly ``SEGMENT_SIZE`` bytes; the last chunk is
    zero-padded if shorter.

    Args:
        text:         Input text string.
        max_segments: Maximum number of segments to return.

    Returns:
        List of byte strings, each of length ``SEGMENT_SIZE``.
    """
    raw = text.encode("utf-8", errors="replace")
    segments: list[bytes] = []
    for i in range(0, len(raw), SEGMENT_SIZE):
        chunk = raw[i : i + SEGMENT_SIZE]
        if len(chunk) < SEGMENT_SIZE:
            chunk = chunk + b"\x00" * (SEGMENT_SIZE - len(chunk))
        segments.append(chunk)
        if len(segments) >= max_segments:
            break
    return segments


# ---------------------------------------------------------------------------
# Student view dataclass
# ---------------------------------------------------------------------------

@dataclass
class StudentView:
    """Output of :func:`generate_student_view`.

    Attributes:
        student_bytes:            List of M lists of ``SEGMENT_SIZE`` ints
                                  (byte values 0-258).
        student_positions:        List of M canonical position indices for
                                  each visible segment.
        student_byte_types:       List of M lists of ``SEGMENT_SIZE`` ints
                                  (CorruptionType per byte).
        segment_loss_weights:     List of N floats — loss weight per canonical
                                  segment.
        segment_corruption_types: List of N ints — CorruptionType per canonical
                                  segment (for diagnostics / logging).
    """

    student_bytes: list[list[int]]         # M × SEGMENT_SIZE
    student_positions: list[int]           # M
    student_byte_types: list[list[int]]    # M × SEGMENT_SIZE
    segment_loss_weights: list[float]      # N
    segment_corruption_types: list[int]    # N


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_byte_corruption(
    seg_bytes: list[int],
    byte_corrupt_prob: float,
    byte_heavy_frac: float,
    rng: random.Random,
) -> tuple[list[int], list[int], CorruptionType]:
    """Apply byte-level corruption within a single segment.

    Returns:
        Tuple of (corrupted byte values, per-byte CorruptionType ints,
        segment-level CorruptionType).
    """
    n_bytes = len(seg_bytes)
    positions_to_corrupt = [i for i in range(n_bytes) if rng.random() < byte_corrupt_prob]

    if not positions_to_corrupt:
        return seg_bytes, [int(CorruptionType.CLEAN)] * n_bytes, CorruptionType.CLEAN

    frac = len(positions_to_corrupt) / n_bytes
    seg_type = (
        CorruptionType.HEAVILY_CORRUPTED
        if frac > byte_heavy_frac
        else CorruptionType.LIGHTLY_CORRUPTED
    )

    corrupted = list(seg_bytes)
    byte_types = [int(CorruptionType.CLEAN)] * n_bytes

    for pos in positions_to_corrupt:
        r = rng.random()
        if r < 0.5:
            corrupted[pos] = rng.randint(0, 255)   # random byte
        elif r < 0.75:
            corrupted[pos] = MASK_BYTE
        else:
            corrupted[pos] = NOISE_BYTE
        byte_types[pos] = int(seg_type)

    return corrupted, byte_types, seg_type


def _seg_to_ints(seg: bytes) -> list[int]:
    """Convert a bytes object to a list of ints padded to SEGMENT_SIZE."""
    vals = list(seg[:SEGMENT_SIZE])
    vals += [PAD_BYTE] * (SEGMENT_SIZE - len(vals))
    return vals


# ---------------------------------------------------------------------------
# Main corruption function
# ---------------------------------------------------------------------------

def generate_student_view(
    clean_segments: list[bytes],
    cfg: CorruptionConfig,
    foreign_pool: Optional[list[bytes]] = None,
    rng: Optional[random.Random] = None,
) -> StudentView:
    """Generate a corrupted student view from clean canonical segments.

    The teacher always receives the unmodified ``clean_segments``.  This
    function returns the student view together with per-segment metadata
    needed for the loss computation.

    Args:
        clean_segments: N segments, each ``SEGMENT_SIZE`` bytes (from
                        :func:`text_to_canonical_segments`).
        cfg:            Corruption configuration.
        foreign_pool:   Optional pool of byte segments from other documents
                        used for foreign-replacement corruption.
        rng:            Optional :class:`random.Random` instance for
                        reproducibility.

    Returns:
        :class:`StudentView` with all fields populated.
    """
    if rng is None:
        rng = random.Random()

    N = len(clean_segments)
    if N == 0:
        return StudentView([], [], [], [], [])

    # --- Internal state per canonical segment ---
    # (visible: bool, byte_vals: list[int], byte_types: list[int], seg_type: CorruptionType)
    state: list[tuple[bool, list[int], list[int], CorruptionType]] = []
    for seg in clean_segments:
        bv = _seg_to_ints(seg)
        state.append((True, bv, [int(CorruptionType.CLEAN)] * SEGMENT_SIZE, CorruptionType.CLEAN))

    # --- 1. Missing span — remove contiguous spans of segments (heaviest corruption) ---
    if rng.random() < cfg.missing_span_prob and N > 2:
        n_missing = max(1, int(rng.uniform(cfg.missing_span_frac_min, cfg.missing_span_frac_max) * N))
        start = rng.randint(0, N - n_missing)
        for i in range(start, start + n_missing):
            vis, bv, bt, _ = state[i]
            state[i] = (False, bv, bt, CorruptionType.MISSING)

    # --- 2. Segment dropout — independently drop segments at random (heavy corruption) ---
    for i in range(N):
        if rng.random() < cfg.segment_dropout_prob:
            vis, bv, bt, cur = state[i]
            state[i] = (False, bv, bt, CorruptionType.MISSING)

    # --- 3. Truncation — remove contiguous spans from start or end (heaviest corruption) ---
    if rng.random() < cfg.truncate_prob and N > 2:
        n_trunc = max(1, int(rng.uniform(0.0, cfg.truncate_frac_max) * N))
        if rng.random() < 0.5:
            for i in range(n_trunc):
                vis, bv, bt, _ = state[i]
                state[i] = (False, bv, bt, CorruptionType.MISSING)
        else:
            for i in range(N - n_trunc, N):
                vis, bv, bt, _ = state[i]
                state[i] = (False, bv, bt, CorruptionType.MISSING)

    # --- 4. Foreign replacement — replace segments with content from other documents (heavy corruption) ---
    if foreign_pool and cfg.foreign_replace_frac > 0.0:
        n_replace = max(0, int(cfg.foreign_replace_frac * N))
        idxs = rng.sample(range(N), k=min(n_replace, N))
        for i in idxs:
            vis, bv, bt, cur = state[i]
            if cur == CorruptionType.MISSING:
                continue
            fseg = rng.choice(foreign_pool)
            foreign_bv = _seg_to_ints(fseg)
            state[i] = (
                True,
                foreign_bv,
                [int(CorruptionType.REPLACED)] * SEGMENT_SIZE,
                CorruptionType.REPLACED,
            )

    # --- 5. Local reorder — swap adjacent segments (light corruption) ---
    if cfg.local_reorder_prob > 0.0:
        for i in range(N - 1):
            if rng.random() < cfg.local_reorder_prob:
                vi, bv_i, bt_i, ti = state[i]
                vj, bv_j, bt_j, tj = state[i + 1]
                if (
                    vi and vj
                    and ti == CorruptionType.CLEAN
                    and tj == CorruptionType.CLEAN
                ):
                    # Swap content, keep canonical positions
                    state[i] = (True, bv_j, bt_j, CorruptionType.LIGHTLY_CORRUPTED)
                    state[i + 1] = (True, bv_i, bt_i, CorruptionType.LIGHTLY_CORRUPTED)

    # --- 6. Byte-level corruption on retained segments — randomly corrupt bytes within segments (light corruption) ---
    updated: list[tuple[bool, list[int], list[int], CorruptionType]] = []
    for vis, bv, bt, seg_type in state:
        if vis and seg_type in (CorruptionType.CLEAN, CorruptionType.LIGHTLY_CORRUPTED):
            new_bv, new_bt, new_seg_type = _apply_byte_corruption(
                bv, cfg.byte_corrupt_prob, cfg.byte_heavy_frac, rng
            )
            # Escalate if byte corruption is heavier
            if new_seg_type != CorruptionType.CLEAN and seg_type == CorruptionType.CLEAN:
                seg_type = new_seg_type
            updated.append((vis, new_bv, new_bt, seg_type))
        else:
            updated.append((vis, bv, bt, seg_type))
    state = updated

    # --- Build outputs — only include visible segments in student view, but track loss weights and corruption types for all segments ---
    student_bytes: list[list[int]] = []
    student_positions: list[int] = []
    student_byte_types: list[list[int]] = []
    segment_loss_weights: list[float] = []
    segment_corruption_types: list[int] = []

    for i, (vis, bv, bt, seg_type) in enumerate(state):
        segment_corruption_types.append(int(seg_type))
        segment_loss_weights.append(SEGMENT_LOSS_WEIGHTS.get(int(seg_type), 0.0))
        if vis:
            # Ensure exactly SEGMENT_SIZE entries
            bv_padded = (bv + [PAD_BYTE] * SEGMENT_SIZE)[:SEGMENT_SIZE]
            bt_padded = (bt + [int(CorruptionType.PADDING)] * SEGMENT_SIZE)[:SEGMENT_SIZE]
            student_bytes.append(bv_padded)
            student_positions.append(i)
            student_byte_types.append(bt_padded)

    return StudentView(
        student_bytes=student_bytes,
        student_positions=student_positions,
        student_byte_types=student_byte_types,
        segment_loss_weights=segment_loss_weights,
        segment_corruption_types=segment_corruption_types,
    )
