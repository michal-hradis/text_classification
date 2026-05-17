"""Tests for JEPA corruption module."""
from __future__ import annotations

import random

import pytest

from text_classification.jepa.corruption import (
    SEGMENT_SIZE,
    PAD_BYTE,
    MASK_BYTE,
    NOISE_BYTE,
    CorruptionConfig,
    CorruptionType,
    SEGMENT_LOSS_WEIGHTS,
    StudentView,
    generate_student_view,
    text_to_canonical_segments,
)


# ---------------------------------------------------------------------------
# text_to_canonical_segments
# ---------------------------------------------------------------------------

class TestTextToCanonicalSegments:
    def test_empty_text(self):
        segs = text_to_canonical_segments("")
        assert segs == []

    def test_segment_length(self):
        text = "a" * 100
        segs = text_to_canonical_segments(text)
        for seg in segs:
            assert len(seg) == SEGMENT_SIZE

    def test_n_segments(self):
        # 96 bytes → ceil(96/32) = 3 segments
        text = "x" * 96
        segs = text_to_canonical_segments(text)
        assert len(segs) == 3

    def test_partial_last_segment_zero_padded(self):
        text = "a" * 33  # 33 ASCII bytes → 2 segments, second has 1 byte + 31 zeros
        segs = text_to_canonical_segments(text)
        assert len(segs) == 2
        assert segs[1][1:] == bytes(31)  # zero-padded

    def test_max_segments(self):
        text = "z" * (SEGMENT_SIZE * 100)
        segs = text_to_canonical_segments(text, max_segments=10)
        assert len(segs) == 10

    def test_unicode_encoded_correctly(self):
        # Czech character ě is 2 bytes in UTF-8
        text = "ě" * 100
        segs = text_to_canonical_segments(text)
        assert len(segs) > 0
        # All segments have exactly SEGMENT_SIZE bytes
        for seg in segs:
            assert len(seg) == SEGMENT_SIZE


# ---------------------------------------------------------------------------
# generate_student_view
# ---------------------------------------------------------------------------

class TestGenerateStudentView:
    def _clean_segs(self, n: int = 20) -> list[bytes]:
        text = "Hello world! " * (n * SEGMENT_SIZE // 12 + 1)
        return text_to_canonical_segments(text, max_segments=n)

    def test_empty_input(self):
        view = generate_student_view([], CorruptionConfig())
        assert view.student_bytes == []
        assert view.student_positions == []
        assert view.segment_loss_weights == []

    def test_output_lengths_consistent(self):
        segs = self._clean_segs(10)
        view = generate_student_view(segs, CorruptionConfig(), rng=random.Random(0))
        N = len(segs)
        assert len(view.segment_loss_weights) == N
        assert len(view.segment_corruption_types) == N
        assert len(view.student_bytes) == len(view.student_positions)
        assert len(view.student_bytes) == len(view.student_byte_types)

    def test_student_positions_are_canonical_subset(self):
        segs = self._clean_segs(10)
        view = generate_student_view(segs, CorruptionConfig(), rng=random.Random(1))
        N = len(segs)
        for pos in view.student_positions:
            assert 0 <= pos < N

    def test_student_positions_sorted(self):
        segs = self._clean_segs(10)
        view = generate_student_view(segs, CorruptionConfig(), rng=random.Random(2))
        assert view.student_positions == sorted(view.student_positions)

    def test_segment_byte_length(self):
        segs = self._clean_segs(10)
        view = generate_student_view(segs, CorruptionConfig(), rng=random.Random(3))
        for seg_bytes in view.student_bytes:
            assert len(seg_bytes) == SEGMENT_SIZE
        for seg_types in view.student_byte_types:
            assert len(seg_types) == SEGMENT_SIZE

    def test_loss_weights_valid(self):
        segs = self._clean_segs(10)
        view = generate_student_view(segs, CorruptionConfig(), rng=random.Random(4))
        valid_weights = set(SEGMENT_LOSS_WEIGHTS.values())
        for w in view.segment_loss_weights:
            assert w in valid_weights, f"Unexpected weight {w}"

    def test_no_corruption_keeps_all_segments(self):
        """With all corruption probabilities set to 0 every segment is visible."""
        cfg = CorruptionConfig(
            missing_span_prob=0.0,
            segment_dropout_prob=0.0,
            truncate_prob=0.0,
            foreign_replace_frac=0.0,
            local_reorder_prob=0.0,
            byte_corrupt_prob=0.0,
        )
        segs = self._clean_segs(8)
        view = generate_student_view(segs, cfg, rng=random.Random(5))
        assert len(view.student_positions) == len(segs)
        for ct in view.segment_corruption_types:
            assert ct == int(CorruptionType.CLEAN)

    def test_missing_segments_absent_from_student(self):
        """Segments marked MISSING must not appear in the student sequence."""
        cfg = CorruptionConfig(
            missing_span_prob=1.0,
            missing_span_frac_min=0.3,
            missing_span_frac_max=0.5,
            segment_dropout_prob=0.0,
            truncate_prob=0.0,
            foreign_replace_frac=0.0,
            byte_corrupt_prob=0.0,
        )
        segs = self._clean_segs(20)
        view = generate_student_view(segs, cfg, rng=random.Random(6))
        missing_positions = {
            i
            for i, ct in enumerate(view.segment_corruption_types)
            if ct == int(CorruptionType.MISSING)
        }
        assert missing_positions.isdisjoint(set(view.student_positions))
        assert len(missing_positions) > 0  # some must be missing

    def test_foreign_replacement_bytes_differ(self):
        """Foreign-replaced segments should use the foreign bytes."""
        segs = self._clean_segs(20)
        foreign = [bytes([42] * SEGMENT_SIZE)] * 20
        cfg = CorruptionConfig(
            missing_span_prob=0.0,
            segment_dropout_prob=0.0,
            truncate_prob=0.0,
            foreign_replace_frac=0.5,
            byte_corrupt_prob=0.0,
        )
        view = generate_student_view(segs, cfg, foreign_pool=foreign, rng=random.Random(7))
        replaced = [
            i
            for i, ct in enumerate(view.segment_corruption_types)
            if ct == int(CorruptionType.REPLACED)
        ]
        # Replaced segments should be in the student sequence
        for pos_idx, pos in enumerate(view.student_positions):
            if view.segment_corruption_types[pos] == int(CorruptionType.REPLACED):
                assert view.student_bytes[pos_idx] == [42] * SEGMENT_SIZE

    def test_fixed_rng_deterministic(self):
        segs = self._clean_segs(10)
        cfg = CorruptionConfig()
        v1 = generate_student_view(segs, cfg, rng=random.Random(99))
        v2 = generate_student_view(segs, cfg, rng=random.Random(99))
        assert v1.student_positions == v2.student_positions
        assert v1.segment_corruption_types == v2.segment_corruption_types

    def test_corruption_type_enum_values(self):
        assert CorruptionType.CLEAN == 0
        assert CorruptionType.PADDING == 6
        assert len(CorruptionType) == 7
