"""Tests for JEPA data module."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from text_classification.jepa.corruption import (
    CorruptionConfig,
    CorruptionType,
    PAD_BYTE,
    SEGMENT_SIZE,
)
from text_classification.jepa.data import JEPADataset, jepa_collate_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    """Write a small JSONL file for testing."""
    records = [
        {"id": f"doc_{i:03d}", "text": f"Document {i}: " + "word " * 30}
        for i in range(5)
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


@pytest.fixture()
def corruption_cfg() -> CorruptionConfig:
    return CorruptionConfig(
        missing_span_prob=0.3,
        segment_dropout_prob=0.05,
        foreign_replace_frac=0.0,
        byte_corrupt_prob=0.02,
        truncate_prob=0.1,
    )


# ---------------------------------------------------------------------------
# JEPADataset
# ---------------------------------------------------------------------------

class TestJEPADataset:
    def test_loads_examples(self, tmp_jsonl: Path, corruption_cfg: CorruptionConfig):
        ds = JEPADataset(tmp_jsonl, corruption_cfg, max_segments=32)
        assert len(ds) == 5

    def test_getitem_keys(self, tmp_jsonl: Path, corruption_cfg: CorruptionConfig):
        ds = JEPADataset(tmp_jsonl, corruption_cfg, max_segments=32)
        ex = ds[0]
        required = {
            "id", "clean_byte_values", "clean_byte_types", "n_canonical",
            "student_bytes", "student_positions", "student_byte_types",
            "segment_loss_weights", "segment_corruption_types",
        }
        assert required.issubset(ex.keys())

    def test_clean_byte_values_shape(self, tmp_jsonl: Path, corruption_cfg: CorruptionConfig):
        ds = JEPADataset(tmp_jsonl, corruption_cfg, max_segments=32)
        ex = ds[0]
        N = ex["n_canonical"]
        assert len(ex["clean_byte_values"]) == N
        for row in ex["clean_byte_values"]:
            assert len(row) == SEGMENT_SIZE

    def test_student_positions_subset(self, tmp_jsonl: Path, corruption_cfg: CorruptionConfig):
        ds = JEPADataset(tmp_jsonl, corruption_cfg, max_segments=32, seed=0, fixed_seed=True)
        ex = ds[0]
        N = ex["n_canonical"]
        for p in ex["student_positions"]:
            assert 0 <= p < N

    def test_n_canonical_bounded(self, tmp_jsonl: Path, corruption_cfg: CorruptionConfig):
        ds = JEPADataset(tmp_jsonl, corruption_cfg, max_segments=4)
        for i in range(len(ds)):
            ex = ds[i]
            assert ex["n_canonical"] <= 4

    def test_fixed_seed_deterministic(self, tmp_jsonl: Path, corruption_cfg: CorruptionConfig):
        ds = JEPADataset(tmp_jsonl, corruption_cfg, max_segments=32, seed=7, fixed_seed=True)
        ex1 = ds[0]
        ex2 = ds[0]
        assert ex1["student_positions"] == ex2["student_positions"]

    def test_empty_lines_skipped(self, tmp_path: Path, corruption_cfg: CorruptionConfig):
        p = tmp_path / "sparse.jsonl"
        p.write_text(
            '{"id": "a", "text": "Hello world"}\n\n{"id": "b", "text": "Foo bar baz"}\n',
            encoding="utf-8",
        )
        ds = JEPADataset(p, corruption_cfg)
        assert len(ds) == 2

    def test_directory_of_shards(self, tmp_path: Path, corruption_cfg: CorruptionConfig):
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        for shard in range(3):
            p = shard_dir / f"shard_{shard:02d}.jsonl"
            p.write_text(
                "\n".join(
                    json.dumps({"id": f"s{shard}_d{i}", "text": "word " * 20})
                    for i in range(4)
                ),
                encoding="utf-8",
            )
        ds = JEPADataset(shard_dir, corruption_cfg, max_segments=16)
        assert len(ds) == 12
        ex = ds[0]
        assert "clean_byte_values" in ex

    def test_long_document_cropped_to_max_segments(self, tmp_path: Path, corruption_cfg: CorruptionConfig):
        """Documents longer than max_segments×32 bytes must be cropped, not truncated."""
        max_seg = 4
        # 10× the max segment length → always triggers crop
        long_text = "A" * (max_seg * SEGMENT_SIZE * 10)
        p = tmp_path / "long.jsonl"
        p.write_text(json.dumps({"id": "long", "text": long_text}), encoding="utf-8")
        ds = JEPADataset(p, corruption_cfg, max_segments=max_seg)
        ex = ds[0]
        assert ex["n_canonical"] == max_seg

    def test_random_crop_visits_different_regions(self, tmp_path: Path, corruption_cfg: CorruptionConfig):
        """Two fetches of the same long document should (with high probability) crop at different offsets."""
        max_seg = 4
        # First byte of each segment encodes position so we can detect the crop start
        raw = bytes(range(256)) * 100  # 25600 bytes = 800 segments
        long_text = raw.decode("latin-1")
        p = tmp_path / "long2.jsonl"
        p.write_text(json.dumps({"id": "x", "text": long_text}), encoding="utf-8")
        ds = JEPADataset(p, corruption_cfg, max_segments=max_seg, fixed_seed=False)
        # Sample several times; first-segment byte should vary
        first_bytes = {ds[0]["clean_byte_values"][0][0] for _ in range(20)}
        assert len(first_bytes) > 1, "Expected random crop to produce different first bytes"

    def test_index_cache_reused(self, tmp_path: Path, corruption_cfg: CorruptionConfig):
        """Second construction with same cache_dir must load from cache (no re-scan)."""
        p = tmp_path / "data.jsonl"
        p.write_text(
            "\n".join(json.dumps({"id": str(i), "text": "hello " * 5}) for i in range(6)),
            encoding="utf-8",
        )
        cache_dir = tmp_path / "idx_cache"
        ds1 = JEPADataset(p, corruption_cfg, index_cache_dir=cache_dir)
        assert len(list(cache_dir.glob("*.npz"))) == 1
        ds2 = JEPADataset(p, corruption_cfg, index_cache_dir=cache_dir)
        assert len(ds1) == len(ds2) == 6


# ---------------------------------------------------------------------------
# jepa_collate_fn
# ---------------------------------------------------------------------------

class TestJEPACollateFn:
    def _make_batch(self, n_docs: int, seg_range=(3, 8)) -> list[dict]:
        import random
        rng = random.Random(42)
        batch = []
        for i in range(n_docs):
            N = rng.randint(*seg_range)
            M = rng.randint(1, N)
            batch.append({
                "id": str(i),
                "clean_byte_values": [[j % 256] * SEGMENT_SIZE for j in range(N)],
                "clean_byte_types": [[int(CorruptionType.CLEAN)] * SEGMENT_SIZE] * N,
                "n_canonical": N,
                "student_bytes": [[j % 200] * SEGMENT_SIZE for j in range(M)],
                "student_positions": sorted(rng.sample(range(N), k=M)),
                "student_byte_types": [[int(CorruptionType.CLEAN)] * SEGMENT_SIZE] * M,
                "segment_loss_weights": [0.15] * N,
                "segment_corruption_types": [int(CorruptionType.CLEAN)] * N,
            })
        return batch

    def test_output_keys(self):
        batch = self._make_batch(3)
        out = jepa_collate_fn(batch)
        expected_keys = {
            "clean_byte_values", "clean_byte_types", "canonical_positions",
            "canonical_mask", "student_bytes", "student_byte_types",
            "student_positions", "student_mask", "segment_loss_weights",
            "segment_corruption_types",
        }
        assert expected_keys == set(out.keys())

    def test_batch_dimension(self):
        B = 4
        out = jepa_collate_fn(self._make_batch(B))
        for key in ["clean_byte_values", "student_bytes", "canonical_mask"]:
            assert out[key].shape[0] == B

    def test_clean_byte_values_shape(self):
        batch = self._make_batch(3)
        out = jepa_collate_fn(batch)
        B, max_N, S = out["clean_byte_values"].shape
        assert S == SEGMENT_SIZE
        assert max_N == max(ex["n_canonical"] for ex in batch)

    def test_canonical_mask_marks_padding(self):
        batch = self._make_batch(3)
        out = jepa_collate_fn(batch)
        for b, ex in enumerate(batch):
            N = ex["n_canonical"]
            assert out["canonical_mask"][b, :N].all()
            if out["canonical_mask"].shape[1] > N:
                assert not out["canonical_mask"][b, N:].any()

    def test_student_mask_marks_valid(self):
        batch = self._make_batch(3)
        out = jepa_collate_fn(batch)
        for b, ex in enumerate(batch):
            M = len(ex["student_positions"])
            assert out["student_mask"][b, :M].all()

    def test_canonical_positions_arange(self):
        batch = self._make_batch(2)
        out = jepa_collate_fn(batch)
        max_N = out["canonical_positions"].shape[1]
        expected = torch.arange(max_N)
        assert (out["canonical_positions"][0] == expected).all()

    def test_segment_loss_weights_zero_for_padding(self):
        batch = self._make_batch(3)
        out = jepa_collate_fn(batch)
        for b, ex in enumerate(batch):
            N = ex["n_canonical"]
            if out["segment_loss_weights"].shape[1] > N:
                assert (out["segment_loss_weights"][b, N:] == 0).all()

    def test_single_example_batch(self):
        batch = self._make_batch(1)
        out = jepa_collate_fn(batch)
        assert out["clean_byte_values"].shape[0] == 1
