"""Dataset and DataModule for JEPA byte-segment pretraining.

Each JSONL row must contain at least ``id`` and ``text`` fields.  On every
``__getitem__`` call the document is:

1. Converted to UTF-8 bytes; if longer than ``max_segments × 32`` bytes a
   random crop is taken so that all parts of long documents are seen over
   multiple epochs.
2. The crop is split into 32-byte canonical segments.
3. A corrupted student view is generated via :func:`generate_student_view`.
4. Clean teacher input + student input + loss metadata are returned.

``JEPADataset`` builds a lightweight byte-offset index at startup (only
storing two integers per document) so that the full corpus is **never loaded
into memory**.  This scales to TB-scale corpora.  ``path`` may point to a
single JSONL file, a directory of ``*.jsonl`` shards, or you can pass a list
of paths explicitly.  An optional ``index_cache_dir`` lets you persist the
index to disk so that subsequent runs skip the scanning step.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from omegaconf import DictConfig

from text_classification.jepa.corruption import (
    CorruptionConfig,
    CorruptionType,
    PAD_BYTE,
    SEGMENT_SIZE,
    StudentView,
    generate_student_view,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _segment_raw_bytes(raw: bytes, max_segments: int) -> list[bytes]:
    """Split raw bytes into canonical SEGMENT_SIZE-byte segments (no text encoding)."""
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
# Dataset
# ---------------------------------------------------------------------------

class JEPADataset(Dataset):
    """Index-based, lazy-loading JEPA pretraining dataset.

    The dataset scans the JSONL file(s) once at construction time to record
    the **byte offset** of every valid line, then seeks directly to that
    offset on each ``__getitem__`` call.  Only one document is in memory at
    a time, regardless of corpus size.

    For documents whose UTF-8 encoding exceeds ``max_segments × 32`` bytes
    a random contiguous crop of exactly ``max_segments`` segments is taken
    on each access, so all parts of a long document are eventually seen.

    Args:
        path:             Path to a JSONL file, a directory of ``*.jsonl``
                          shards, or a list of such paths.
        corruption_cfg:   Corruption parameters applied to the student view.
        max_segments:     Maximum canonical segments per training example.
        seed:             Base random seed for fixed-seed (validation) mode.
        fixed_seed:       Deterministic corruption per index (for val sets).
        index_cache_dir:  Directory to cache the byte-offset index so that
                          subsequent runs skip the full scan.  ``None``
                          disables caching.
        crop_long_documents: If True (default), randomly crop documents
                          longer than ``max_segments``.  If False, the first
                          ``max_segments`` segments are used (old behaviour).
    """

    def __init__(
        self,
        path: Union[str, Path, list],
        corruption_cfg: CorruptionConfig,
        max_segments: int = 2048,
        seed: Optional[int] = None,
        fixed_seed: bool = False,
        index_cache_dir: Optional[Union[str, Path]] = None,
        crop_long_documents: bool = True,
    ) -> None:
        # Resolve shard paths
        if isinstance(path, (list, tuple)):
            self.paths = [Path(p) for p in path]
        else:
            p = Path(path)
            if p.is_dir():
                self.paths = sorted(p.glob("*.jsonl"))
            else:
                self.paths = [p]
        if not self.paths:
            raise ValueError(f"No JSONL files found at {path!r}")

        self.corruption_cfg = corruption_cfg
        self.max_segments = max_segments
        self.seed = seed
        self.fixed_seed = fixed_seed
        self.crop_long_documents = crop_long_documents

        # Build or load byte-offset index
        self._file_indices, self._byte_offsets = self._build_or_load_index(index_cache_dir)
        logger.info(
            "JEPADataset: %d documents across %d shard(s)",
            len(self._byte_offsets),
            len(self.paths),
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _index_cache_key(self) -> str:
        h = hashlib.sha256()
        for p in self.paths:
            stat = p.stat()
            h.update(f"{p}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        return h.hexdigest()[:24]

    def _build_or_load_index(
        self,
        cache_dir: Optional[Union[str, Path]],
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"jepa_index_{self._index_cache_key()}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                logger.info(
                    "Loaded document index (%d docs) from cache %s",
                    len(data["file_indices"]),
                    cache_path,
                )
                return data["file_indices"], data["byte_offsets"]

        logger.info("Building document index from %d shard(s) ...", len(self.paths))
        file_idx_list: list[int] = []
        offset_list: list[int] = []
        skipped = 0

        for file_idx, path in enumerate(self.paths):
            with path.open("rb") as fh:
                while True:
                    offset = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    if stripped and stripped[0:1] == b"{":
                        file_idx_list.append(file_idx)
                        offset_list.append(offset)
                    elif stripped:
                        skipped += 1

        if skipped:
            logger.warning("Skipped %d non-JSON-object lines during indexing.", skipped)

        file_indices = np.array(file_idx_list, dtype=np.uint32)
        byte_offsets = np.array(offset_list, dtype=np.int64)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, file_indices=file_indices, byte_offsets=byte_offsets)
            logger.info("Saved document index to %s", cache_path)

        return file_indices, byte_offsets

    # ------------------------------------------------------------------
    # Pickle safety for DataLoader workers
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_file_handles", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def _read_line(self, idx: int) -> dict[str, Any]:
        """Seek to the byte offset for *idx* and parse the JSON line."""
        file_idx = int(self._file_indices[idx])
        offset = int(self._byte_offsets[idx])

        # Cache one open file handle per shard per worker process.
        # _file_handles is excluded from pickling so it is always created
        # fresh inside the worker, preventing cross-process fd sharing.
        if not hasattr(self, "_file_handles"):
            self._file_handles: dict[int, Any] = {}
        if file_idx not in self._file_handles:
            self._file_handles[file_idx] = self.paths[file_idx].open("rb")

        fh = self._file_handles[file_idx]
        fh.seek(offset)
        return json.loads(fh.readline())

    def __getitem__(self, idx: int) -> dict[str, Any]:

        obj = self._read_line(idx)
        text: str = str(obj.get("text") or "")

        # RNG: deterministic per-example for validation, free for training
        rng = (
            random.Random((self.seed or 0) + idx)
            if self.fixed_seed
            else random.Random()
        )

        # Encode once; apply random crop for long documents
        raw = text.encode("utf-8", errors="replace")
        if raw:
            total_segs = (len(raw) + SEGMENT_SIZE - 1) // SEGMENT_SIZE
            if self.crop_long_documents and total_segs > self.max_segments:
                max_start = total_segs - self.max_segments
                start = rng.randint(0, max_start)
                raw = raw[start * SEGMENT_SIZE : (start + self.max_segments) * SEGMENT_SIZE]

        clean_segs = _segment_raw_bytes(raw, self.max_segments) if raw else []
        # Guarantee at least one segment so the model always has input
        if not clean_segs:
            clean_segs = [b"\x00" * SEGMENT_SIZE]
        N = len(clean_segs)

        # Foreign pool: lazily read one other document (skipped when frac==0)
        foreign_pool: Optional[list[bytes]] = None
        if self.corruption_cfg.foreign_replace_frac > 0.0 and len(self) > 1:
            other_idx = (idx + 1 + rng.randint(0, len(self) - 2)) % len(self)
            try:
                other_obj = self._read_line(other_idx)
                other_raw = str(other_obj.get("text") or "").encode("utf-8", errors="replace")
                foreign_pool = _segment_raw_bytes(other_raw, self.max_segments)
            except Exception:  # noqa: BLE001
                pass

        view: StudentView = generate_student_view(clean_segs, self.corruption_cfg, foreign_pool, rng)

        # Clean teacher input
        clean_byte_values = [
            list(seg[:SEGMENT_SIZE]) + [0] * max(0, SEGMENT_SIZE - len(seg))
            for seg in clean_segs
        ]
        clean_byte_types = [[int(CorruptionType.CLEAN)] * SEGMENT_SIZE] * N

        return {
            "id": obj.get("id", str(idx)),
            # Teacher
            "clean_byte_values": clean_byte_values,        # N × 32
            "clean_byte_types": clean_byte_types,          # N × 32
            "n_canonical": N,
            # Student
            "student_bytes": view.student_bytes,            # M × 32
            "student_positions": view.student_positions,    # M
            "student_byte_types": view.student_byte_types,  # M × 32
            # Loss metadata
            "segment_loss_weights": view.segment_loss_weights,        # N
            "segment_corruption_types": view.segment_corruption_types, # N
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def jepa_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate JEPA examples into padded batch tensors.

    Teacher sequences are padded to the maximum canonical length in the batch
    (``max_N``); student sequences are padded to the maximum visible length
    (``max_M``).  All padding positions are masked out via boolean masks.

    Returns a dict with keys::

        clean_byte_values   (B, max_N, 32)  long
        clean_byte_types    (B, max_N, 32)  long
        canonical_positions (B, max_N)      long  — 0-indexed canonical ids
        canonical_mask      (B, max_N)      bool  — True = valid
        student_bytes       (B, max_M, 32)  long
        student_byte_types  (B, max_M, 32)  long
        student_positions   (B, max_M)      long  — canonical ids of visible segs
        student_mask        (B, max_M)      bool  — True = valid
        segment_loss_weights(B, max_N)      float
        segment_corruption_types (B, max_N) long
    """
    B = len(batch)
    max_N = max(ex["n_canonical"] for ex in batch)
    max_M = max(max(len(ex["student_positions"]), 1) for ex in batch)

    # Teacher tensors
    clean_bv = torch.full((B, max_N, SEGMENT_SIZE), PAD_BYTE, dtype=torch.long)
    clean_bt = torch.full((B, max_N, SEGMENT_SIZE), int(CorruptionType.PADDING), dtype=torch.long)
    canonical_mask = torch.zeros(B, max_N, dtype=torch.bool)

    # Student tensors
    student_bv = torch.full((B, max_M, SEGMENT_SIZE), PAD_BYTE, dtype=torch.long)
    student_bt = torch.full((B, max_M, SEGMENT_SIZE), int(CorruptionType.PADDING), dtype=torch.long)
    student_pos = torch.zeros(B, max_M, dtype=torch.long)
    student_mask = torch.zeros(B, max_M, dtype=torch.bool)

    # Loss tensors
    seg_weights = torch.zeros(B, max_N, dtype=torch.float)
    seg_corr_types = torch.full((B, max_N), int(CorruptionType.PADDING), dtype=torch.long)

    for b, ex in enumerate(batch):
        N = ex["n_canonical"]
        M = len(ex["student_positions"])

        clean_bv[b, :N] = torch.tensor(ex["clean_byte_values"], dtype=torch.long)
        clean_bt[b, :N] = torch.tensor(ex["clean_byte_types"], dtype=torch.long)
        canonical_mask[b, :N] = True

        if M > 0:
            student_bv[b, :M] = torch.tensor(ex["student_bytes"], dtype=torch.long)
            student_bt[b, :M] = torch.tensor(ex["student_byte_types"], dtype=torch.long)
            student_pos[b, :M] = torch.tensor(ex["student_positions"], dtype=torch.long)
            student_mask[b, :M] = True

        seg_weights[b, :N] = torch.tensor(ex["segment_loss_weights"], dtype=torch.float)
        seg_corr_types[b, :N] = torch.tensor(ex["segment_corruption_types"], dtype=torch.long)

    canonical_positions = torch.arange(max_N, dtype=torch.long).unsqueeze(0).expand(B, -1).clone()

    return {
        "clean_byte_values": clean_bv,
        "clean_byte_types": clean_bt,
        "canonical_positions": canonical_positions,
        "canonical_mask": canonical_mask,
        "student_bytes": student_bv,
        "student_byte_types": student_bt,
        "student_positions": student_pos,
        "student_mask": student_mask,
        "segment_loss_weights": seg_weights,
        "segment_corruption_types": seg_corr_types,
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class JEPADataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for JEPA pretraining.

    Expected config layout::

        data:
          train: path/to/train.jsonl          # file, directory, or glob
          val:                                # single path or list of {path, name}
            - path: path/to/val.jsonl
              name: val
          max_segments: 2048
          num_workers: 4
          index_cache_dir: .cache/jepa_index  # optional; persists byte-offset index
        training:
          batch_size: 32
          val_batch_size: 32                  # optional
        corruption:                           # optional overrides for CorruptionConfig
          missing_span_prob: 0.3
          ...

    ``data.train`` may point to:

    * A single ``.jsonl`` file.
    * A directory — all ``*.jsonl`` files inside are used as shards.
    * A glob pattern — matched files are used as shards.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[JEPADataset] = None
        self.val_datasets: list[JEPADataset] = []
        self.val_names: list[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_corruption_cfg(self) -> CorruptionConfig:
        c = self.cfg.get("corruption", {})
        return CorruptionConfig(
            missing_span_prob=float(c.get("missing_span_prob", 0.3)),
            missing_span_frac_min=float(c.get("missing_span_frac_min", 0.05)),
            missing_span_frac_max=float(c.get("missing_span_frac_max", 0.25)),
            segment_dropout_prob=float(c.get("segment_dropout_prob", 0.05)),
            foreign_replace_frac=float(c.get("foreign_replace_frac", 0.05)),
            local_reorder_prob=float(c.get("local_reorder_prob", 0.0)),
            truncate_prob=float(c.get("truncate_prob", 0.1)),
            truncate_frac_max=float(c.get("truncate_frac_max", 0.2)),
            byte_corrupt_prob=float(c.get("byte_corrupt_prob", 0.05)),
            byte_heavy_frac=float(c.get("byte_heavy_frac", 0.10)),
        )

    def _resolve_paths(self, raw: str) -> list[Path]:
        """Resolve a file/directory/glob string to a sorted list of paths."""
        p = Path(raw)
        if p.is_dir():
            paths = sorted(p.glob("*.jsonl"))
        elif p.is_file():
            paths = [p]
        else:
            # Try as a glob pattern
            import glob as _glob
            matched = sorted(Path(m) for m in _glob.glob(raw, recursive=True))
            paths = matched
        if not paths:
            raise FileNotFoundError(f"No JSONL files found for data path: {raw!r}")
        return paths

    def _build_dataset(
        self,
        path: str | Path,
        fixed_seed: bool,
    ) -> JEPADataset:
        data_cfg = self.cfg.data
        paths = self._resolve_paths(str(path))
        return JEPADataset(
            path=paths,
            corruption_cfg=self._build_corruption_cfg(),
            max_segments=int(data_cfg.get("max_segments", 2048)),
            seed=int(self.cfg.get("seed", 42)),
            fixed_seed=fixed_seed,
            index_cache_dir=data_cfg.get("index_cache_dir", None),
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str = "fit") -> None:
        data_cfg = self.cfg.data

        if stage in ("fit", "train", None):
            self.train_dataset = self._build_dataset(data_cfg.train, fixed_seed=False)

        # Validation datasets
        val_cfg = data_cfg.get("val", None)
        if val_cfg is None:
            return

        self.val_datasets = []
        self.val_names = []

        if isinstance(val_cfg, str):
            self.val_datasets.append(self._build_dataset(val_cfg, fixed_seed=True))
            self.val_names.append("val")
        else:
            for entry in val_cfg:
                if hasattr(entry, "get"):
                    path = entry.get("path", str(entry))
                    name = entry.get("name", "val")
                else:
                    path = str(entry)
                    name = "val"
                self.val_datasets.append(self._build_dataset(path, fixed_seed=True))
                self.val_names.append(name)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        data_cfg = self.cfg.data
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=True,
            num_workers=int(data_cfg.get("num_workers", 4)),
            collate_fn=jepa_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        data_cfg = self.cfg.data
        val_bs = int(self.cfg.training.get("val_batch_size", self.cfg.training.batch_size))
        return [
            DataLoader(
                ds,
                batch_size=val_bs,
                shuffle=False,
                num_workers=int(data_cfg.get("num_workers", 4)),
                collate_fn=jepa_collate_fn,
                pin_memory=True,
            )
            for ds in self.val_datasets
        ]
