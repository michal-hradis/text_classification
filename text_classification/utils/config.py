"""YAML configuration loading with optional base-config inheritance.

Usage::

    cfg = load_config("configs/robeczech_base.yaml", overrides=["optimizer.lr=3e-5"])

Config files may declare ``_base_: <relative_path>`` to inherit and override
from another config file (resolved relative to the config file's own directory).
Inheritance is recursive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(
    path: str | Path,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Load a YAML config file, resolve ``_base_`` inheritance, apply overrides.

    Args:
        path: Path to a YAML configuration file.
        overrides: Optional list of dot-notation overrides, e.g.
            ``["optimizer.lr=3e-5", "training.batch_size=32"]``.

    Returns:
        Merged :class:`omegaconf.DictConfig`.
    """
    path = Path(path)
    raw: DictConfig = OmegaConf.load(path)

    if "_base_" in raw:
        base_path = path.parent / str(raw["_base_"])
        base_cfg = load_config(base_path)  # recursive
        # Merge: base first, then current file (without _base_ key)
        raw_no_base = OmegaConf.masked_copy(raw, [k for k in raw if k != "_base_"])
        cfg: DictConfig = OmegaConf.merge(base_cfg, raw_no_base)
    else:
        cfg = raw

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg


def validate_config(cfg: DictConfig) -> None:
    """Raise ``ValueError`` for missing required configuration fields."""
    required = [
        "model.name_or_path",
        "data.train",
        "training.batch_size",
        "training.max_steps",
        "optimizer.lr",
        "tasks",
    ]
    for field in required:
        node: Any = cfg
        for key in field.split("."):
            try:
                node = node[key]
            except (KeyError, TypeError):
                raise ValueError(f"Missing required config field: {field!r}")
