"""Logging setup and ClearML integration.

Usage::

    setup_logging("INFO")
    loggers = build_loggers(cfg)   # returns list[pl.loggers.Logger]
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to write to stdout with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def build_loggers(cfg: DictConfig) -> list[pl.loggers.Logger]:
    """Instantiate all configured experiment loggers.

    Currently supports:
    - ClearML (``clearml.enabled: true`` in config)
    """
    loggers: list[pl.loggers.Logger] = []

    clearml_cfg = cfg.get("clearml", None)
    if clearml_cfg is not None and clearml_cfg.get("enabled", False):
        cl = _build_clearml_logger(cfg, clearml_cfg)
        if cl is not None:
            loggers.append(cl)

    return loggers


def _build_clearml_logger(
    cfg: DictConfig, clearml_cfg: DictConfig
) -> Optional["ClearMLLogger"]:
    try:
        from clearml import Task  # type: ignore[import]
    except ImportError:
        logger.warning(
            "ClearML is not installed. Install with: pip install clearml"
        )
        return None

    task = Task.init(
        project_name=clearml_cfg.get("project", "text_classification"),
        task_name=clearml_cfg.get("task_name", "training"),
        auto_connect_frameworks=True,
    )
    task.connect(OmegaConf.to_container(cfg, resolve=True))
    return ClearMLLogger(task=task)


class ClearMLLogger(pl.loggers.Logger):
    """Minimal PyTorch Lightning logger that reports metrics to ClearML."""

    def __init__(self, task: Any) -> None:
        super().__init__()
        self._task = task
        self._cl_logger = task.get_logger()

    @property
    def name(self) -> str:
        return "ClearML"

    @property
    def version(self) -> str:
        return str(self._task.id)

    @property
    def experiment(self) -> Any:
        return self._task

    def log_metrics(
        self, metrics: dict[str, float], step: Optional[int] = None
    ) -> None:
        iteration = step or 0
        for key, value in metrics.items():
            parts = key.split("/", 1)
            title = parts[0]
            series = parts[1] if len(parts) > 1 else key
            self._cl_logger.report_scalar(
                title=title, series=series, value=value, iteration=iteration
            )

    def log_hyperparams(self, params: Any) -> None:
        pass  # already connected via task.connect in _build_clearml_logger

    def finalize(self, status: str) -> None:
        self._task.close()
