#!/usr/bin/env python3
"""JEPA byte-segment pretraining — package entry point.

This module is the canonical entry point (referenced by ``pyproject.toml``
as ``tc-jepa-pretrain``) and is also called by the convenience wrapper
``jepa_pretrain.py`` at the repository root.

Usage::

    python jepa_pretrain.py configs/jepa_base.yaml
    python jepa_pretrain.py configs/jepa_smoketest.yaml model.seg_dim=128
    tc-jepa-pretrain configs/jepa_base.yaml data.train=data/pretrain.jsonl

Config overrides use dot-notation and are applied on top of the loaded YAML.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from text_classification.jepa.data import JEPADataModule
from text_classification.jepa.lightning_module import JEPAPretrainingModule
from text_classification.utils.config import load_config
from text_classification.utils.logging import build_loggers, setup_logging

logger = logging.getLogger(__name__)


def validate_jepa_config(cfg) -> None:
    """Raise ``ValueError`` for missing required JEPA configuration fields."""
    required = [
        "data.train",
        "training.batch_size",
        "training.max_steps",
        "optimizer.lr",
    ]
    for field in required:
        node = cfg
        for key in field.split("."):
            try:
                node = node[key]
            except (KeyError, TypeError):
                raise ValueError(f"Missing required config field: {field!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="JEPA byte-segment pretraining.")
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dot-notation config overrides, e.g. optimizer.lr=1e-4.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    validate_jepa_config(cfg)

    setup_logging(cfg.get("log_level", "INFO"))
    logger.info("Effective configuration:\n%s", OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # DataModule
    data_module = JEPADataModule(cfg)
    data_module.setup("fit")
    val_names = data_module.val_names or ["val"]

    # Model
    model_module = JEPAPretrainingModule(cfg, val_dataset_names=val_names)

    # Callbacks
    callbacks: list[pl.Callback] = [
        LearningRateMonitor(logging_interval="step"),
    ]

    ckpt_cfg = cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        monitor = ckpt_cfg.get(
            "monitor",
            f"val/{val_names[0]}/loss" if val_names else "train/loss",
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_cfg.get("dirpath", "checkpoints/jepa"),
                filename="jepa-step={step}-{" + monitor.replace("/", "_") + ":.4f}",
                monitor=monitor,
                mode=ckpt_cfg.get("mode", "min"),
                save_top_k=ckpt_cfg.get("save_top_k", 3),
                save_last=ckpt_cfg.get("save_last", True),
                every_n_train_steps=cfg.training.get("val_every_n_steps", None),
                auto_insert_metric_name=False,
            )
        )

    # Loggers
    pl_loggers = build_loggers(cfg)

    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        val_check_interval=cfg.training.get("val_every_n_steps", 1.0),
        accumulate_grad_batches=cfg.training.get("grad_accumulation", 1),
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=cfg.training.get("gradient_clip_algorithm", "norm"),
        precision=cfg.training.get("precision", "bf16-mixed"),
        devices=cfg.training.get("devices", "auto"),
        strategy=cfg.training.get("strategy", "auto"),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 10),
        callbacks=callbacks,
        logger=pl_loggers if pl_loggers else True,
        enable_checkpointing=ckpt_cfg.get("enabled", True),
        deterministic=cfg.get("deterministic", False),
    )

    ckpt_path = (
        ckpt_cfg.get("ckpt_path", None)
        or ckpt_cfg.get("resume_from_checkpoint", None)
        or cfg.get("ckpt_path", None)
        or cfg.get("resume_from_checkpoint", None)
    )

    if ckpt_path:
        logger.info("Resuming JEPA pretraining from checkpoint: %s", ckpt_path)

    logger.info("Starting JEPA pretraining (max_steps=%d)…", cfg.training.max_steps)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)
    logger.info("JEPA pretraining complete.")


if __name__ == "__main__":
    main()
