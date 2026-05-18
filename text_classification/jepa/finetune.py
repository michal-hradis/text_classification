#!/usr/bin/env python3
"""Finetuning entry point for JEPA-based text classification.

Loads a pretrained JEPA byte-segment encoder from a checkpoint and finetunes
it for multi-task multi-label text classification using the standard
classification pipeline.

Usage::

    python -m text_classification.jepa.finetune configs/jepa_finetune_base.yaml
    python -m text_classification.jepa.finetune configs/jepa_finetune_base.yaml \\
        model.checkpoint_path=checkpoints/jepa/last.ckpt \\
        optimizer.lr=1e-4

    # Console entry point (after pip install -e .)
    tc-jepa-finetune configs/jepa_finetune_base.yaml

Config overrides use dot-notation and are applied on top of the loaded YAML.

Required config fields (not present in ``base.yaml``)::

    model.checkpoint_path   Path to a JEPA pretraining ``.ckpt`` file.
    data.max_segments       Maximum byte-segments per example (replaces max_length).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from text_classification.jepa.finetune_data import JEPAClassificationDataModule
from text_classification.jepa.finetune_module import JEPAFinetuneModule
from text_classification.utils.config import load_config
from text_classification.utils.logging import build_loggers, setup_logging

logger = logging.getLogger(__name__)


def _validate_jepa_finetune_config(cfg: object) -> None:
    """Raise ``ValueError`` for missing required JEPA finetune config fields."""
    required = [
        "model.checkpoint_path",
        "data.train",
        "training.batch_size",
        "training.max_steps",
        "optimizer.lr",
        "tasks",
    ]
    for field in required:
        node = cfg
        for key in field.split("."):
            try:
                node = node[key]  # type: ignore[index]
            except (KeyError, TypeError):
                raise ValueError(f"Missing required config field: {field!r}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Finetune a pretrained JEPA encoder for multi-label text classification."
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file.")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dot-notation overrides applied on top of the config.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    _validate_jepa_finetune_config(cfg)

    setup_logging(cfg.get("log_level", "INFO"))
    logger.info("Effective configuration:\n%s", OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Tasks and class definitions
    tasks: list[str] = list(cfg.tasks.keys())
    class_lists: dict[str, list[str]] = {
        task: list(cfg.tasks[task].classes) for task in tasks
    }

    # Data (no tokenizer — JEPA processes raw bytes)
    data_module = JEPAClassificationDataModule(
        cfg=cfg,
        tasks=tasks,
        class_lists=class_lists,
    )
    data_module.setup("fit")
    val_names = data_module.val_dataset_names

    # Model
    model_module = JEPAFinetuneModule(
        cfg=cfg,
        tasks=tasks,
        class_lists=class_lists,
        val_dataset_names=val_names,
    )

    # Callbacks
    callbacks: list[pl.Callback] = [
        LearningRateMonitor(logging_interval="step"),
    ]

    ckpt_cfg = cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        monitor_metric = ckpt_cfg.get(
            "monitor",
            f"val/{val_names[0]}/loss" if val_names else "train/loss",
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_cfg.get("dirpath", "checkpoints/jepa_finetune"),
                filename="step={step}-{" + monitor_metric.replace("/", "_") + ":.4f}",
                monitor=monitor_metric,
                mode=ckpt_cfg.get("mode", "min"),
                save_top_k=ckpt_cfg.get("save_top_k", 3),
                save_last=ckpt_cfg.get("save_last", True),
                every_n_train_steps=cfg.training.get("val_every_n_steps", None),
                auto_insert_metric_name=False,
            )
        )

    # Experiment loggers
    pl_loggers = build_loggers(cfg)

    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        val_check_interval=cfg.training.get("val_every_n_steps", 1.0),
        accumulate_grad_batches=cfg.training.get("grad_accumulation", 1),
        gradient_clip_val=cfg.training.get("gradient_clip_val", None),
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

    logger.info("Starting JEPA finetuning (max_steps=%d)…", cfg.training.max_steps)
    trainer.fit(model_module, datamodule=data_module)
    logger.info("Finetuning complete.")


if __name__ == "__main__":
    main()
