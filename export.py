#!/usr/bin/env python3
"""Export a trained model's HuggingFace encoder to disk.

Loads a Lightning checkpoint and saves:
  - The HuggingFace encoder (``model.encoder``) via ``save_pretrained``
  - The tokenizer used during training
  - The classification heads as a plain ``state_dict`` in ``heads.pt``

Usage::

    python export.py configs/robeczech_base.yaml checkpoints/last.ckpt outputs/my_model
    tc-export configs/robeczech_base.yaml checkpoints/last.ckpt outputs/my_model
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from text_classification.training.lightning_module import TextClassificationModule
from text_classification.utils.config import load_config
from text_classification.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a trained text classifier.")
    parser.add_argument("config", type=Path, help="Training YAML config.")
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt file.")
    parser.add_argument("output_dir", type=Path, help="Directory to write exports.")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Dot-notation config overrides.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.get("log_level", "INFO"))

    tasks: list[str] = list(cfg.tasks.keys())
    class_lists: dict[str, list[str]] = {
        task: list(cfg.tasks[task].classes) for task in tasks
    }

    logger.info("Loading checkpoint: %s", args.checkpoint)
    module = TextClassificationModule.load_from_checkpoint(
        str(args.checkpoint),
        cfg=cfg,
        tasks=tasks,
        class_lists=class_lists,
    )
    module.eval()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Save HuggingFace encoder
    encoder_dir = out / "encoder"
    logger.info("Saving encoder to %s", encoder_dir)
    module.model.encoder.save_pretrained(str(encoder_dir))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(encoder_dir))

    # Save classification heads
    heads_path = out / "heads.pt"
    logger.info("Saving classification heads to %s", heads_path)
    torch.save(module.model.heads.state_dict(), str(heads_path))

    # Save config snapshot
    config_path = out / "config.yaml"
    with open(config_path, "w") as fh:
        fh.write(OmegaConf.to_yaml(cfg))

    logger.info("Export complete → %s", out)


if __name__ == "__main__":
    main()
