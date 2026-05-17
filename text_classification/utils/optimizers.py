"""Shared optimizer and learning-rate scheduler factory.

Both the BERT-based and LLM-based Lightning modules delegate to these
helpers so scheduler behaviour stays consistent across pipelines.

Usage::

    optimizer = build_optimizer(model.named_parameters(), cfg.optimizer)
    sched_dict = build_scheduler(optimizer, cfg.scheduler, max_steps=trainer.max_steps)
    # sched_dict is ready to be returned from configure_optimizers
"""
from __future__ import annotations

from typing import Any, Iterator

import torch
from omegaconf import DictConfig

# Parameter names that should not receive weight decay
_NO_DECAY = {"bias", "LayerNorm.weight", "layer_norm.weight"}


def build_optimizer(
    named_params: Iterator[tuple[str, torch.nn.Parameter]],
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer with weight-decay group splitting.

    Parameters whose names contain any of ``bias``, ``LayerNorm.weight``, or
    ``layer_norm.weight`` are placed in a zero-weight-decay group; all others
    receive ``cfg.weight_decay``.

    Args:
        named_params: Iterator of ``(name, param)`` pairs, typically from
            ``model.named_parameters()``.
        cfg: Optimizer config node. Required key: ``lr``. Optional keys:
            ``weight_decay`` (default 0.01), ``betas`` (default [0.9, 0.999]),
            ``eps`` (default 1e-8).

    Returns:
        Configured :class:`torch.optim.AdamW` optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.get("weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=cfg.lr,
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
        eps=cfg.get("eps", 1e-8),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    max_steps: int = 10_000,
) -> dict[str, Any]:
    """Build a LR scheduler and return a Lightning-compatible dict.

    Args:
        optimizer: The optimizer to attach the scheduler to.
        cfg: Scheduler config node. Required key: ``name``. Available
            schedulers and their extra keys:

            - ``cosine``: ``T_max`` (default ``max_steps``), ``eta_min``
            - ``linear_warmup_cosine``: ``warmup_steps``, ``total_steps``
            - ``linear_warmup``: ``warmup_steps``, ``total_steps``
            - ``constant_warmup``: ``warmup_steps``
            - ``reduce_on_plateau``: ``mode``, ``factor``, ``patience``
            - ``one_cycle``: ``total_steps``, ``max_lr``, ``pct_start``

        max_steps: Fallback total-step count when the config doesn't specify
            ``T_max`` / ``total_steps``.

    Returns:
        Dict with keys ``scheduler``, ``interval``, ``frequency``, and
        ``monitor`` — suitable for direct return from
        ``configure_optimizers``.
    """
    from torch.optim import lr_scheduler

    name: str = cfg.name

    if name == "cosine":
        total = cfg.get("T_max", max_steps)
        sched = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total, eta_min=cfg.get("eta_min", 0.0)
        )

    elif name == "linear_warmup_cosine":
        from transformers import get_cosine_schedule_with_warmup

        sched = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.get("total_steps", max_steps),
        )

    elif name == "linear_warmup":
        from transformers import get_linear_schedule_with_warmup

        sched = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.get("total_steps", max_steps),
        )

    elif name == "constant_warmup":
        from transformers import get_constant_schedule_with_warmup

        sched = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.warmup_steps
        )

    elif name == "reduce_on_plateau":
        sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.get("mode", "min"),
            factor=cfg.get("factor", 0.5),
            patience=cfg.get("patience", 5),
        )

    elif name == "one_cycle":
        total = cfg.get("total_steps", max_steps)
        sched = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.get("max_lr", optimizer.param_groups[0]["lr"]),
            total_steps=total,
            pct_start=cfg.get("pct_start", 0.3),
        )

    else:
        raise ValueError(f"Unknown scheduler: {name!r}")

    return {
        "scheduler": sched,
        "interval": cfg.get("interval", "step"),
        "frequency": cfg.get("frequency", 1),
        "monitor": cfg.get("monitor", "val/loss"),
    }
