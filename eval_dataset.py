#!/usr/bin/env python3
"""Evaluate a predicted JSONL file against a ground-truth JSONL file.

Computes per-class, per-task, and overall precision / recall / F1.  No
confidence scores are assumed – each predicted example either contains a class
or it does not (same convention as the trained classifier at threshold 0.5).

Usage::

    python eval_dataset.py gt.jsonl pred.jsonl config.yaml
    python eval_dataset.py gt.jsonl pred.jsonl config.yaml \\
        --output-md results.md --output-csv results.csv

Examples are matched by the ``id`` field.  If ``id`` is absent, examples are
matched positionally (first GT row with first prediction row, etc.).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_binary_matrix(
    examples: list[dict[str, Any]],
    task: str,
    classes: list[str],
    id_order: list[str] | None = None,
    example_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, valid_mask) binary arrays for a given task.

    labels:     (N, C) float32
    valid_mask: (N,)   bool – True when the example has GT for this task
    """
    c2i = {cls: i for i, cls in enumerate(classes)}
    n = len(id_order) if id_order else len(examples)
    labels = np.zeros((n, len(classes)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    if id_order is not None and example_map is not None:
        iterator = [(i, example_map.get(eid)) for i, eid in enumerate(id_order)]
    else:
        iterator = [(i, ex) for i, ex in enumerate(examples)]

    for i, ex in iterator:
        if ex is None:
            continue
        gt = ex.get(task)
        if gt is not None:
            for cls in gt.get("classes", []):
                if cls in c2i:
                    labels[i, c2i[cls]] = 1.0
            valid[i] = True

    return labels, valid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_task_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    valid_mask: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Compute per-class and macro precision/recall/F1 for one task.

    Only examples where *both* GT and pred have a valid annotation are used.
    Macro averages skip classes with no positive GT examples (same convention
    as MultiLabelMetrics in the training code).
    """
    if valid_mask.sum() == 0:
        return {"classes": {}, "macro": {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}, "n_samples": 0}

    gt = gt_labels[valid_mask]
    pred = pred_labels[valid_mask]
    n = int(valid_mask.sum())

    per_class: dict[str, dict[str, float]] = {}
    prec_vals, rec_vals, f1_vals = [], [], []

    for i, cls in enumerate(classes):
        t = gt[:, i].astype(int)
        p = pred[:, i].astype(int)
        pr = float(precision_score(t, p, zero_division=0))
        re = float(recall_score(t, p, zero_division=0))
        f1 = float(f1_score(t, p, zero_division=0))
        n_pos = int(t.sum())
        per_class[cls] = {"precision": pr, "recall": re, "f1": f1, "n_pos": n_pos}
        if n_pos > 0:
            prec_vals.append(pr)
            rec_vals.append(re)
            f1_vals.append(f1)

    macro = {
        "precision": float(np.mean(prec_vals)) if prec_vals else float("nan"),
        "recall": float(np.mean(rec_vals)) if rec_vals else float("nan"),
        "f1": float(np.mean(f1_vals)) if f1_vals else float("nan"),
    }
    return {"classes": per_class, "macro": macro, "n_samples": n}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W_CLASS = 38
_W_NUM = 10


def _fmt(v: float) -> str:
    return f"{v:.4f}" if v == v else "  NaN "


def _header_line() -> str:
    return (
        f"{'Class':<{_W_CLASS}} {'Precision':>{_W_NUM}} {'Recall':>{_W_NUM}} {'F1':>{_W_NUM}} {'N_pos':>{_W_NUM}}"
    )


def _sep_line() -> str:
    return "-" * (_W_CLASS + 3 * (_W_NUM + 1) + _W_NUM + 4)


def format_task_stdout(task: str, metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Task: {task}  (n={metrics['n_samples']})")
    lines.append(_sep_line())
    lines.append(_header_line())
    lines.append(_sep_line())
    for cls, m in metrics["classes"].items():
        lines.append(
            f"{cls:<{_W_CLASS}} {_fmt(m['precision']):>{_W_NUM}} {_fmt(m['recall']):>{_W_NUM}} {_fmt(m['f1']):>{_W_NUM}} {m['n_pos']:>{_W_NUM}}"
        )
    lines.append(_sep_line())
    mac = metrics["macro"]
    lines.append(
        f"{'MACRO':<{_W_CLASS}} {_fmt(mac['precision']):>{_W_NUM}} {_fmt(mac['recall']):>{_W_NUM}} {_fmt(mac['f1']):>{_W_NUM}} {'':>{_W_NUM}}"
    )
    return "\n".join(lines)


def format_overall_stdout(
    task_metrics: dict[str, dict[str, Any]],
    overall: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append(f"\n{'='*60}")
    lines.append("OVERALL (macro across tasks)")
    lines.append(_sep_line())
    lines.append(_header_line())
    lines.append(_sep_line())
    for task, metrics in task_metrics.items():
        mac = metrics["macro"]
        lines.append(
            f"{task:<{_W_CLASS}} {_fmt(mac['precision']):>{_W_NUM}} {_fmt(mac['recall']):>{_W_NUM}} {_fmt(mac['f1']):>{_W_NUM}} {'':>{_W_NUM}}"
        )
    lines.append(_sep_line())
    lines.append(
        f"{'TOTAL':<{_W_CLASS}} {_fmt(overall['precision']):>{_W_NUM}} {_fmt(overall['recall']):>{_W_NUM}} {_fmt(overall['f1']):>{_W_NUM}} {'':>{_W_NUM}}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _md_table_header() -> str:
    return (
        "| Class | Precision | Recall | F1 | N_pos |\n"
        "|---|---:|---:|---:|---:|"
    )


def _md_row(name: str, pr: float, re: float, f1: float, n_pos: int | str = "") -> str:
    return f"| {name} | {_fmt(pr)} | {_fmt(re)} | {_fmt(f1)} | {n_pos} |"


def format_task_md(task: str, metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"### {task}")
    lines.append(f"Evaluated on **{metrics['n_samples']}** samples.\n")
    lines.append(_md_table_header())
    for cls, m in metrics["classes"].items():
        lines.append(_md_row(cls, m["precision"], m["recall"], m["f1"], m["n_pos"]))
    mac = metrics["macro"]
    lines.append(_md_row("**MACRO**", mac["precision"], mac["recall"], mac["f1"], ""))
    return "\n".join(lines)


def format_overall_md(
    task_metrics: dict[str, dict[str, Any]],
    overall: dict[str, float],
    gt_path: str,
    pred_path: str,
) -> str:
    lines: list[str] = []
    lines.append("## Summary across all tasks\n")
    lines.append(f"- Ground truth: `{gt_path}`")
    lines.append(f"- Predictions:  `{pred_path}`\n")
    lines.append(_md_table_header())
    for task, metrics in task_metrics.items():
        mac = metrics["macro"]
        lines.append(_md_row(task, mac["precision"], mac["recall"], mac["f1"], ""))
    lines.append(_md_row("**TOTAL**", overall["precision"], overall["recall"], overall["f1"], ""))
    return "\n".join(lines)


def build_markdown(
    task_metrics: dict[str, dict[str, Any]],
    overall: dict[str, float],
    gt_path: str,
    pred_path: str,
) -> str:
    lines: list[str] = []
    lines.append("# Evaluation Results\n")
    lines.append(format_overall_md(task_metrics, overall, gt_path, pred_path))
    lines.append("\n---\n")
    lines.append("## Per-task Results\n")
    for task, metrics in task_metrics.items():
        lines.append(format_task_md(task, metrics))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def build_csv_row(
    task_metrics: dict[str, dict[str, Any]],
    overall: dict[str, float],
    gt_path: str,
    pred_path: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {"gt_file": gt_path, "pred_file": pred_path}
    for task, metrics in task_metrics.items():
        mac = metrics["macro"]
        row[f"{task}/f1"] = _fmt(mac["f1"])
        row[f"{task}/precision"] = _fmt(mac["precision"])
        row[f"{task}/recall"] = _fmt(mac["recall"])
        row[f"{task}/n_samples"] = metrics["n_samples"]
        for cls, m in metrics["classes"].items():
            row[f"{task}/{cls}/f1"] = _fmt(m["f1"])
            row[f"{task}/{cls}/precision"] = _fmt(m["precision"])
            row[f"{task}/{cls}/recall"] = _fmt(m["recall"])
            row[f"{task}/{cls}/n_pos"] = m["n_pos"]
    row["total/f1"] = _fmt(overall["f1"])
    row["total/precision"] = _fmt(overall["precision"])
    row["total/recall"] = _fmt(overall["recall"])
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted JSONL against ground-truth JSONL."
    )
    parser.add_argument("gt", type=Path, help="Ground-truth JSONL file.")
    parser.add_argument("pred", type=Path, help="Predictions JSONL file.")
    parser.add_argument("config", type=Path, help="Training YAML config with tasks section.")
    parser.add_argument("--output-md", type=Path, default=None, metavar="PATH",
                        help="Write results as a Markdown file.")
    parser.add_argument("--output-csv", type=Path, default=None, metavar="PATH",
                        help="Write results as a single-row CSV file.")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    # Load config
    cfg = OmegaConf.load(args.config)
    # Resolve _base_ inheritance if present
    if "_base_" in cfg:
        from text_classification.utils.config import load_config
        cfg = load_config(args.config)
    tasks: list[str] = list(cfg.tasks.keys())
    class_lists: dict[str, list[str]] = {t: list(cfg.tasks[t].classes) for t in tasks}

    # Load data
    gt_examples = load_jsonl(args.gt)
    pred_examples = load_jsonl(args.pred)
    logger.info("Loaded %d GT and %d prediction examples.", len(gt_examples), len(pred_examples))

    # Build ID-keyed maps; fall back to positional matching
    gt_has_ids = all("id" in ex for ex in gt_examples)
    pred_has_ids = all("id" in ex for ex in pred_examples)
    use_id_matching = gt_has_ids and pred_has_ids

    if use_id_matching:
        gt_map = {ex["id"]: ex for ex in gt_examples}
        pred_map = {ex["id"]: ex for ex in pred_examples}
        all_ids = list(gt_map.keys())
        # Warn about IDs present only in one file
        missing_in_pred = set(gt_map) - set(pred_map)
        missing_in_gt = set(pred_map) - set(gt_map)
        if missing_in_pred:
            logger.warning("%d GT ids have no prediction.", len(missing_in_pred))
        if missing_in_gt:
            logger.warning("%d prediction ids have no GT.", len(missing_in_gt))
    else:
        logger.warning("Not all examples have 'id' – using positional matching.")
        n = min(len(gt_examples), len(pred_examples))
        if len(gt_examples) != len(pred_examples):
            logger.warning(
                "GT has %d examples, predictions have %d – using first %d.",
                len(gt_examples), len(pred_examples), n,
            )
        gt_examples = gt_examples[:n]
        pred_examples = pred_examples[:n]
        all_ids = None
        gt_map = pred_map = None

    # Compute metrics per task
    task_metrics: dict[str, dict[str, Any]] = {}
    for task in tasks:
        classes = class_lists[task]
        if use_id_matching:
            gt_labels, gt_valid = build_binary_matrix([], task, classes, all_ids, gt_map)
            pred_labels, pred_valid = build_binary_matrix([], task, classes, all_ids, pred_map)
        else:
            gt_labels, gt_valid = build_binary_matrix(gt_examples, task, classes)
            pred_labels, pred_valid = build_binary_matrix(pred_examples, task, classes)

        # Valid only where both have annotations
        combined_valid = gt_valid & pred_valid
        task_metrics[task] = compute_task_metrics(gt_labels, pred_labels, combined_valid, classes)

    # Overall (macro across tasks, skipping NaN)
    prec_vals = [m["macro"]["precision"] for m in task_metrics.values() if m["macro"]["precision"] == m["macro"]["precision"]]
    rec_vals  = [m["macro"]["recall"]    for m in task_metrics.values() if m["macro"]["recall"]    == m["macro"]["recall"]]
    f1_vals   = [m["macro"]["f1"]        for m in task_metrics.values() if m["macro"]["f1"]        == m["macro"]["f1"]]
    overall = {
        "precision": float(np.mean(prec_vals)) if prec_vals else float("nan"),
        "recall":    float(np.mean(rec_vals))  if rec_vals  else float("nan"),
        "f1":        float(np.mean(f1_vals))   if f1_vals   else float("nan"),
    }

    # stdout
    for task in tasks:
        print(format_task_stdout(task, task_metrics[task]))
    print(format_overall_stdout(task_metrics, overall))

    # Markdown
    if args.output_md is not None:
        md = build_markdown(task_metrics, overall, str(args.gt), str(args.pred))
        args.output_md.write_text(md, encoding="utf-8")
        print(f"\nMarkdown written to {args.output_md}")

    # CSV
    if args.output_csv is not None:
        row = build_csv_row(task_metrics, overall, str(args.gt), str(args.pred))
        write_header = not args.output_csv.exists()
        with open(args.output_csv, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"CSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
