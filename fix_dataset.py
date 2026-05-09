#!/usr/bin/env python3
"""Validate and fix JSONL dataset files for text classification.

Reads one or more JSONL files (file or directory input), applies the following
corrections to each task field, writes fixed files to the output path, and
prints a summary of all changes.

Fixes applied per task field in each example:
  1. Task value is a list  → treated as the .classes list; wrapped into
                             {"classes": [...], "reason": ""}.
  2. Task value is a string → treated as a single class name; wrapped into
                             {"classes": [value], "reason": ""}.
  3. Task value is a dict but contains .class field instead of .classes → .classes set to [.class]
  4. Task value is a dict but .classes is not a list → .classes set to [].
  5. Unknown class names inside .classes → deleted; counted per (task, class).
  6. (optional) Keys whose name contains the substring "thinking" are removed.

All other task values (None, well-formed dict) are left unchanged.

Usage::

    python fix_dataset.py data.jsonl fixed.jsonl configs/base.yaml
    python fix_dataset.py data_dir/ fixed_dir/ configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_tasks(config_path: Path) -> dict[str, list[str]]:
    """Return {task_name: [class, ...]} from a YAML config (with _base_ support)."""
    from text_classification.utils.config import load_config
    cfg = load_config(config_path)
    return {t: list(cfg.tasks[t].classes) for t in cfg.tasks}


# ---------------------------------------------------------------------------
# Per-example fixing
# ---------------------------------------------------------------------------

# Counts: task -> kind -> count
# kinds: "list_wrapped", "string_wrapped", "class_field_renamed", "bad_classes_type", "unknown_class:<name>"
Counts = dict[str, dict[str, int]]

# ClassCounts: task -> class_name -> count of appearances in final output
ClassCounts = dict[str, dict[str, int]]



def rename_class_string(value: str) -> str:
    """Apply renaming rules to class names to fix common typos and inconsistencies."""

    # LLMs tend to misspell the topic categorids starting with "ddc_" - ass "ddd_" od "dd_"
    # Resplace these only at the start of the scring
    if value.startswith("ddd_"):
        value = "ddc_" + value[4:]
    elif value.startswith("dd_"):
        value = "ddc_" + value[3:]
    return value


def _inc(counts: Counts, task: str, key: str) -> None:
    counts[task][key] = counts[task].get(key, 0) + 1


def fix_example(
    example: dict[str, Any],
    tasks: dict[str, list[str]],
    counts: Counts,
    remove_thinking: bool = False,
) -> dict[str, Any]:
    """Return a (possibly modified) copy of *example*."""
    result = dict(example)

    # ---- Fix 6: remove keys containing "thinking" --------------------------
    if remove_thinking:
        thinking_keys = [k for k in result if "thinking" in k]
        for k in thinking_keys:
            del result[k]
            _inc(counts, "__global__", f"thinking_key_removed:{k}")

    for task, valid_classes in tasks.items():
        if task not in result:
            continue  # not annotated – fine

        value = result[task]

        if value is None:
            continue  # explicitly absent – fine

        # ---- Fix 1 & 2: wrong container type ---------------------------------
        if isinstance(value, list):
            _inc(counts, task, "list_wrapped")
            value = {"classes": value, "reason": ""}

        elif isinstance(value, str):
            _inc(counts, task, "string_wrapped")
            value = {"classes": [value], "reason": ""}

        elif not isinstance(value, dict):
            # Some other scalar – drop the whole field
            _inc(counts, task, "non_dict_dropped")
            result[task] = None
            continue

        # value is now guaranteed to be a dict
        # ---- Fix 3: .class field instead of .classes ------------------------
        if "class" in value and "classes" not in value:
            _inc(counts, task, "class_field_renamed")
            value = dict(value)
            value["classes"] = [value.pop("class")]

        # ---- Fix 4: .classes not a list --------------------------------------
        if not isinstance(value.get("classes"), list):
            _inc(counts, task, "bad_classes_type")
            value = dict(value)
            value["classes"] = []

        # ---- Fix 4: unknown class names --------------------------------------
        valid_set = set(valid_classes)
        cleaned: list[str] = []
        for cls in value["classes"]:
            cls = rename_class_string(cls)
            if cls in valid_set:
                cleaned.append(cls)
            else:
                _inc(counts, task, f"unknown_class:{cls}")
        value = dict(value)
        value["classes"] = cleaned

        result[task] = value

    return result


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(
    src: Path,
    dst: Path,
    tasks: dict[str, list[str]],
    counts: Counts,
    class_counts: ClassCounts,
    remove_thinking: bool = False,
) -> tuple[int, int]:
    """Process *src* → *dst*.  Returns (total_examples, changed_examples)."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for lineno, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                example = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("%s line %d: JSON parse error – skipped (%s)", src, lineno, exc)
                continue

            fixed = fix_example(example, tasks, counts, remove_thinking=remove_thinking)
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            total += 1
            if fixed != example:
                changed += 1
            for task in tasks:
                val = fixed.get(task)
                if val is not None and isinstance(val, dict) and isinstance(val.get("classes"), list):
                    for cls in val["classes"]:
                        class_counts[task][cls] = class_counts[task].get(cls, 0) + 1

    return total, changed


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(counts: Counts, class_counts: ClassCounts, file_stats: list[tuple[Path, Path, int, int]]) -> None:
    print("\n" + "=" * 70)
    print("FILE SUMMARY")
    print("=" * 70)
    total_ex = total_changed = 0
    for src, dst, n, ch in file_stats:
        print(f"  {src}  →  {dst}")
        print(f"    examples: {n}  changed: {ch}")
        total_ex += n
        total_changed += ch
    print(f"\n  TOTAL: {total_ex} examples, {total_changed} changed")

    # Show thinking-key removals (global, not per-task)
    global_counts = counts.get("__global__", {})
    thinking_removed = {k[len("thinking_key_removed:"):]: v for k, v in global_counts.items() if k.startswith("thinking_key_removed:")}
    if thinking_removed:
        print(f"\n  Thinking keys removed ({sum(thinking_removed.values())} total):")
        for key, cnt in sorted(thinking_removed.items(), key=lambda kv: -kv[1]):
            print(f"    {key!r}: {cnt}")

    task_counts_nonempty = {t: v for t, v in counts.items() if t != "__global__" and v}
    if not task_counts_nonempty and not thinking_removed:
        print("\nNo corrections needed.")
        return
    if not task_counts_nonempty:
        print()
        return

    print("\n" + "=" * 70)
    print("CORRECTIONS BY TASK")
    print("=" * 70)
    for task in sorted(t for t in counts if t != "__global__"):
        task_counts = counts[task]
        if not task_counts:
            continue
        print(f"\n  {task}")

        # Structural fixes first
        for key in ("list_wrapped", "string_wrapped", "class_field_renamed", "bad_classes_type", "non_dict_dropped"):
            if key in task_counts:
                label = {
                    "list_wrapped":       "list → wrapped as object",
                    "string_wrapped":     "string → wrapped as single-class object",
                    "class_field_renamed": ".class → renamed to .classes",
                    "bad_classes_type":   ".classes not a list → reset to []",
                    "non_dict_dropped":   "non-dict value → set to null",
                }[key]
                print(f"    {label}: {task_counts[key]}")

        # Unknown classes
        unknown = {
            k[len("unknown_class:"):]: v
            for k, v in task_counts.items()
            if k.startswith("unknown_class:")
        }
        if unknown:
            print(f"    Unknown classes removed ({sum(unknown.values())} total):")
            for cls, cnt in sorted(unknown.items(), key=lambda kv: -kv[1]):
                print(f"      {cls!r}: {cnt}")

    print()

    if any(class_counts.values()):
        print("=" * 70)
        print("CLASS APPEARANCE COUNTS (in final output)")
        print("=" * 70)
        for task in sorted(class_counts):
            task_cls = class_counts[task]
            if not task_cls:
                continue
            total_cls = sum(task_cls.values())
            print(f"\n  {task}  (total labelled: {total_cls})")
            for cls, cnt in sorted(task_cls.items(), key=lambda kv: -kv[1]):
                print(f"    {cls}: {cnt}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate and fix JSONL dataset files.",
    )
    parser.add_argument("input", type=Path,
                        help="Input JSONL file or directory of JSONL files.")
    parser.add_argument("output", type=Path,
                        help="Output file (if input is a file) or directory.")
    parser.add_argument("config", type=Path,
                        help="Training YAML config with tasks section.")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--remove-thinking", action="store_true",
                        help="Remove all keys whose name contains 'thinking'.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    tasks = load_tasks(args.config)
    print(f"Tasks loaded from config: {', '.join(tasks)}")

    # Collect (src, dst) pairs
    pairs: list[tuple[Path, Path]] = []
    if args.input.is_dir():
        src_files = sorted(args.input.glob("*.jsonl"))
        if not src_files:
            print(f"No .jsonl files found in {args.input}", file=sys.stderr)
            sys.exit(1)
        for src in src_files:
            pairs.append((src, args.output / src.name))
    elif args.input.is_file():
        dst = args.output if args.output.suffix else args.output / args.input.name
        pairs.append((args.input, dst))
    else:
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    counts: Counts = defaultdict(dict)
    class_counts: ClassCounts = defaultdict(dict)
    file_stats: list[tuple[Path, Path, int, int]] = []

    for src, dst in pairs:
        total, changed = process_file(src, dst, tasks, counts, class_counts, remove_thinking=args.remove_thinking)
        file_stats.append((src, dst, total, changed))
        logger.info("Processed %s → %s (%d examples, %d changed)", src, dst, total, changed)

    print_summary(dict(counts), dict(class_counts), file_stats)


if __name__ == "__main__":
    main()
