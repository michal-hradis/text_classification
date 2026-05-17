#!/usr/bin/env python3
"""Concatenate classifier-training JSONL segment files into per-document JSONL.

Each input JSONL file represents a single document: every line is a JSON object
with at least the fields ``document`` (UUID used as ``id``), ``order`` (int),
and ``text`` (str).  The objects are sorted by ``order`` and their ``text``
values are concatenated (space-separated) to produce one output line per file.

Files are processed one at a time so the full dataset is never held in memory.

Usage::

    python concat_document_segments.py input_dir/ output.jsonl
    python concat_document_segments.py input_dir/ output.jsonl --separator "\\n"
    python concat_document_segments.py input_dir/ output.jsonl --glob "*.jsonl"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def process_file(path: Path, separator: str) -> dict:
    """Read one JSONL file, sort by ``order``, concatenate ``text``.

    Returns a dict with ``id`` and ``text``.
    Raises ``ValueError`` if the file is empty or missing required fields.
    """
    segments: list[tuple[int, str]] = []
    document_id: str | None = None

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc

            # Validate required fields
            for field in ("document", "order", "text"):
                if field not in obj:
                    raise ValueError(
                        f"{path}:{lineno}: missing required field {field!r}"
                    )

            doc_id = obj["document"]
            if document_id is None:
                document_id = doc_id
            elif doc_id != document_id:
                logger.warning(
                    "%s:%d: document id %r does not match first id %r — skipping line",
                    path,
                    lineno,
                    doc_id,
                    document_id,
                )
                continue

            try:
                order = int(obj["order"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: 'order' must be an integer, got {obj['order']!r}"
                ) from exc

            segments.append((order, str(obj["text"])))

    if not segments:
        raise ValueError(f"{path}: file is empty or contains no valid lines")

    segments.sort(key=lambda t: t[0])
    return {"id": document_id, "text": separator.join(text for _, text in segments)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate per-segment JSONL files into per-document JSONL."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input JSONL files.")
    parser.add_argument("output", type=Path, help="Output JSONL file path.")
    parser.add_argument(
        "--separator",
        default=" ",
        help="String inserted between consecutive segment texts (default: space).",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob pattern for input files inside input_dir (default: '*.jsonl').",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"input_dir {args.input_dir!r} is not a directory")

    input_files = sorted(args.input_dir.glob(args.glob))
    if not input_files:
        logger.warning("No files matching %r found in %s", args.glob, args.input_dir)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    errors = 0

    with args.output.open("w", encoding="utf-8") as out_fh:
        for path in input_files:
            try:
                doc = process_file(path, args.separator)
            except ValueError as exc:
                logger.error("%s", exc)
                errors += 1
                continue

            out_fh.write(json.dumps(doc, ensure_ascii=False))
            out_fh.write("\n")
            written += 1

    logger.info("Wrote %d documents to %s (%d errors)", written, args.output, errors)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
