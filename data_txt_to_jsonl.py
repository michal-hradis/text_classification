"""
Read large text files where each lines is a document, and write them as JSONL files with ``id`` and ``text`` fields.
Changes:
- Remove <eos>
- Change <br> to \n
- Remove `^TITLE: '
"""

import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert text files to JSONL format")
    parser.add_argument("input_dir", type=Path, help="Directory with input text files")
    parser.add_argument("output_dir", type=Path, help="Directory to save output jsonl files")
    parser.add_argument("--glob", type=str, default="*.txt", help="Glob pattern to match input files")
    return parser.parse_args()


def process_line(line: str) -> str:
    line = line.replace(" <eos>", "")
    line = line.replace(" <br> ", "\n")
    if line.startswith("^TITLE: "):
        line = line[len("^TITLE: "):]
    return line.strip()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(args.input_dir.glob(args.glob))
    if not input_files:
        logger.warning(f"No files found in {args.input_dir} matching {args.glob}")
        return

    for input_file in tqdm(input_files, desc="Processing files"):
        output_file = args.output_dir / f"{input_file.stem}.jsonl"
        with input_file.open("r", encoding="utf-8") as f, output_file.open("w", encoding="utf-8") as out_f:
            for lineno, line in tqdm(enumerate(f), desc=f"Processing {input_file.name}", total=sum(1 for _ in f)):
                line = process_line(line)
                if not line:
                    continue
                document_id = f"{input_file.stem}_{lineno}"
                json.dump({"id": document_id, "text": line}, out_f)
                out_f.write("\n")

