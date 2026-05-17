"""
This scrip reads JEPA dataset - jsonl files with documents - {"id": "doc_uuid", "text": "document text"}
It splits the documents into segments of specified length - and saves them as jsonl files with {"id": f"doc_uuid-{segment_index}", "text": "segment text"}.
"""

import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Split documents into segments")
    parser.add_argument("input_dir", type=Path, help="Directory with input jsonl files")
    parser.add_argument("output_dir", type=Path, help="Directory to save output jsonl files")
    parser.add_argument("--segment_length", type=int, default=10000, help="Length of each segment in characters")
    parser.add_argument("--glob", type=str, default="*.jsonl", help="Glob pattern to match input files")
    return parser.parse_args()


def split_document(document: dict, segment_length: int) -> list[dict]:
    text = document["text"]
    segments = []
    for i in range(0, len(text), segment_length):
        segment_text = text[i:i+segment_length]
        segment_id = f"{document['id']}-{i//segment_length}"
        segments.append({"id": segment_id, "text": segment_text})
    return segments


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(args.input_dir.glob(args.glob))
    if not input_files:
        logger.warning(f"No files found in {args.input_dir} matching {args.glob}")
        return

    for input_file in tqdm(input_files, desc="Processing files"):
        with input_file.open("r", encoding="utf-8") as f:
            output_file = args.output_dir / input_file.name
            with output_file.open("w", encoding="utf-8") as out_f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.error(f"Invalid JSON in {input_file}: {exc}")
                        continue

                    segments = split_document(document, args.segment_length)
                    for segment in segments:
                        output_file = args.output_dir / f"{segment['id']}.jsonl"
                        json.dump(segment, out_f)
                        out_f.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
