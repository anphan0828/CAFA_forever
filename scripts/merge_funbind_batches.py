#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and merge FunBind batch outputs into one TSV."
    )
    parser.add_argument(
        "--batch-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Directory containing funbind_batch_*.tsv and matching manifest files.",
    )
    parser.add_argument(
        "--expected-batch-count",
        type=int,
        required=True,
        help="Expected number of batch prediction files.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output path for merged FunBind predictions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    batch_dir = args.batch_dir.resolve()
    output_path = args.output
    expected_batch_count = args.expected_batch_count

    batch_pattern = re.compile(r"^funbind_batch_(\d{3})\.tsv$")
    manifest_pattern = re.compile(r"^funbind_batch_(\d{3})\.manifest\.tsv$")

    batch_files = []
    manifest_files = []
    for path in sorted(batch_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if manifest_pattern.match(path.name):
            manifest_files.append(path)
            continue
        if batch_pattern.match(path.name):
            batch_files.append(path)

    if not batch_files:
        raise SystemExit("No FunBind batch prediction files were produced")

    if len(batch_files) != len(manifest_files):
        raise SystemExit(
            f"Mismatch between FunBind batch predictions ({len(batch_files)}) and manifests ({len(manifest_files)})"
        )

    batch_indexes = []
    for path in batch_files:
        match = batch_pattern.match(path.name)
        if not match:
            raise SystemExit(f"Unexpected FunBind batch file name: {path.name}")
        batch_indexes.append(int(match.group(1)))

    manifest_indexes = []
    for path in manifest_files:
        match = manifest_pattern.match(path.name)
        if not match:
            raise SystemExit(f"Unexpected FunBind batch manifest name: {path.name}")
        manifest_indexes.append(int(match.group(1)))

    if batch_indexes != manifest_indexes:
        raise SystemExit("FunBind batch predictions and manifests are not aligned")

    if len(batch_files) != expected_batch_count:
        raise SystemExit(
            f"Expected {expected_batch_count} FunBind batch files but found {len(batch_files)}"
        )

    expected_indexes = list(range(1, expected_batch_count + 1))
    if batch_indexes != expected_indexes:
        raise SystemExit(
            f"Missing FunBind batch indexes: expected {expected_indexes}, observed {batch_indexes}"
        )

    part_names = set()
    for manifest_path in manifest_files:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry:
                continue
            if entry in part_names:
                raise SystemExit(
                    f"Duplicate FunBind split FASTA assignment detected: {entry}"
                )
            part_names.add(entry)

    if not part_names:
        raise SystemExit("No FunBind split FASTA assignments were recorded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_handle:
        for batch_path in batch_files:
            with batch_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
