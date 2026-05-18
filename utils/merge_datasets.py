#!/usr/bin/env python3
"""Merge multiple COCO-format player detection datasets into one.

Usage:
    python utils/merge_datasets.py \
        --output data/input/train/player_merged \
        data/input/train/player \
        data/input/train2

    # Or with explicit annotation files:
    python utils/merge_datasets.py \
        --output data/input/train/player_merged \
        --annotations data/input/train/player/_annotations.coco.json \
        --annotations data/input/train2/_annotations.coco.json \
        data/input/train/player \
        data/input/train2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List

# Import the merge function from data_download.py
from data_download import merge_coco_datasets, _dir_has_files


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple COCO-format datasets into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source_dirs",
        nargs="+",
        help="Source directories containing COCO datasets (images + _annotations.coco.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--annotations",
        action="append",
        dest="annotations",
        help="Explicit COCO annotation file paths (optional, auto-detected if not provided)",
    )
    args = parser.parse_args()

    source_dirs = [os.path.abspath(d) for d in args.source_dirs]
    output_dir = os.path.abspath(args.output)

    print(f"[MERGE] Merging {len(source_dirs)} datasets")
    for i, d in enumerate(source_dirs, 1):
        n_imgs = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"  [{i}] {d}  ({n_imgs} images)")

    # Sanity checks
    for d in source_dirs:
        if not os.path.isdir(d):
            print(f"[ERROR] Directory not found: {d}")
            return False
        if not _dir_has_files(d):
            print(f"[ERROR] Directory is empty: {d}")
            return False

    print(f"\n[MERGE] Output: {output_dir}\n")

    # Call merge function from data_download.py
    merge_coco_datasets(source_dirs, output_dir)

    # Verify output
    out_json = os.path.join(output_dir, "_annotations.coco.json")
    if os.path.isfile(out_json):
        with open(out_json) as f:
            data = json.load(f)
        n_imgs = len(data.get("images", []))
        n_anns = len(data.get("annotations", []))
        cats = [c["name"] for c in data.get("categories", [])]
        print(f"\n[MERGE] ✓ Success!")
        print(f"  Total images:       {n_imgs}")
        print(f"  Total annotations:  {n_anns}")
        print(f"  Categories:         {cats}")
        print(f"  Output JSON:        {out_json}")
        return True
    else:
        print(f"[ERROR] Merge failed — no output JSON found")
        return False


if __name__ == "__main__":
    ok = main()
    exit(0 if ok else 1)
