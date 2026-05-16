"""Dataset download utilities.

Usage:
    python utils/data_download.py --roboflow [--api-key KEY]
    python utils/data_download.py --finebadminton [--hf-dir /oscar/scratch/$USER/finebadminton20k]
    python utils/data_download.py --all

Roboflow (player + shuttle detection, ~1 GB):
    Downloads badminton-hehp8 dataset (COCO format) to data/input/train/.
    Uses roboflow package if API key is provided; otherwise prints manual instructions.
    API key: the *private* key from your Roboflow account → Settings → API Keys.

FineBadminton20k (stroke classification, ~43 GB):
    Source: HuggingFace — search for "finebadminton" or "FineBadminton20K" on
            https://huggingface.co/datasets
    --hf-dir sets the local download target (default: data/input/finebadminton).
    On OSCAR this should be a /scratch path to avoid quota issues.

Both: skip if target directory already has files (idempotent).
"""

from __future__ import annotations

import argparse
import os
import sys

ROBOFLOW_WORKSPACE = "badminton-rojkf"
ROBOFLOW_PROJECT = "badminton-hehp8"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "coco"
ROBOFLOW_TRAIN_DIR = "data/input/train"

FINEBADMINTON_DIR    = "data/input/finebadminton"
FINEBADMINTON_HF_ID  = "ILEARN-Lab/FineBadminton20K"   # confirm on HuggingFace before use
FINEBADMINTON_URL    = "https://huggingface.co/datasets/ILEARN-Lab/FineBadminton20K"


def _dir_has_files(directory: str) -> bool:
    """Return True if directory exists and contains at least one file."""
    if not os.path.isdir(directory):
        return False
    for _, _, files in os.walk(directory):
        if files:
            return True
    return False


def download_roboflow(api_key: str | None = None) -> None:
    """Download Roboflow badminton-hehp8 dataset (COCO format)."""
    target = ROBOFLOW_TRAIN_DIR

    if _dir_has_files(target):
        print(f"[DOWNLOAD] Roboflow dataset already present at '{target}' — skipping.")
        return

    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if api_key:
        try:
            from roboflow import Roboflow

            os.makedirs(target, exist_ok=True)
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
            dataset = project.version(ROBOFLOW_VERSION).download(
                ROBOFLOW_FORMAT,
                location=target,
                overwrite=False,
            )
            print(f"[DOWNLOAD] Roboflow dataset downloaded to: {target}")
        except ImportError:
            print("[DOWNLOAD] 'roboflow' package not installed. Install with: pip install roboflow")
            _print_roboflow_manual()
        except Exception as e:
            print(f"[DOWNLOAD] Roboflow download failed: {e}")
            _print_roboflow_manual()
    else:
        print("[DOWNLOAD] No API key provided.")
        _print_roboflow_manual()


def _print_roboflow_manual() -> None:
    print(
        "\n[DOWNLOAD] Manual Roboflow download instructions:\n"
        "  1. Visit: https://universe.roboflow.com/badminton-rojkf/badminton-hehp8\n"
        "  2. Click 'Download Dataset' → select 'COCO' format.\n"
        f"  3. Extract the downloaded ZIP into: {os.path.abspath(ROBOFLOW_TRAIN_DIR)}/\n"
        "     Expected layout:\n"
        f"       {ROBOFLOW_TRAIN_DIR}/_annotations.coco.json\n"
        f"       {ROBOFLOW_TRAIN_DIR}/<image_files>.jpg\n"
    )


def download_finebadminton(hf_dir: str | None = None) -> None:
    """Download FineBadminton20k from HuggingFace, or print instructions.

    Args:
        hf_dir: Local directory to save the dataset.
                On OSCAR, pass /oscar/scratch/$USER/finebadminton20k.
                Default: data/input/finebadminton (fine for small machines).
    """
    target = hf_dir or FINEBADMINTON_DIR

    if _dir_has_files(target):
        print(f"[DOWNLOAD] FineBadminton dataset already present at '{target}' — skipping.")
        return

    print(f"\n[DOWNLOAD] Attempting HuggingFace download → {target}")
    print(f"           Dataset ID: {FINEBADMINTON_HF_ID}")

    # Try programmatic download via huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(target, exist_ok=True)
        snapshot_download(
            repo_id=FINEBADMINTON_HF_ID,
            repo_type="dataset",
            local_dir=target,
            local_dir_use_symlinks=False,
        )
        print(f"[DOWNLOAD] FineBadminton20k downloaded to: {target}")
        return
    except ImportError:
        pass   # huggingface_hub not installed — fall through to instructions
    except Exception as e:
        print(f"[DOWNLOAD] huggingface_hub download failed ({e})")

    # Fall back: print CLI instructions
    print(
        "\n[DOWNLOAD] FineBadminton20k — manual download (43 GB):\n"
        "\n"
        "  Option A — huggingface-cli (recommended on OSCAR):\n"
        f"    pip install huggingface_hub\n"
        f"    huggingface-cli download {FINEBADMINTON_HF_ID} \\\n"
        f"        --repo-type dataset \\\n"
        f"        --local-dir {target}\n"
        "\n"
        "  Option B — Python:\n"
        "    from huggingface_hub import snapshot_download\n"
        f"   snapshot_download(repo_id='{FINEBADMINTON_HF_ID}',\n"
        f"                     repo_type='dataset', local_dir='{target}')\n"
        "\n"
        "  Option C — datasets library (downloads + converts to Arrow format):\n"
        "    from datasets import load_dataset\n"
        f"   ds = load_dataset('{FINEBADMINTON_HF_ID}', cache_dir='{target}')\n"
        f"   ds.save_to_disk('{target}')   # saves as load_from_disk-compatible format\n"
        "\n"
        "  NOTE: Confirm the exact dataset ID at:\n"
        f"        {FINEBADMINTON_URL}\n"
        "\n"
        "  After download:\n"
        f"    python training/train_stroke.py --data-dir {target}\n"
        "  Or on OSCAR via SLURM:\n"
        f"    sbatch --export=MODE=train-stroke,FINEBADMINTON_DIR={target} slurm_train.sh\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download badminton training datasets."
    )
    parser.add_argument(
        "--roboflow",
        action="store_true",
        help="Download Roboflow badminton-hehp8 dataset.",
    )
    parser.add_argument(
        "--finebadminton",
        action="store_true",
        help="Download FineBadminton20k from HuggingFace (or print instructions).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both --roboflow and --finebadminton.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Roboflow private API key (or set ROBOFLOW_API_KEY env var).",
    )
    parser.add_argument(
        "--hf-dir",
        dest="hf_dir",
        default=None,
        help=(
            "Local directory for FineBadminton20k download. "
            "On OSCAR use /oscar/scratch/$USER/finebadminton20k to avoid quota issues. "
            "Default: data/input/finebadminton"
        ),
    )
    args = parser.parse_args()

    if not (args.roboflow or args.finebadminton or args.all):
        parser.print_help()
        sys.exit(1)

    if args.roboflow or args.all:
        download_roboflow(api_key=args.api_key)

    if args.finebadminton or args.all:
        download_finebadminton(hf_dir=args.hf_dir)


if __name__ == "__main__":
    main()
