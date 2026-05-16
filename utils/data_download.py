"""Dataset download utilities.

Usage:
    python utils/data_download.py --roboflow [--api-key KEY] [--rf-workspace WS] [--rf-project PROJ]
    python utils/data_download.py --finebadminton [--hf-dir /oscar/scratch/$USER/finebadminton20k]
    python utils/data_download.py --all

Roboflow (player + shuttle detection):
    Downloads a badminton detection dataset (COCO format) to data/input/train/.
    Uses roboflow package if API key is provided; otherwise prints manual instructions.
    API key: the *private* key from your Roboflow account → Settings → API Keys.

    Finding a working Roboflow dataset
    -----------------------------------
    The default workspace/project slug is a placeholder. To find a real dataset:
      1. Go to https://universe.roboflow.com
      2. Search "badminton" — filter by Object Detection, publicly available
      3. Confirm a "Download Dataset" button exists on the project page
      4. Copy the workspace and project slugs from the URL:
           https://universe.roboflow.com/<workspace>/<project>
      5. Pass them: --rf-workspace <workspace> --rf-project <project>

    Known-good public alternatives (verify availability before use):
      • badminton-shuttlecock  (shuttlecock only)
          workspace: kj-wkp2y   project: shuttlecock-detection-cqp8u
      • badminton-player-detection (players + shuttle)
          Search "badminton player" on Universe; pick one with a Download button.

    If a project has no Download button it is either private or has no published
    version — you cannot download it via the API.

FineBadminton20k (stroke classification, ~43 GB):
    Source: HuggingFace — https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K
    --hf-dir sets the local download target (default: data/input/finebadminton).
    On OSCAR this should be a /scratch path to avoid quota issues.

Both: skip if target directory already has files (idempotent).
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Roboflow defaults ─────────────────────────────────────────────────────────
# These are placeholders.  Override at runtime with --rf-workspace / --rf-project
# (or find the correct slugs on https://universe.roboflow.com).
ROBOFLOW_WORKSPACE = "badminton-rojkf"   # PLACEHOLDER — replace with a real slug
ROBOFLOW_PROJECT   = "badminton-hehp8"   # PLACEHOLDER — replace with a real slug
ROBOFLOW_FORMAT    = "coco"
ROBOFLOW_TRAIN_DIR = "data/input/train"
# Version is discovered automatically at runtime — do not hardcode.

FINEBADMINTON_DIR    = "data/input/finebadminton"
FINEBADMINTON_HF_ID  = "iLearn-Lab/Finebadminton-20K"   # confirm on HuggingFace before use
FINEBADMINTON_URL    = "https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K"


def _dir_has_files(directory: str) -> bool:
    """Return True if directory exists and contains at least one file."""
    if not os.path.isdir(directory):
        return False
    for _, _, files in os.walk(directory):
        if files:
            return True
    return False


def _get_latest_version(project) -> int:
    """Return the highest available version number for a Roboflow project.

    The SDK exposes project.versions() which returns a list of version objects,
    each with a .version attribute (int).  If that call fails for any reason,
    we fall back to probing versions 1-10 sequentially.
    """
    # Preferred: ask the SDK for all versions and take the max
    try:
        versions = project.versions()
        if versions:
            return max(int(v.version) for v in versions)
    except Exception:
        pass

    # Fallback: probe version numbers until we hit one that works
    for v in range(1, 11):
        try:
            project.version(v)   # raises if version doesn't exist
            return v
        except Exception:
            continue

    raise RuntimeError(
        f"No published versions found for {ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}. "
        "Check the project URL in Roboflow Universe."
    )


def download_roboflow(
    api_key: str | None = None,
    workspace: str | None = None,
    project_slug: str | None = None,
) -> None:
    """Download the latest published version of a Roboflow badminton dataset (COCO format).

    workspace and project_slug override the module-level defaults.
    Version number is discovered at runtime.
    """
    ws   = workspace     or ROBOFLOW_WORKSPACE
    proj = project_slug  or ROBOFLOW_PROJECT
    target = ROBOFLOW_TRAIN_DIR

    if _dir_has_files(target):
        print(f"[DOWNLOAD] Roboflow dataset already present at '{target}' — skipping.")
        return

    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    print(f"[DOWNLOAD] Roboflow workspace='{ws}'  project='{proj}'")

    if api_key:
        try:
            from roboflow import Roboflow

            rf      = Roboflow(api_key=api_key)
            project = rf.workspace(ws).project(proj)

            version_num = _get_latest_version(project)
            print(f"[DOWNLOAD] Using Roboflow version {version_num}")

            os.makedirs(target, exist_ok=True)
            project.version(version_num).download(
                ROBOFLOW_FORMAT,
                location=target,
                overwrite=False,
            )
            print(f"[DOWNLOAD] Roboflow dataset downloaded to: {target}")
        except ImportError:
            print("[DOWNLOAD] 'roboflow' package not installed.  pip install roboflow")
            _print_roboflow_manual(ws, proj)
        except Exception as e:
            print(f"[DOWNLOAD] Roboflow download failed: {e}")
            _print_roboflow_manual(ws, proj)
    else:
        print("[DOWNLOAD] No API key provided.")
        _print_roboflow_manual(ws, proj)


def _print_roboflow_manual(workspace: str = ROBOFLOW_WORKSPACE,
                            project: str = ROBOFLOW_PROJECT) -> None:
    url = f"https://universe.roboflow.com/{workspace}/{project}"
    print(
        f"\n[DOWNLOAD] Manual Roboflow download instructions:\n"
        f"  1. Confirm a working dataset exists — visit:\n"
        f"       {url}\n"
        f"     If there is NO 'Download Dataset' button, this project is private or\n"
        f"     has no published version.  Search https://universe.roboflow.com for\n"
        f"     'badminton' (filter: Object Detection, public) and find one that does.\n"
        f"     Then re-run with:\n"
        f"       python utils/data_download.py --roboflow \\\n"
        f"           --api-key YOUR_KEY \\\n"
        f"           --rf-workspace <workspace> \\\n"
        f"           --rf-project  <project>\n"
        f"\n"
        f"  2. Or download manually:\n"
        f"     a. Click 'Download Dataset' → select 'COCO' format.\n"
        f"     b. Extract the ZIP into: {os.path.abspath(ROBOFLOW_TRAIN_DIR)}/\n"
        f"        Expected layout:\n"
        f"          {ROBOFLOW_TRAIN_DIR}/_annotations.coco.json\n"
        f"          {ROBOFLOW_TRAIN_DIR}/<image_files>.jpg\n"
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
        "--rf-workspace",
        dest="rf_workspace",
        default=None,
        help=(
            "Roboflow workspace slug (the first path component in the Universe URL). "
            "Overrides the module-level default. "
            "Find a working public dataset at https://universe.roboflow.com"
        ),
    )
    parser.add_argument(
        "--rf-project",
        dest="rf_project",
        default=None,
        help=(
            "Roboflow project slug (the second path component in the Universe URL). "
            "Overrides the module-level default."
        ),
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
        download_roboflow(
            api_key=args.api_key,
            workspace=args.rf_workspace,
            project_slug=args.rf_project,
        )

    if args.finebadminton or args.all:
        download_finebadminton(hf_dir=args.hf_dir)


if __name__ == "__main__":
    main()
