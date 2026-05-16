"""Dataset download utilities.

Usage:
    # Single Roboflow dataset
    python utils/data_download.py --roboflow \\
        --api-key KEY \\
        --rf-dataset workspace/project

    # Multiple Roboflow datasets merged into one training dir
    python utils/data_download.py --roboflow \\
        --api-key KEY \\
        --rf-dataset workspace1/project1 \\
        --rf-dataset workspace2/project2

    # FineBadminton20k stroke classifier dataset
    python utils/data_download.py --finebadminton \\
        [--hf-dir /oscar/scratch/$USER/finebadminton20k]

    # Both
    python utils/data_download.py --all --api-key KEY \\
        --rf-dataset workspace1/project1 \\
        --rf-dataset workspace2/project2

Roboflow (player + shuttle detection):
    Each --rf-dataset entry is downloaded to data/input/roboflow/<project>/.
    After all downloads finish, all COCO datasets are merged into
    data/input/train/ (images + a single _annotations.coco.json).
    API key: the *private* key from your Roboflow account → Settings → API Keys.

    Finding working Roboflow datasets
    -----------------------------------
    1. Go to https://universe.roboflow.com
    2. Search "badminton" or "shuttlecock" — filter Object Detection, Public
    3. Confirm a "Download Dataset" button exists (= published version available)
    4. Copy slugs from the URL: universe.roboflow.com/<workspace>/<project>
    5. Pass as: --rf-dataset <workspace>/<project>

    If a project has no Download button it is private or has no published version.

    Category normalisation:
    Common shuttle label variants ("shuttle", "bird", "cock", "feather") are
    remapped to "shuttlecock".  Player variants ("person", "athlete",
    "badminton_player", "badminton-player") are remapped to "player".
    Everything else keeps its original name.

FineBadminton20k (stroke classification, ~43 GB):
    Source: https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K
    --hf-dir sets the local download target (default: data/input/finebadminton).
    On OSCAR this should be a /scratch path to avoid quota issues.

Both: skip if target directory already has files (idempotent).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# ── Roboflow constants ────────────────────────────────────────────────────────
ROBOFLOW_FORMAT     = "coco"
ROBOFLOW_RAW_DIR    = "data/input/roboflow"   # per-project subdirs live here
ROBOFLOW_TRAIN_DIR  = "data/input/train"       # merged output

# Category normalisation map — keys are lower-cased source names.
# Add more aliases here as needed.
CATEGORY_REMAP: dict[str, str] = {
    # shuttlecock aliases
    "shuttle":           "shuttlecock",
    "shuttlecock":       "shuttlecock",
    "bird":              "shuttlecock",
    "cock":              "shuttlecock",
    "feather":           "shuttlecock",
    "feathercock":       "shuttlecock",
    "badminton_shuttle": "shuttlecock",
    "badminton-shuttle": "shuttlecock",
    # player aliases
    "person":                "player",
    "player":                "player",
    "athlete":               "player",
    "badminton_player":      "player",
    "badminton-player":      "player",
    "badminton player":      "player",
}

# ── FineBadminton constants ───────────────────────────────────────────────────
FINEBADMINTON_DIR   = "data/input/finebadminton"
FINEBADMINTON_HF_ID = "iLearn-Lab/Finebadminton-20K"
FINEBADMINTON_URL   = "https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dir_has_files(directory: str) -> bool:
    """Return True if directory exists and contains at least one file."""
    if not os.path.isdir(directory):
        return False
    for _, _, files in os.walk(directory):
        if files:
            return True
    return False


VALID_TYPES = {"shuttle", "player"}

def _parse_rf_dataset(spec: str) -> Tuple[str, str, str | None]:
    """Parse 'workspace/project[:type]' into (workspace, project, type).

    type must be one of: shuttle, player  (or omitted → None, goes to data/input/train/).
    Examples:
      ws/shuttle-ds:shuttle   → type='shuttle' → merged into data/input/train/shuttle/
      ws/player-ds:player     → type='player'  → merged into data/input/train/player/
      ws/mixed-ds             → type=None       → merged into data/input/train/
    """
    # Split off optional :type suffix
    if ":" in spec:
        ws_proj, dtype = spec.rsplit(":", 1)
        dtype = dtype.lower().strip()
        if dtype not in VALID_TYPES:
            raise argparse.ArgumentTypeError(
                f"Unknown dataset type {dtype!r}. Valid types: {sorted(VALID_TYPES)}"
            )
    else:
        ws_proj = spec
        dtype = None

    parts = ws_proj.strip().split("/", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--rf-dataset must be in 'workspace/project[:type]' form, got: {spec!r}"
        )
    return parts[0], parts[1], dtype


def _get_latest_version(project) -> int:
    """Return the highest available version number for a Roboflow project."""
    try:
        versions = project.versions()
        if versions:
            return max(int(v.version) for v in versions)
    except Exception:
        pass
    for v in range(1, 11):
        try:
            project.version(v)
            return v
        except Exception:
            continue
    raise RuntimeError(
        f"No published versions found for this project. "
        "Ensure the project has a published version and a Download button on Universe."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-dataset Roboflow download
# ─────────────────────────────────────────────────────────────────────────────

def _roboflow_api_download(
    api_key: str,
    workspace: str,
    project_slug: str,
    version_num: int,
    target: str,
) -> bool:
    """Download a Roboflow dataset by hitting the REST API directly.

    Bypasses the Roboflow Python SDK's extraction logic, which silently fails
    on some HPC/NFS environments.  Downloads the COCO zip via requests and
    extracts it with zipfile — fully under our control.

    Returns True on success.
    """
    import zipfile
    import tempfile

    try:
        import requests
    except ImportError:
        print("[DOWNLOAD] 'requests' not installed — pip install requests")
        return False

    url = (
        f"https://api.roboflow.com/{workspace}/{project_slug}"
        f"/{version_num}/{ROBOFLOW_FORMAT}"
        f"?api_key={api_key}"
    )
    print(f"[DOWNLOAD]   → REST API: GET {url.split('?')[0]}?api_key=***")

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"[DOWNLOAD]   API error {resp.status_code}: {resp.text[:300]}")
        return False

    # The API returns JSON with a {"export": {"link": "..."}} field
    info = resp.json()
    zip_url = (
        info.get("export", {}).get("link")
        or info.get("link")
        or info.get("url")
    )
    if not zip_url:
        print(f"[DOWNLOAD]   Unexpected API response shape: {list(info.keys())}")
        return False

    print(f"[DOWNLOAD]   → downloading zip …")
    zip_resp = requests.get(zip_url, stream=True, timeout=300)
    zip_resp.raise_for_status()

    # Stream to a temp file then extract — avoids holding GBs in memory
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
        downloaded = 0
        for chunk in zip_resp.iter_content(chunk_size=1 << 20):  # 1 MB
            tmp.write(chunk)
            downloaded += len(chunk)
        print(f"[DOWNLOAD]   → {downloaded / 1e6:.1f} MB downloaded")

    print(f"[DOWNLOAD]   → extracting to {target} …")
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(tmp_path, "r") as zf:
        zf.extractall(target)
    os.unlink(tmp_path)

    n_files = sum(len(fs) for _, _, fs in os.walk(target))
    print(f"[DOWNLOAD]   → extracted {n_files} files")
    return n_files > 0


def _download_one_roboflow(
    api_key: str,
    workspace: str,
    project_slug: str,
) -> str | None:
    """Download one Roboflow dataset to data/input/roboflow/<project_slug>/.

    Strategy:
      1. Try the Roboflow Python SDK (fast, well-tested on normal machines).
      2. If SDK leaves the target directory empty, fall back to the REST API
         + direct zip extraction (reliable on HPC/NFS environments where the
         SDK's extraction silently fails).

    Returns the local directory path on success, None on failure.
    """
    target = os.path.abspath(os.path.join(ROBOFLOW_RAW_DIR, project_slug))

    if _dir_has_files(target):
        print(f"[DOWNLOAD] '{workspace}/{project_slug}' already present at '{target}' — skipping.")
        return target

    print(f"[DOWNLOAD] Downloading Roboflow workspace='{workspace}'  project='{project_slug}'")

    version_num: int | None = None

    # ── Attempt 1: Roboflow Python SDK ────────────────────────────────────────
    try:
        from roboflow import Roboflow

        rf      = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_slug)
        version_num = _get_latest_version(project)
        print(f"[DOWNLOAD]   → version {version_num}  (via SDK)")

        os.makedirs(target, exist_ok=True)
        project.version(version_num).download(
            ROBOFLOW_FORMAT,
            location=target,
            overwrite=True,
        )
    except ImportError:
        print("[DOWNLOAD]   roboflow SDK not installed — skipping to REST fallback")
    except Exception as e:
        print(f"[DOWNLOAD]   SDK error: {e} — trying REST fallback")

    # ── Check if SDK wrote anything ───────────────────────────────────────────
    if _dir_has_files(target):
        n = sum(len(fs) for _, _, fs in os.walk(target))
        print(f"[DOWNLOAD]   → SDK wrote {n} files to {target}")
        return target

    print(f"[DOWNLOAD]   SDK left target empty — falling back to REST API download")

    # ── Attempt 2: REST API + direct zip extraction ───────────────────────────
    if version_num is None:
        # SDK wasn't available; probe version via API
        try:
            import requests
            for v in range(1, 11):
                r = requests.get(
                    f"https://api.roboflow.com/{workspace}/{project_slug}/{v}"
                    f"?api_key={api_key}",
                    timeout=10,
                )
                if r.status_code == 200:
                    version_num = v
                    break
        except Exception:
            pass

    if version_num is None:
        print(f"[DOWNLOAD]   Could not determine version number — giving up.")
        _print_roboflow_manual(workspace, project_slug)
        return None

    try:
        ok = _roboflow_api_download(api_key, workspace, project_slug, version_num, target)
        if ok:
            print(f"[DOWNLOAD]   → REST download succeeded: {target}")
            return target
    except Exception as e:
        print(f"[DOWNLOAD]   REST download failed: {e}")

    _print_roboflow_manual(workspace, project_slug)
    return None


def _print_roboflow_manual(workspace: str, project: str) -> None:
    url = f"https://universe.roboflow.com/{workspace}/{project}"
    print(
        f"\n[DOWNLOAD] Manual download instructions for {workspace}/{project}:\n"
        f"  1. Confirm a Download button exists at:\n"
        f"       {url}\n"
        f"     If there is NO 'Download Dataset' button, the project is private or\n"
        f"     has no published version.  Search https://universe.roboflow.com for\n"
        f"     a public alternative and pass it via --rf-dataset workspace/project.\n"
        f"\n"
        f"  2. Or download manually:\n"
        f"     a. Click 'Download Dataset' → select 'COCO' format.\n"
        f"     b. Extract the ZIP into:\n"
        f"          {os.path.abspath(os.path.join(ROBOFLOW_RAW_DIR, project))}/\n"
        f"        Expected layout:\n"
        f"          {ROBOFLOW_RAW_DIR}/{project}/_annotations.coco.json\n"
        f"          {ROBOFLOW_RAW_DIR}/{project}/<image_files>.jpg\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# COCO merge
# ─────────────────────────────────────────────────────────────────────────────

def _find_coco_json(directory: str) -> str | None:
    """Find the COCO annotation JSON inside a Roboflow download directory.

    Roboflow SDK nests downloads like:
      <location>/<ProjectName>-<version>/train/_annotations.coco.json
    or (simpler layout):
      <location>/train/_annotations.coco.json
    or directly:
      <location>/_annotations.coco.json

    We always prefer the 'train' split over 'valid'/'test'.
    """
    # Priority 1: direct or explicit train/ paths
    candidates = [
        os.path.join(directory, "_annotations.coco.json"),
        os.path.join(directory, "train", "_annotations.coco.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Priority 2: walk entire subtree — collect all annotation JSONs,
    # prefer any path whose parent directory is named 'train'.
    found: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Skip valid/test splits so we don't accidentally use them
        dirs[:] = [d for d in dirs if d not in ("valid", "test", "validation")]
        for f in files:
            if f.endswith(".json") and "annotation" in f.lower():
                found.append(os.path.join(root, f))

    if not found:
        return None

    # Prefer whichever path contains 'train' in its directory components
    for p in found:
        if "train" in Path(p).parts:
            return p
    return found[0]


def _normalise_category_name(name: str) -> str:
    return CATEGORY_REMAP.get(name.lower().strip(), name.lower().strip())


def merge_coco_datasets(source_dirs: List[str], output_dir: str) -> None:
    """Merge multiple COCO-format directories into a single output_dir.

    Strategy:
    - Unified category list, normalised names (e.g. "shuttle" → "shuttlecock")
    - Images copied to output_dir/ with dataset prefix to avoid name collisions
    - Image IDs and annotation IDs re-mapped to be globally unique
    - Writes output_dir/_annotations.coco.json
    """
    if _dir_has_files(output_dir):
        print(f"[MERGE] '{output_dir}' already has files — skipping merge.")
        print(f"        Delete it to force a re-merge.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Pass 1: collect all unique normalised category names ──────────────────
    all_category_names: list[str] = []
    seen_names: set[str] = set()
    for src_dir in source_dirs:
        ann_path = _find_coco_json(src_dir)
        if ann_path is None:
            print(f"[MERGE] WARNING: No COCO JSON found in '{src_dir}' — skipping.")
            continue
        with open(ann_path) as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            norm = _normalise_category_name(cat["name"])
            if norm not in seen_names:
                seen_names.add(norm)
                all_category_names.append(norm)

    if not all_category_names:
        print("[MERGE] No valid COCO datasets found — nothing to merge.")
        return

    # Assign stable category IDs (sorted for determinism)
    all_category_names.sort()
    unified_categories = [
        {"id": i + 1, "name": name, "supercategory": "none"}
        for i, name in enumerate(all_category_names)
    ]
    cat_name_to_id = {c["name"]: c["id"] for c in unified_categories}

    print(f"[MERGE] Unified categories: {all_category_names}")

    # ── Pass 2: merge images + annotations ───────────────────────────────────
    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    global_image_id = 1
    global_ann_id   = 1

    for src_dir in source_dirs:
        ann_path = _find_coco_json(src_dir)
        if ann_path is None:
            continue

        src_name  = Path(src_dir).name   # used as filename prefix
        image_dir = os.path.dirname(ann_path)

        with open(ann_path) as f:
            data = json.load(f)

        # Build per-source category remap: old_id → unified_id
        src_cat_remap: dict[int, int] = {}
        for cat in data.get("categories", []):
            norm = _normalise_category_name(cat["name"])
            if norm in cat_name_to_id:
                src_cat_remap[cat["id"]] = cat_name_to_id[norm]

        # Remap images
        src_image_remap: dict[int, int] = {}   # old_id → new_id
        for img in data.get("images", []):
            old_id   = img["id"]
            new_id   = global_image_id
            global_image_id += 1
            src_image_remap[old_id] = new_id

            old_fname = img["file_name"]
            # Roboflow sometimes stores file_name as "train/img.jpg" — use basename
            # for both the output name and the source lookup.
            bare_fname = os.path.basename(old_fname)
            new_fname  = f"{src_name}__{bare_fname}"

            # Probe candidate source locations
            src_candidates = [
                os.path.join(image_dir, old_fname),          # exact as stored
                os.path.join(image_dir, bare_fname),         # basename only
                os.path.join(image_dir, "..", bare_fname),   # one level up
            ]
            src_path = next((p for p in src_candidates if os.path.isfile(p)), None)
            dst_path = os.path.join(output_dir, new_fname)

            if src_path and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            elif not src_path:
                print(
                    f"[MERGE] WARNING: image not found — tried:\n"
                    + "\n".join(f"  {p}" for p in src_candidates)
                )

            merged_images.append({
                "id":        new_id,
                "file_name": new_fname,
                "width":     img.get("width", 0),
                "height":    img.get("height", 0),
            })

        # Remap annotations
        for ann in data.get("annotations", []):
            new_cat_id = src_cat_remap.get(ann["category_id"])
            if new_cat_id is None:
                continue   # category not in unified set — drop
            new_img_id = src_image_remap.get(ann["image_id"])
            if new_img_id is None:
                continue

            merged_annotations.append({
                "id":           global_ann_id,
                "image_id":     new_img_id,
                "category_id":  new_cat_id,
                "bbox":         ann.get("bbox", []),
                "area":         ann.get("area", 0),
                "segmentation": ann.get("segmentation", []),
                "iscrowd":      ann.get("iscrowd", 0),
            })
            global_ann_id += 1

        n_imgs = len(data.get("images", []))
        n_anns = len(data.get("annotations", []))
        print(f"[MERGE]   {src_name}: {n_imgs} images, {n_anns} annotations")

    # ── Write merged JSON ─────────────────────────────────────────────────────
    merged_coco = {
        "info":        {"description": "Merged badminton detection dataset"},
        "licenses":    [],
        "categories":  unified_categories,
        "images":      merged_images,
        "annotations": merged_annotations,
    }
    out_json = os.path.join(output_dir, "_annotations.coco.json")
    with open(out_json, "w") as f:
        json.dump(merged_coco, f)

    print(
        f"\n[MERGE] Done.\n"
        f"  Datasets merged : {len(source_dirs)}\n"
        f"  Total images    : {len(merged_images)}\n"
        f"  Total annotations: {len(merged_annotations)}\n"
        f"  Categories      : {all_category_names}\n"
        f"  Output          : {os.path.abspath(output_dir)}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-dataset Roboflow entry point
# ─────────────────────────────────────────────────────────────────────────────

def download_roboflow(
    api_key: str | None = None,
    datasets: List[Tuple[str, str, str | None]] | None = None,
) -> None:
    """Download Roboflow datasets and merge each type group independently.

    datasets: list of (workspace, project_slug, type) triples.
      type = 'shuttle' → merged into data/input/train/shuttle/
      type = 'player'  → merged into data/input/train/player/
      type = None      → merged into data/input/train/  (untagged / legacy)

    This keeps shuttle and player training data strictly separated, preventing
    the missing-labels problem that occurs when a shuttle-only dataset is merged
    with a player-only dataset and a single YOLO model is asked to detect both.
    """
    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not datasets:
        datasets = [(_DEFAULT_WORKSPACE, _DEFAULT_PROJECT, None)]

    if not api_key:
        print("[DOWNLOAD] No API key provided.")
        for ws, proj, _ in datasets:
            _print_roboflow_manual(ws, proj)
        return

    # ── Download each dataset to its raw staging dir ──────────────────────────
    # Map (workspace, project, type) → local raw dir
    downloaded: list[Tuple[str, str | None]] = []   # (raw_dir, type)
    for ws, proj, dtype in datasets:
        raw_dir = _download_one_roboflow(api_key, ws, proj)
        if raw_dir:
            downloaded.append((raw_dir, dtype))

    if not downloaded:
        print("[DOWNLOAD] No datasets downloaded — merge skipped.")
        return

    # ── Group by type ─────────────────────────────────────────────────────────
    from collections import defaultdict
    groups: dict[str | None, list[str]] = defaultdict(list)
    for raw_dir, dtype in downloaded:
        groups[dtype].append(raw_dir)

    # ── Merge each group into its output directory ────────────────────────────
    for dtype, dirs in groups.items():
        if dtype is None:
            out_dir = ROBOFLOW_TRAIN_DIR              # untagged → data/input/train/
        else:
            out_dir = os.path.join(ROBOFLOW_TRAIN_DIR, dtype)  # data/input/train/shuttle/ etc.

        label = f"'{dtype}'" if dtype else "untagged"
        print(f"\n[MERGE] Group {label}: {len(dirs)} dataset(s) → '{out_dir}'")
        merge_coco_datasets(dirs, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[DOWNLOAD] All done. Training data locations:")
    for dtype, dirs in groups.items():
        if dtype is None:
            out_dir = ROBOFLOW_TRAIN_DIR
        else:
            out_dir = os.path.join(ROBOFLOW_TRAIN_DIR, dtype)
        label = dtype or "untagged"
        print(f"  {label:10s} ({len(dirs)} dataset(s)) → {os.path.abspath(out_dir)}")
    print()
    print("  Pass these to training:")
    if "shuttle" in groups:
        print(f"    TrackNet : --data-dir {os.path.join(ROBOFLOW_TRAIN_DIR, 'shuttle')}")
    if "player" in groups:
        print(f"    YOLO     : --data-dir {os.path.join(ROBOFLOW_TRAIN_DIR, 'player')}")


# ── Legacy defaults (used when no --rf-dataset given) ────────────────────────
_DEFAULT_WORKSPACE = "badminton-rojkf"   # PLACEHOLDER — replace via --rf-dataset
_DEFAULT_PROJECT   = "badminton-hehp8"   # PLACEHOLDER — replace via --rf-dataset


# ─────────────────────────────────────────────────────────────────────────────
# FineBadminton20k
# ─────────────────────────────────────────────────────────────────────────────

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
        pass
    except Exception as e:
        print(f"[DOWNLOAD] huggingface_hub download failed ({e})")

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
        f"    snapshot_download(repo_id='{FINEBADMINTON_HF_ID}',\n"
        f"                      repo_type='dataset', local_dir='{target}')\n"
        "\n"
        "  Option C — datasets library:\n"
        "    from datasets import load_dataset\n"
        f"    ds = load_dataset('{FINEBADMINTON_HF_ID}', cache_dir='{target}')\n"
        f"    ds.save_to_disk('{target}')\n"
        "\n"
        f"  Confirm dataset ID at: {FINEBADMINTON_URL}\n"
        "\n"
        "  After download:\n"
        f"    python training/train_stroke.py --data-dir {target}\n"
        "  Or on OSCAR via SLURM:\n"
        f"    sbatch --export=MODE=train-stroke,FINEBADMINTON_DIR={target} slurm_train.sh\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download badminton training datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # 3 shuttle datasets + 1 player dataset, merged separately:\n"
            "  python utils/data_download.py --roboflow \\\n"
            "      --api-key $ROBOFLOW_API_KEY \\\n"
            "      --rf-dataset ws1/shuttle-ds1:shuttle \\\n"
            "      --rf-dataset ws2/shuttle-ds2:shuttle \\\n"
            "      --rf-dataset ws3/shuttle-ds3:shuttle \\\n"
            "      --rf-dataset ws4/player-ds:player\n"
            "\n"
            "  Output:\n"
            "    data/input/train/shuttle/  ← 3 shuttle datasets merged\n"
            "    data/input/train/player/   ← 1 player dataset\n"
            "\n"
            "  # FineBadminton on OSCAR scratch:\n"
            "  python utils/data_download.py --finebadminton \\\n"
            "      --hf-dir /oscar/scratch/$USER/finebadminton20k\n"
        ),
    )
    parser.add_argument(
        "--roboflow",
        action="store_true",
        help="Download Roboflow dataset(s) and merge into data/input/train/.",
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
        "--rf-dataset",
        dest="rf_datasets",
        metavar="WORKSPACE/PROJECT[:TYPE]",
        action="append",
        default=[],
        type=_parse_rf_dataset,
        help=(
            "Roboflow dataset in 'workspace/project[:type]' form. "
            "type must be 'shuttle' or 'player'. "
            "Datasets with the same type are merged together into "
            "data/input/train/<type>/. "
            "Repeat this flag for multiple datasets. "
            "Find slugs at https://universe.roboflow.com"
        ),
    )
    parser.add_argument(
        "--hf-dir",
        dest="hf_dir",
        default=None,
        help=(
            "Local directory for FineBadminton20k download. "
            "On OSCAR use /oscar/scratch/$USER/finebadminton20k. "
            "Default: data/input/finebadminton"
        ),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Print a structured report of what is on disk under data/input/ "
            "without downloading or merging anything. Useful for debugging."
        ),
    )
    args = parser.parse_args()

    if args.diagnose:
        _diagnose()
        return

    if not (args.roboflow or args.finebadminton or args.all):
        parser.print_help()
        sys.exit(1)

    if args.roboflow or args.all:
        datasets = args.rf_datasets if args.rf_datasets else None
        download_roboflow(api_key=args.api_key, datasets=datasets)

    if args.finebadminton or args.all:
        download_finebadminton(hf_dir=args.hf_dir)


def _diagnose() -> None:
    """Print a structured report of data/input/ to help debug download/merge issues."""
    import glob

    print("=" * 60)
    print("DIAGNOSE: data/input layout")
    print(f"  CWD: {os.getcwd()}")
    print("=" * 60)

    for section, root in [
        ("RAW ROBOFLOW DOWNLOADS", ROBOFLOW_RAW_DIR),
        ("MERGED TRAIN DIR",       ROBOFLOW_TRAIN_DIR),
        ("FINEBADMINTON",          FINEBADMINTON_DIR),
    ]:
        print(f"\n── {section} ({root}) ──")
        abs_root = os.path.abspath(root)
        if not os.path.isdir(abs_root):
            print(f"   [NOT FOUND]  {abs_root}")
            continue

        total_files = 0
        for dirpath, dirnames, filenames in os.walk(abs_root):
            dirnames.sort()
            rel = os.path.relpath(dirpath, abs_root)
            indent = "   " + "  " * rel.count(os.sep)
            if rel == ".":
                print(f"   {abs_root}/")
            else:
                print(f"{indent}{os.path.basename(dirpath)}/")
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                size  = os.path.getsize(fpath)
                total_files += 1
                if total_files <= 30 or fname.endswith(".json"):
                    print(f"{indent}  {fname}  ({size:,} bytes)")
                elif total_files == 31:
                    print(f"{indent}  ... (truncated, use 'find {root} -type f | wc -l' for count)")

        # COCO JSON report
        json_files = []
        for dirpath, _, filenames in os.walk(abs_root):
            for f in filenames:
                if f.endswith(".json") and "annotation" in f.lower():
                    json_files.append(os.path.join(dirpath, f))
        if json_files:
            print(f"\n   COCO JSONs found:")
            for jf in json_files:
                try:
                    with open(jf) as f:
                        d = json.load(f)
                    n_imgs  = len(d.get("images", []))
                    n_anns  = len(d.get("annotations", []))
                    cats    = [c["name"] for c in d.get("categories", [])]
                    # Check how many image files actually exist on disk
                    img_dir   = os.path.dirname(jf)
                    n_present = sum(
                        1 for img in d.get("images", [])
                        if os.path.isfile(os.path.join(img_dir, os.path.basename(img["file_name"])))
                        or os.path.isfile(os.path.join(img_dir, img["file_name"]))
                    )
                    print(f"     {os.path.relpath(jf, abs_root)}")
                    print(f"       images in JSON : {n_imgs}  (files present: {n_present})")
                    print(f"       annotations    : {n_anns}")
                    print(f"       categories     : {cats}")
                    if n_present < n_imgs:
                        print(f"       *** WARNING: {n_imgs - n_present} image files missing on disk ***")
                        # Show first missing example
                        for img in d.get("images", [])[:3]:
                            p1 = os.path.join(img_dir, os.path.basename(img["file_name"]))
                            p2 = os.path.join(img_dir, img["file_name"])
                            if not os.path.isfile(p1) and not os.path.isfile(p2):
                                print(f"       Example missing: file_name={img['file_name']!r}")
                                print(f"         tried: {os.path.relpath(p1, abs_root)}")
                                print(f"         tried: {os.path.relpath(p2, abs_root)}")
                                break
                except Exception as e:
                    print(f"     {jf}  [ERROR reading: {e}]")
        else:
            print(f"\n   No COCO annotation JSONs found under {root}/")
        print(f"\n   Total files: {total_files}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
