"""YOLO fine-tuning on Roboflow badminton dataset.

Converts Roboflow COCO annotations to YOLO format if needed, then
calls model.train(...) via Ultralytics.

Usage:
    python training/train_yolo.py --data-dir data/input/train_mog_reflect --output-dir data/output

Best weights saved to weights/yolo_badminton.pt.

CLI args: --data-dir, --output-dir, --epochs, --batch-size, --lr, --device
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def coco_to_yolo_format(data_dir: str, yolo_dir: str) -> str:
    """Convert COCO annotations to YOLO format for Ultralytics.

    Returns path to generated data.yaml.
    """
    ann_path = os.path.join(data_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")

    with open(ann_path, "r") as f:
        coco = json.load(f)

    # Category mapping: keep person (class 0) and shuttle
    cat_id_to_yolo = {}
    cat_names = []
    for cat in coco.get("categories", []):
        name = cat["name"].lower()
        if name in ["person", "player", "player1", "player2"]:
            yolo_id = 0
            # Always map all player variants to class 0
            cat_id_to_yolo[cat["id"]] = yolo_id
            cat_names.append((yolo_id, cat["name"]))
            print(f"[TRAIN_YOLO] Mapped COCO cat_id={cat['id']} name={cat['name']} -> yolo_id=0")
        elif "shuttle" in name or "badminton" in name:
            yolo_id = 1
            cat_id_to_yolo[cat["id"]] = yolo_id
            cat_names.append((yolo_id, cat["name"]))
            print(f"[TRAIN_YOLO] Mapped COCO cat_id={cat['id']} name={cat['name']} -> yolo_id=1")
        else:
            continue

    print(f"[TRAIN_YOLO] Total mapped categories: {len(cat_id_to_yolo)}")
    
    # Deduplicate names by id (keep first occurrence)
    id_to_name = {}
    for yid, n in cat_names:
        if yid not in id_to_name:
            id_to_name[yid] = n
    nc = max(id_to_name.keys()) + 1 if id_to_name else 1
    names = [id_to_name.get(i, f"class_{i}") for i in range(nc)]
    print(f"[TRAIN_YOLO] YOLO class count: {nc}, names: {names}")

    # Build image -> annotations map
    img_map = {img["id"]: img for img in coco["images"]}
    ann_by_img: dict[int, list] = {img["id"]: [] for img in coco["images"]}
    for ann in coco.get("annotations", []):
        if ann["category_id"] in cat_id_to_yolo:
            ann_by_img[ann["image_id"]].append(ann)
    
    # Count labels
    total_labels = sum(len(anns) for anns in ann_by_img.values())
    images_with_labels = sum(1 for anns in ann_by_img.values() if anns)
    print(f"[TRAIN_YOLO] Total annotations in COCO: {len(coco.get('annotations', []))}")
    print(f"[TRAIN_YOLO] Matched annotations: {total_labels} across {images_with_labels} images")


    # Write YOLO label files
    imgs_dir = os.path.join(yolo_dir, "images", "train")
    labels_dir = os.path.join(yolo_dir, "labels", "train")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for img_id, img_info in img_map.items():
        src = os.path.join(data_dir, img_info["file_name"])
        if not os.path.exists(src):
            continue

        img_w = float(img_info.get("width", 1))
        img_h = float(img_info.get("height", 1))
        dst_img = os.path.join(imgs_dir, os.path.basename(img_info["file_name"]))
        if not os.path.exists(dst_img):
            shutil.copy2(src, dst_img)

        label_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        lines = []
        for ann in ann_by_img.get(img_id, []):
            yolo_cls = cat_id_to_yolo[ann["category_id"]]
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / img_w
            cy = (by + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # Write data.yaml
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # Use same for simplicity; override as needed
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

    print(f"[TRAIN_YOLO] YOLO dataset written to {yolo_dir}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="YOLO fine-tuning")
    parser.add_argument("--data-dir", default="data/input/train_mog_reflect")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.config_loader import load_config

    cfg = load_config(args.config)
    device = args.device or getattr(cfg, "device", "cpu")
    weights_out = "weights/yolo_badminton.pt"
    os.makedirs("weights", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert COCO to YOLO format
    yolo_dir = os.path.join(args.output_dir, "yolo_dataset")
    yaml_path = coco_to_yolo_format(args.data_dir, yolo_dir)

    # Train with Ultralytics
    from ultralytics import YOLO

    base_weights = getattr(cfg, "player_weights", "weights/yolo_badminton.pt")
    if not os.path.exists(base_weights):
        base_weights = "yolov8n.pt"

    model = YOLO(base_weights)
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr,
        device=device,
        project=args.output_dir,
        name="yolo_train",
        verbose=True,
    )

    # Copy best weights
    best = os.path.join(args.output_dir, "yolo_train", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy2(best, weights_out)
        print(f"[TRAIN_YOLO] Best weights saved to {weights_out}")
    else:
        print("[TRAIN_YOLO] Warning: could not find best.pt to copy.")

    print(f"[TRAIN_YOLO] Training complete.")


if __name__ == "__main__":
    main()
