"""TrackNet fine-tuning with DDP support.

Usage:
    # Single GPU / CPU:
    python training/train_tracknet.py --data-dir data/input/train_mog_reflect --output-dir data/output

    # Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=2 training/train_tracknet.py --world-size 2 ...

Generates heatmap GT as 2D Gaussian centered at shuttle bbox center (sigma=2px).
Loss: BCELoss on heatmap output.
Checkpoints: {output_dir}/checkpoints/tracknet_epoch_{n}.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 2.0) -> np.ndarray:
    """Generate a 2D Gaussian heatmap of size (h, w) centered at (cx, cy)."""
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    hm = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)


class ShuttleHeatmapDataset(Dataset):
    """Dataset loading 3-frame sequences and generating Gaussian heatmap GT."""

    def __init__(self, data_dir: str, annotations_file: str, expected_h: int = 288, expected_w: int = 512):
        self.data_dir = data_dir
        self.expected_h = expected_h
        self.expected_w = expected_w

        ann_path = os.path.join(data_dir, annotations_file)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotations not found: {ann_path}")

        with open(ann_path, "r") as f:
            coco = json.load(f)

        # Build image id -> info map
        self.id_to_img = {img["id"]: img for img in coco["images"]}

        # Build image id -> shuttle bbox (first shuttle annotation)
        # COCO categories: look for "shuttle" or "badminton"
        shuttle_cat_ids = set()
        for cat in coco.get("categories", []):
            name = cat.get("name", "").lower()
            if "shuttle" in name or "badminton" in name:
                shuttle_cat_ids.add(cat["id"])

        self.id_to_bbox: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            if ann["category_id"] in shuttle_cat_ids:
                iid = ann["image_id"]
                if iid not in self.id_to_bbox:
                    self.id_to_bbox[iid] = ann["bbox"]  # [x, y, w, h] COCO format

        self.image_ids = list(self.id_to_img.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        img_info = self.id_to_img[image_id]
        img_path = os.path.join(self.data_dir, img_info["file_name"])

        frame = cv2.imread(img_path)
        if frame is None:
            frame = np.zeros((self.expected_h, self.expected_w, 3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (self.expected_w, self.expected_h))

        # Build 9-channel input (replicate frame 3 times)
        img_float = frame_rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_float, (2, 0, 1))  # 3, H, W
        inp = np.concatenate([chw, chw, chw], axis=0)  # 9, H, W

        # Build Gaussian heatmap target
        bbox = self.id_to_bbox.get(image_id)
        if bbox is not None:
            bx, by, bw, bh = bbox
            # Scale to expected resolution
            orig_h, orig_w = frame.shape[:2]
            cx = (bx + bw / 2) * (self.expected_w / max(orig_w, 1))
            cy = (by + bh / 2) * (self.expected_h / max(orig_h, 1))
            hm = gaussian_heatmap(self.expected_h, self.expected_w, cx, cy, sigma=2.0)
        else:
            hm = np.zeros((self.expected_h, self.expected_w), dtype=np.float32)

        target = hm[np.newaxis, :, :]  # 1, H, W

        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target),
        }


def train_one_epoch(model, dataloader, optimizer, criterion, device, rank: int = 0) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # Use first output channel only
        loss = criterion(outputs[:, :1, :, :], targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="TrackNet fine-tuning")
    parser.add_argument("--data-dir", default="data/input/train_mog_reflect")
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weights", default=None, help="Starting weights path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--annotations", default="_annotations.coco.json")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.config_loader import load_config
    from models.TrackNet import TrackNet
    from training.evaluate import evaluate_tracknet_epoch

    cfg = load_config(args.config)
    expected_h = int(getattr(cfg, "tracknet_expected_h", 288))
    expected_w = int(getattr(cfg, "tracknet_expected_w", 512))
    val_split = float(getattr(cfg, "train_val_split", 0.8))

    # ── DDP setup ─────────────────────────────────────────────────────────────
    use_ddp = args.world_size > 1 and "RANK" in os.environ
    rank = 0
    local_rank = 0
    if use_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.device:
        device = torch.device(args.device)
    elif use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[TRAIN_TN] device={device} world_size={args.world_size}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ShuttleHeatmapDataset(
        args.data_dir,
        args.annotations,
        expected_h=expected_h,
        expected_w=expected_w,
    )
    n_val = max(1, int(len(dataset) * (1 - val_split)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    sampler = DistributedSampler(train_ds) if use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(getattr(cfg, "train_num_workers", 4)),
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TrackNet(out_channels=1)
    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location="cpu")
        sd = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f"[TRAIN_TN] Loaded starting weights from {args.weights}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=float(getattr(cfg, "train_min_lr", 1e-6)),
    )
    criterion = nn.BCELoss()

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        if use_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, rank)
        scheduler.step()

        if rank == 0:
            val_metrics = evaluate_tracknet_epoch(
                model.module if use_ddp else model,
                val_loader,
                device,
                cfg,
            )
            print(
                f"[TRAIN_TN] epoch={epoch}/{args.epochs} "
                f"loss={train_loss:.5f} "
                f"dist_err={val_metrics['distance_error_px']:.2f}px "
                f"mAP50={val_metrics['map_50']:.4f}"
            )

            ckpt_path = os.path.join(ckpt_dir, f"tracknet_epoch_{epoch}.pt")
            torch.save(
                {"epoch": epoch, "state_dict": (model.module if use_ddp else model).state_dict()},
                ckpt_path,
            )

    if use_ddp:
        dist.destroy_process_group()

    if rank == 0:
        print(f"[TRAIN_TN] Training complete. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
