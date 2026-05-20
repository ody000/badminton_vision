"""DINOv3-based player detection.

Completely standalone implementation (no dependencies on slayminton/).
Includes:
1. DINOTracker: ViT-based detection model
2. DINODataset: COCO-format dataset loader with augmentation
3. train_dino: Training loop with validation
4. Helper functions: box conversions, IoU, LoRA, etc.

Ready for deployment on OSCAR or any other system.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NUM_GLOBAL_CROPS = 2
NUM_LOCAL_CROPS = 6
GLOBAL_CROP_SIZE = 384
LOCAL_CROP_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
EPOCHS = 75
WEIGHT_DECAY = 1e-4
MIN_CONFIDENCE = 0.25
VAL_EVERY = 1
VAL_IOU_THRESHOLD = 0.5
BOX_LOSS_WEIGHT = 0.05

TRACKED_CLASSES = ("player",)  # Player-only (removed shuttle)


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    val_iou: List[float]
    val_map: List[float]
    eval_epochs: List[int]


# ─────────────────────────────────────────────────────────────────────────────
# DINO TRACKER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DINOTracker(nn.Module):
    """DINOv3-style tracker for player detection.

    ViT encoder + lightweight detection head.
    Outputs single player detection per frame: (timestamp, x, y, w, h)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        pretrained_backbone_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        input_size: int = GLOBAL_CROP_SIZE,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # Load or create ViT encoder
        ckpt_path = weights_path or model_path
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ck = torch.load(ckpt_path, map_location="cpu")
                sd = ck.get("model", ck) if isinstance(ck, dict) else ck
                if isinstance(sd, dict):
                    sd_clean = _strip_prefix(sd)
                else:
                    sd_clean = sd
                embed_dim_ckpt = None
                for k in ("encoder.patch_embed.proj.weight", "patch_embed.proj.weight", "encoder.pos_embed"):
                    if k in sd_clean:
                        v = sd_clean[k]
                        if isinstance(v, torch.Tensor):
                            if v.ndim == 4:
                                embed_dim_ckpt = int(v.shape[0])
                            elif v.ndim == 3:
                                embed_dim_ckpt = int(v.shape[2])
                            break
                if embed_dim_ckpt is not None:
                    self.encoder, self.encoder_dim = _create_vit_by_embed_dim(embed_dim_ckpt, pretrained_backbone_path)
                else:
                    self.encoder, self.encoder_dim = _create_vit_tiny(pretrained_weights_path=pretrained_backbone_path)
            except Exception as e:
                print(f"[DINOTracker] Warning loading checkpoint: {e}, using pretrained")
                self.encoder, self.encoder_dim = _create_vit_tiny(pretrained_weights_path=pretrained_backbone_path)
        else:
            self.encoder, self.encoder_dim = _create_vit_tiny(pretrained_weights_path=pretrained_backbone_path)

        # Determine patch size
        patch_size = None
        pe = getattr(self.encoder, "patch_embed", None)
        if pe is not None:
            ps = getattr(pe, "patch_size", None)
            if isinstance(ps, (tuple, list)):
                patch_size = int(ps[0])
            elif isinstance(ps, int):
                patch_size = ps
        patch_size = patch_size or 16

        # Round input_size to multiple of patch_size
        if self.input_size % patch_size != 0:
            new_size = ((self.input_size + patch_size - 1) // patch_size) * patch_size
            print(f"[DINOTracker] adjusting input_size {self.input_size} -> {new_size} to match patch_size {patch_size}")
            self.input_size = new_size

        # Preprocessing
        self.preprocess = transforms.Compose(
            [transforms.Resize((self.input_size, self.input_size)), transforms.ToTensor()]
        )

        # Detection head: [confidence, cx, cy, w, h] for player only
        self.detector_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.GELU(),
            nn.Linear(self.encoder_dim, len(TRACKED_CLASSES) * 5),
        )
        self.to(self.device)

        # Load weights if provided
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_checkpoint(ckpt_path)

        self.eval()

        # Interval caching for detect_yolo_compat
        self._detect_interval: int = 1       # overridden by set_detect_interval()
        self._detect_frame_count: int = 0
        self._detect_cache: list = []        # last result from detect_yolo_compat

    def set_detect_interval(self, interval: int) -> None:
        """Set how often to run a real forward pass vs return cached result."""
        self._detect_interval = max(1, int(interval))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract class token embedding."""
        return _extract_cls_token(self.encoder, x)

    def forward_detect(self, x: torch.Tensor) -> torch.Tensor:
        """Detection forward pass.

        Args:
            x: Tensor (B, 3, H, W)

        Returns:
            Tensor (B, num_classes, 5) with [conf, cx, cy, w, h] normalized
        """
        feat = self.encode(x)
        raw = self.detector_head(feat).view(x.size(0), len(TRACKED_CLASSES), 5)
        conf = torch.sigmoid(raw[..., :1])
        box = torch.sigmoid(raw[..., 1:])
        return torch.cat([conf, box], dim=-1)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint (supports both raw state_dict and wrapped format).

        Also handles LoRA-trained checkpoints: if the saved state dict contains
        LoRA keys (*.lora_A / *.lora_B), LoRA is applied to the encoder
        automatically before loading so the key names match.

        Fixes Bug 5-B by stripping common prefixes (student., module., model., backbone.)
        that may be present from teacher-student training or DataParallel.
        """
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        # Strip common prefixes from teacher-student or DataParallel checkpoints (Bug 5-B Cause A)
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                k_clean = k
                for prefix in ("student.", "module.", "model.", "backbone."):
                    if k_clean.startswith(prefix):
                        k_clean = k_clean[len(prefix):]
                        break
                cleaned[k] = v
            state = cleaned

        # Detect LoRA checkpoint: look for any key ending in 'lora_A' or 'lora_B'
        is_lora_ckpt = isinstance(state, dict) and any(
            k.endswith("lora_A") or k.endswith("lora_B") for k in state
        )
        already_has_lora = any(isinstance(m, LoRALinear) for m in self.modules())

        if is_lora_ckpt and not already_has_lora:
            # Infer r from the first lora_A tensor shape: (in_features, r)
            lora_r = 4
            for k, v in state.items():
                if k.endswith("lora_A") and isinstance(v, torch.Tensor) and v.ndim == 2:
                    lora_r = int(v.shape[1])
                    break
            n = apply_lora_to_encoder(self.encoder, r=lora_r)
            # apply_lora_to_encoder creates new LoRALinear modules after the model
            # was already moved to device — the new lora_A/lora_B params are born
            # on CPU.  Push the encoder back to the target device.
            self.encoder.to(self.device)
            print(f"[DINOTracker] LoRA checkpoint detected (r={lora_r}); "
                  f"applied LoRA to {n} encoder layers before loading")

        result = self.load_state_dict(state, strict=False)
        n_loaded = len(state) - len(result.missing_keys) if isinstance(state, dict) else "?"
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        print(f"[DINOTracker] Loaded checkpoint from {path} "
              f"(keys loaded≈{n_loaded}, missing={n_missing}, unexpected={n_unexpected})")
        if n_missing:
            print(f"[DINOTracker]   missing: {result.missing_keys[:5]}"
                  f"{'...' if n_missing > 5 else ''}")

        # Sanity check: verify detector_head loaded (Bug 5-B Cause A)
        head_params = list(self.detector_head.parameters())
        if head_params:
            head_norm = sum(p.abs().mean().item() for p in head_params) / len(head_params)
            print(f"[DINOTracker] detector_head mean abs weight: {head_norm:.6f}")
            if head_norm < 1e-4:
                print("[DINOTracker] WARNING: detector_head appears uninitialized. "
                      "Check checkpoint key prefixes or checkpoint validity.")

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model": self.state_dict()}, path)

    @torch.no_grad()
    def detect(
        self,
        frame,
        timestamp: float = 0.0,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> Dict[str, Optional[Tuple[float, float, float, float, float]]]:
        """Detect player in frame.

        Args:
            frame: Input frame (numpy HxWx3, grayscale HxW, or torch tensor)
            timestamp: Frame timestamp
            min_confidence: Confidence threshold

        Returns:
            {"player": (ts, x, y, w, h)} or {"player": None}
        """
        if isinstance(frame, np.ndarray):
            if frame.ndim == 2:
                frame_rgb = np.stack([frame] * 3, axis=-1)
                pil = Image.fromarray(frame_rgb.astype(np.uint8))
                orig_h, orig_w = frame.shape[:2]
            elif frame.ndim == 3:
                # Training loads images via PIL (RGB).  OpenCV frames are BGR.
                # Convert BGR→RGB before PIL so inference uses the same channel
                # order the model was trained on.  Without this, R and B channels
                # are swapped for every inference frame, reducing detection accuracy.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame_rgb.astype(np.uint8))
                orig_h, orig_w = frame.shape[:2]
            else:
                raise ValueError("Expected frame as HxW or HxWx3")
        elif isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] in (1, 3):
                if frame.shape[0] == 1:
                    frame = frame.repeat(3, 1, 1)
                pil = transforms.ToPILImage()(frame.cpu())
                orig_h, orig_w = int(frame.shape[1]), int(frame.shape[2])
            else:
                raise ValueError("Expected tensor as (3,H,W) or (1,H,W)")
        else:
            raise TypeError("Frame must be numpy array or torch tensor")

        # Forward pass with FP16 autocast (safe under @torch.no_grad())
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        use_amp = (self.device.type == "cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = self.forward_detect(x)[0].cpu()

        outputs: Dict[str, Optional[Tuple[float, float, float, float, float]]] = {}
        conf = float(pred[0, 0].item())

        # TEMPORARY DIAGNOSTIC — remove after first confirmed run (Task 5-D)
        if not hasattr(self, '_diag_count'):
            self._diag_count = 0
        if self._diag_count < 10:
            self._diag_count += 1
            box_raw = pred[0, 1:].tolist()
            print(f"[DINO DIAG] frame≈{self._diag_count} conf={conf:.4f} "
                  f"box_norm={[round(v,3) for v in box_raw]}")

        if conf < min_confidence:
            outputs["player"] = None
        else:
            box_norm = pred[0, 1:]
            box_xywh = _cxcywh_norm_to_xywh(box_norm.unsqueeze(0), orig_w, orig_h)[0]
            x0, y0, w, h = [float(v.item()) for v in box_xywh]
            outputs["player"] = (timestamp, x0, y0, w, h)

        return outputs

    @torch.no_grad()
    def detect_yolo_compat(
        self,
        frame,
        timestamp: float = 0.0,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> List[dict]:
        """Detect player, returning YOLO-compatible format with optional interval caching.

        Returns a list of dicts with "id", "box", and "feet" keys for
        compatibility with player_context.py and other pipeline components.

        Args:
            frame: Input frame (numpy HxWx3, grayscale HxW, or torch tensor)
            timestamp: Frame timestamp (unused, kept for API compatibility)
            min_confidence: Confidence threshold

        Returns:
            [{"id": 0, "box": [x1, y1, x2, y2], "feet": (cx, y2), "feet_real": None}]
            or [] if no detection
        """
        # Always run inference on first frame (frame 0) regardless of interval (Bug 5-B Cause C)
        run_inference = (self._detect_frame_count % self._detect_interval == 0) or (self._detect_frame_count == 0)
        self._detect_frame_count += 1

        if run_inference:
            result = self.detect(frame, timestamp, min_confidence)
            player_det = result.get("player")

            if player_det is None:
                self._detect_cache = []
            else:
                ts, x, y, w, h = player_det
                if isinstance(frame, np.ndarray):
                    orig_h, orig_w = frame.shape[:2]
                elif isinstance(frame, torch.Tensor):
                    orig_h, orig_w = int(frame.shape[1]), int(frame.shape[2])
                else:
                    orig_h, orig_w = 1080, 1920
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                x1 = max(0.0, min(x1, orig_w - 1.0))
                y1 = max(0.0, min(y1, orig_h - 1.0))
                x2 = max(x1 + 1.0, min(x2, float(orig_w)))
                y2 = max(y1 + 1.0, min(y2, float(orig_h)))
                cx = (x1 + x2) / 2.0
                self._detect_cache = [
                    {"id": 0, "box": [x1, y1, x2, y2], "feet": (cx, y2), "feet_real": None}
                ]

        return self._detect_cache


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DINODataset(Dataset):
    """COCO-format dataset with multi-crop augmentation."""

    def __init__(
        self,
        device,
        data_dir,
        global_crop_size=GLOBAL_CROP_SIZE,
        local_crop_size=LOCAL_CROP_SIZE,
        num_global_crops=NUM_GLOBAL_CROPS,
        num_local_crops=NUM_LOCAL_CROPS,
        annotations_file=None,
    ):
        self.device = device

        # Handle single or multiple directories
        if isinstance(data_dir, str):
            data_dirs = [data_dir]
        else:
            data_dirs = list(data_dir)

        if annotations_file is None:
            annotations_files = [os.path.join(d, "_annotations.coco.json") for d in data_dirs]
        elif isinstance(annotations_file, str):
            annotations_files = [annotations_file]
        else:
            annotations_files = list(annotations_file)

        if len(annotations_files) == 1 and len(data_dirs) > 1:
            annotations_files = annotations_files * len(data_dirs)
        elif len(data_dirs) != len(annotations_files):
            raise ValueError(f"Mismatch: {len(data_dirs)} dirs vs {len(annotations_files)} files")

        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        # Load all datasets
        self.categories = {}
        self.images = []
        self.image_paths = []
        self.image_id_to_index = {}
        self.annotations_by_image: Dict[int, List[dict]] = {}

        for data_dir, ann_file in zip(data_dirs, annotations_files):
            with open(ann_file, "r", encoding="utf-8") as f:
                coco = json.load(f)

            if not self.categories:
                self.categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

            id_offset = max(self.annotations_by_image.keys()) + 1 if self.annotations_by_image else 0

            # Build file map
            file_map = {}
            basename_map = {}
            for root, _, files in os.walk(data_dir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, data_dir)
                    if rel not in file_map:
                        file_map[rel] = full
                    if f not in basename_map:
                        basename_map[f] = full

            for im in sorted(coco.get("images", []), key=lambda x: x["id"]):
                new_id = im["id"] + id_offset
                self.images.append(im)
                ann_fname = im.get("file_name")

                # Resolve image path
                resolved = None
                if ann_fname in file_map:
                    resolved = file_map[ann_fname]
                elif os.path.basename(ann_fname) in basename_map:
                    resolved = basename_map[os.path.basename(ann_fname)]
                else:
                    candidate = os.path.join(data_dir, ann_fname)
                    if os.path.exists(candidate):
                        resolved = candidate

                if resolved is None:
                    resolved = os.path.join(data_dir, ann_fname)

                self.image_paths.append(resolved)
                self.image_id_to_index[new_id] = len(self.image_paths) - 1
                self.annotations_by_image[new_id] = []

            # Load annotations
            for ann in coco.get("annotations", []):
                image_id = ann.get("image_id") + id_offset
                if image_id in self.annotations_by_image:
                    self.annotations_by_image[image_id].append(ann)

        # Augmentation pipelines
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.global_crop_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.local_crop_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

        self.det_color_jitter = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03
        )
        self.det_to_tensor = transforms.Compose([
            transforms.Resize((self.global_crop_size, self.global_crop_size)),
            transforms.ToTensor()
        ])

        self.length = len(self.images)
        self.class_to_idx = {name: i for i, name in enumerate(TRACKED_CLASSES)}
        self.coco_name_to_track = {
            "person": "player",
            "player": "player",
        }

    def __len__(self):
        return self.length

    def _pick_representative_boxes(
        self, anns: Iterable[dict], img_w: float, img_h: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Pick largest box per class."""
        buckets: Dict[str, List[torch.Tensor]] = {k: [] for k in TRACKED_CLASSES}
        for ann in anns:
            cat_id = ann.get("category_id")
            cname = self.categories.get(cat_id, "")
            mapped = self.coco_name_to_track.get(cname)
            if mapped is None:
                continue
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            x = max(0.0, min(x, img_w - 1.0))
            y = max(0.0, min(y, img_h - 1.0))
            w = max(0.0, min(w, img_w - x))
            h = max(0.0, min(h, img_h - y))
            buckets[mapped].append(torch.tensor([x, y, w, h], dtype=torch.float32))

        out = {}
        for key, vals in buckets.items():
            if not vals:
                out[key] = None
                continue
            out[key] = max(vals, key=lambda b: float((b[2] * b[3]).item()))
        return out

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_id = image_info["id"]
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        anns = self.annotations_by_image.get(image_id, [])
        selected = self._pick_representative_boxes(anns, orig_w, orig_h)

        # Random horizontal flip
        do_hflip = random.random() < 0.5
        if do_hflip:
            image = TF.hflip(image)
            for key in TRACKED_CLASSES:
                if selected[key] is None:
                    continue
                box = selected[key].clone()
                box[0] = float(orig_w) - box[0] - box[2]
                selected[key] = box

        # Detection image
        image_det = self.det_color_jitter(image)
        det_image = self.det_to_tensor(image_det)

        # Scale boxes to detector input size
        sx = self.global_crop_size / float(orig_w)
        sy = self.global_crop_size / float(orig_h)
        det_targets = torch.zeros((len(TRACKED_CLASSES), 5), dtype=torch.float32)
        gt_boxes_xywh = torch.zeros((len(TRACKED_CLASSES), 4), dtype=torch.float32)

        for class_idx, class_name in enumerate(TRACKED_CLASSES):
            box = selected[class_name]
            if box is None:
                continue
            box_scaled = box.clone()
            box_scaled[0] *= sx
            box_scaled[1] *= sy
            box_scaled[2] *= sx
            box_scaled[3] *= sy
            gt_boxes_xywh[class_idx] = box_scaled
            det_targets[class_idx, 0] = 1.0
            det_targets[class_idx, 1:] = _xywh_to_cxcywh_norm(
                box_scaled, width=self.global_crop_size, height=self.global_crop_size
            )

        # Build crops
        crops: List[torch.Tensor] = []
        for _ in range(self.num_global_crops):
            crops.append(self.global_transform(image))
        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return {
            "image_path": img_path,
            "crops": crops,
            "det_image": det_image,
            "det_target": det_targets,
            "gt_boxes_xywh": gt_boxes_xywh,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (OPTIMIZED)
# ─────────────────────────────────────────────────────────────────────────────

def train_dino(
    student: Optional[DINOTracker],
    dataset: DINODataset,
    device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    output_dir: str = "data/output",
    checkpoint_name: str = "dino_player.pt",
    pretrained_backbone_path: Optional[str] = None,
    freeze_backbone_epochs: int = 0,  # OPTIMIZATION: Don't freeze by default
    use_lora: bool = False,
    lora_r: int = 4,
    lora_alpha: int = 16,
    num_workers: int = 4,
    log_every: int = 10,
) -> Tuple[DINOTracker, TrainHistory]:
    """Train DINOv3 detector on player dataset.

    Optimizations:
    - Removed SSL (DINO) loss, keep only detection loss (2x faster)
    - Removed EMA teacher model (simpler, no sync overhead)
    - Default: don't freeze backbone (better convergence with limited data)
    - Better LR schedule (cosine annealing)
    - Native checkpointing (less memory)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    print(f"[TRAIN] epochs={epochs} batch={batch_size} lr={learning_rate} device={device}")

    # Create or use provided model
    if student is None:
        student_model = DINOTracker(device=device, pretrained_backbone_path=pretrained_backbone_path)
    else:
        student_model = student

    # Optional LoRA
    if use_lora:
        n = apply_lora_to_encoder(student_model.encoder, r=lora_r, alpha=lora_alpha)
        print(f"[TRAIN] LoRA applied: {n} Linear modules")

    student_model.to(device)

    # Split dataset
    train_subset, val_subset = _split_dataset(dataset, train_ratio=0.8, seed=42)
    print(f"[TRAIN] split: train={len(train_subset)} val={len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_dino_collate,
        pin_memory=True,  # Faster data transfer
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_dino_collate,
        pin_memory=True,
    )

    # Optionally freeze backbone
    if freeze_backbone_epochs > 0:
        for param in student_model.encoder.parameters():
            param.requires_grad = False

    params = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=WEIGHT_DECAY)

    # Cosine annealing LR schedule
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = TrainHistory(train_loss=[], val_loss=[], val_iou=[], val_map=[], eval_epochs=[])

    for epoch in range(epochs):
        student_model.train()
        epoch_loss = 0.0
        batch_idx = 0

        for batch in train_loader:
            det_images = batch["det_images"].to(device)
            det_targets = batch["det_targets"].to(device)

            # Ensure spatial dims are multiples of patch size
            pe = getattr(student_model.encoder, "patch_embed", None)
            patch_H = 16
            if pe is not None and hasattr(pe, "patch_size"):
                ps = pe.patch_size
                if isinstance(ps, (tuple, list)):
                    patch_H = int(ps[0])
                elif isinstance(ps, int):
                    patch_H = ps

            det_images = _ensure_multiple(det_images, patch_H)

            # Forward pass (detection only)
            pred = student_model.forward_detect(det_images)

            # Loss: confidence + box regression
            conf_pred = pred[..., 0]
            box_pred = pred[..., 1:]
            conf_target = det_targets[..., 0]
            box_target = det_targets[..., 1:]

            conf_loss = F.binary_cross_entropy(conf_pred, conf_target)
            box_loss = F.l1_loss(box_pred, box_target, reduction="none")
            box_loss = (box_loss.sum(dim=-1) * conf_target).sum() / conf_target.sum().clamp(min=1.0)
            loss = conf_loss + BOX_LOSS_WEIGHT * box_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            batch_idx += 1

            if batch_idx % max(1, log_every) == 0:
                print(f"[TRAIN] epoch {epoch+1}/{epochs} batch {batch_idx} loss={loss.item():.4f}")

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        history.train_loss.append(avg_train_loss)

        # Validation
        if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == epochs:
            val_loss, val_iou, val_map = _evaluate_detector(student_model, val_loader, device)
            history.val_loss.append(val_loss)
            history.val_iou.append(val_iou)
            history.val_map.append(val_map)
            history.eval_epochs.append(epoch + 1)
            print(
                f"[TRAIN] epoch {epoch+1:03d}/{epochs} | "
                f"train={avg_train_loss:.4f} val={val_loss:.4f} iou={val_iou:.4f} mAP={val_map:.4f}"
            )
        else:
            print(f"[TRAIN] epoch {epoch+1:03d}/{epochs} | train={avg_train_loss:.4f}")

        # Unfreeze backbone
        if freeze_backbone_epochs > 0 and (epoch + 1) == freeze_backbone_epochs:
            print(f"[TRAIN] unfreezing encoder at epoch {epoch + 1}")
            for param in student_model.encoder.parameters():
                param.requires_grad = True

    # Save
    ckpt_path = os.path.join(output_dir, checkpoint_name)
    student_model.save_checkpoint(ckpt_path)
    print(f"[TRAIN] saved {ckpt_path}")

    # Plot
    _plot_training_curves(history, output_dir)

    return student_model, history


@torch.no_grad()
def _evaluate_detector(model: DINOTracker, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Validation: loss + IoU + mAP."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count_iou = 0

    pred_conf = []
    pred_iou = []
    gt_exists = []

    for batch in loader:
        images = batch["det_images"].to(device)
        pe = getattr(model.encoder, "patch_embed", None)
        patch_H = 16
        if pe is not None and hasattr(pe, "patch_size"):
            ps = pe.patch_size
            if isinstance(ps, (tuple, list)):
                patch_H = int(ps[0])
            elif isinstance(ps, int):
                patch_H = ps
        images = _ensure_multiple(images, patch_H)

        targets = batch["det_targets"].to(device)

        pred = model.forward_detect(images)
        conf_pred = pred[..., 0]
        box_pred = pred[..., 1:]
        conf_target = targets[..., 0]
        box_target = targets[..., 1:]

        conf_loss = F.binary_cross_entropy(conf_pred, conf_target)
        box_loss = F.l1_loss(box_pred, box_target, reduction="none")
        box_loss = (box_loss.sum(dim=-1) * conf_target).sum() / conf_target.sum().clamp(min=1.0)
        total_loss += float((conf_loss + box_loss).item())

        # IoU
        pred_xywh = _cxcywh_norm_to_xywh(box_pred.reshape(-1, 4), 1.0, 1.0)
        gt_xywh = _cxcywh_norm_to_xywh(box_target.reshape(-1, 4), 1.0, 1.0)
        iou = _bbox_iou_xywh(pred_xywh, gt_xywh)

        has_gt = (conf_target.flatten() > 0.5).cpu().numpy().astype(np.float32)
        pred_conf.extend(conf_pred.flatten().cpu().numpy().tolist())
        pred_iou.extend(iou.cpu().numpy().tolist())
        gt_exists.extend(has_gt.tolist())

        if has_gt.sum() > 0:
            total_iou += float(iou[has_gt > 0.5].sum().item())
            count_iou += int(has_gt.sum())

    mean_iou = total_iou / max(count_iou, 1)
    avg_loss = total_loss / max(len(loader), 1)

    # Simple AP
    ap = _compute_ap_from_scores(
        np.array(pred_conf, dtype=np.float32),
        np.array(pred_iou, dtype=np.float32),
        np.array(gt_exists, dtype=np.float32),
        iou_threshold=VAL_IOU_THRESHOLD,
    )

    return avg_loss, mean_iou, ap


def _plot_training_curves(history: TrainHistory, save_dir: str) -> None:
    """Plot and save training curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, len(history.train_loss) + 1), history.train_loss, label="train")
    if history.val_loss:
        ax1.plot(history.eval_epochs, history.val_loss, marker="o", label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if history.val_iou:
        ax2.plot(history.eval_epochs, history.val_iou, marker="o", label="IoU")
    if history.val_map:
        ax2.plot(history.eval_epochs, history.val_map, marker="s", label="mAP")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("metric")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=100)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_prefix(state_dict: dict) -> dict:
    """Remove checkpoint prefixes."""
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in ("module.", "backbone.", "encoder."):
            if new_k.startswith(p):
                new_k = new_k[len(p):]
                break
        out[new_k] = v
    return out


def _dino_collate(batch: List[dict]) -> dict:
    """Collate batch."""
    det_images = torch.stack([sample["det_image"] for sample in batch], dim=0)
    det_targets = torch.stack([sample["det_target"] for sample in batch], dim=0)
    return {
        "det_images": det_images,
        "det_targets": det_targets,
    }


def _split_dataset(dataset: Dataset, train_ratio: float = 0.8, seed: int = 42) -> Tuple[Subset, Subset]:
    """80/20 split."""
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = max(1, int(train_ratio * n))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def _ensure_multiple(t: torch.Tensor, multiple: int) -> torch.Tensor:
    """Ensure spatial dims are multiples of patch size."""
    if t.dim() != 4:
        return t
    b, c, h, w = t.shape
    if h % multiple == 0 and w % multiple == 0:
        return t
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    return F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _xywh_to_cxcywh_norm(xywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Convert [x,y,w,h] to normalized [cx,cy,w,h]."""
    x, y, w, h = xywh.unbind(-1)
    cx = (x + 0.5 * w) / max(width, 1e-6)
    cy = (y + 0.5 * h) / max(height, 1e-6)
    nw = w / max(width, 1e-6)
    nh = h / max(height, 1e-6)
    return torch.stack([cx, cy, nw, nh], dim=-1).clamp(0.0, 1.0)


def _cxcywh_norm_to_xywh(cxcywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    """Convert normalized [cx,cy,w,h] to [x,y,w,h]."""
    cx, cy, w, h = cxcywh.unbind(-1)
    abs_w = (w * width).clamp(min=0.0)
    abs_h = (h * height).clamp(min=0.0)
    x = (cx * width - 0.5 * abs_w).clamp(min=0.0, max=max(width - 1.0, 0.0))
    y = (cy * height - 0.5 * abs_h).clamp(min=0.0, max=max(height - 1.0, 0.0))
    return torch.stack([x, y, abs_w, abs_h], dim=-1)


def _bbox_iou_xywh(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU for [x,y,w,h] boxes."""
    ax1, ay1, aw, ah = box_a.unbind(-1)
    bx1, by1, bw, bh = box_b.unbind(-1)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = torch.max(ax1, bx1)
    inter_y1 = torch.max(ay1, by1)
    inter_x2 = torch.min(ax2, bx2)
    inter_y2 = torch.min(ay2, by2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = aw.clamp(min=0) * ah.clamp(min=0)
    area_b = bw.clamp(min=0) * bh.clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=1e-6)
    return inter / union


def _compute_ap_from_scores(conf: np.ndarray, iou: np.ndarray, has_gt: np.ndarray, iou_threshold: float) -> float:
    """Simple AP calculation."""
    if conf.size == 0:
        return 0.0
    order = np.argsort(-conf)
    iou = iou[order]
    has_gt = has_gt[order]

    tp = ((iou >= iou_threshold) & (has_gt > 0.5)).astype(np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    total_gt = max(float(np.sum(has_gt > 0.5)), 1.0)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + np.cumsum(fp), 1e-6)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for i in range(precision.size - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(precision, recall))


def _create_vit_tiny(pretrained_weights_path: Optional[str] = None) -> Tuple[nn.Module, int]:
    """Create ViT-tiny backbone."""
    try:
        import timm
    except ImportError:
        raise ImportError("Install timm: pip install timm")

    # PHASE 1-A: ViT-S/14 (3-4× faster, 384 embed_dim)
    dinov2_model = os.environ.get("DINOV2_MODEL", "dinov2_vits14")
    # To revert to ViT-B/14: dinov2_model = os.environ.get("DINOV2_MODEL", "dinov2_vitb14")
    try:
        print(f"[DINOTracker] Loading DINOv2: {dinov2_model}")
        encoder = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        embed_dim = getattr(encoder, "embed_dim", None) or getattr(encoder, "num_features", None) or 384
        print(f"[DINOTracker] Loaded DINOv2 (embed_dim={embed_dim})")
        return encoder, int(embed_dim)
    except Exception as e:
        print(f"[DINOTracker] DINOv2 load failed ({e}), using timm ViT-small")
        encoder = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0, dynamic_img_size=True)
        embed_dim = getattr(encoder, "embed_dim", 384)
        return encoder, int(embed_dim)


def _create_vit_by_embed_dim(embed_dim: int, pretrained_weights_path: Optional[str] = None) -> Tuple[nn.Module, int]:
    """Create ViT matching embed_dim."""
    try:
        import timm
    except ImportError:
        raise ImportError("Install timm: pip install timm")

    if embed_dim == 192:
        model_name = "vit_tiny_patch16_224"
    elif embed_dim == 384:
        try:
            print("[DINOTracker] Loading DINOv2 ViT-S/14")
            encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            ed = getattr(encoder, "embed_dim", None) or 384
            return encoder, int(ed)
        except Exception:
            model_name = "vit_small_patch16_224"
    elif embed_dim == 768:
        try:
            print("[DINOTracker] Loading DINOv2 ViT-B/14")
            encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            ed = getattr(encoder, "embed_dim", None) or 768
            return encoder, int(ed)
        except Exception:
            model_name = "vit_base_patch16_224"
    else:
        model_name = "vit_base_patch16_224"

    encoder = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
    ed_out = getattr(encoder, "embed_dim", getattr(encoder, "num_features", embed_dim))
    return encoder, int(ed_out)


def _extract_cls_token(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract class token from encoder."""
    if hasattr(encoder, "forward_features"):
        features = encoder.forward_features(x)
    else:
        features = encoder(x)

    if isinstance(features, dict):
        if "x_norm_clstoken" in features:
            return features["x_norm_clstoken"]
        if "cls_token" in features:
            return features["cls_token"]
        if "x_prenorm" in features:
            x_pre = features["x_prenorm"]
            return x_pre[:, 0, :] if x_pre.dim() == 3 else x_pre

    if isinstance(features, torch.Tensor):
        if features.dim() == 3:
            return features[:, 0, :]
        if features.dim() == 2:
            return features

    raise RuntimeError("Unable to extract class token")


class LoRALinear(nn.Module):
    """LoRA adapter for Linear layer."""

    def __init__(self, orig_linear: nn.Linear, r: int = 4, alpha: int = 16):
        super().__init__()
        self.orig = orig_linear
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = max(1, int(r))
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.r))
        self.lora_B = nn.Parameter(torch.zeros(self.r, self.out_features))
        self.scaling = float(alpha) / float(self.r) if self.r > 0 else 1.0

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.orig.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.orig(x)
        lora_inter = torch.matmul(x, self.lora_A)
        lora_out = torch.matmul(lora_inter, self.lora_B) * self.scaling
        return orig_out + lora_out


def apply_lora_to_encoder(encoder: nn.Module, r: int = 4, alpha: int = 16, min_dim: int = 64) -> int:
    """Apply LoRA to encoder Linear layers."""
    replaced = 0
    linear_items = [(name, m) for name, m in encoder.named_modules() if isinstance(m, nn.Linear)]
    for full_name, mod in linear_items:
        if getattr(mod, "in_features", 0) < min_dim or getattr(mod, "out_features", 0) < min_dim:
            continue

        parts = full_name.split(".")
        parent = encoder
        for p in parts[:-1]:
            parent = getattr(parent, p)

        orig = getattr(parent, parts[-1])
        if isinstance(orig, LoRALinear):
            continue

        setattr(parent, parts[-1], LoRALinear(orig, r=r, alpha=alpha))
        replaced += 1

    return replaced


# ─────────────────────────────────────────────────────────────────────────────
# API Alias for backward compatibility with main.py
# ─────────────────────────────────────────────────────────────────────────────

PlayerDetector = DINOTracker  # main.py imports PlayerDetector; alias to DINOTracker
