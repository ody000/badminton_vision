#!/usr/bin/env python3
"""
DINO Diagnostic: Run inference on random frames and visualize detections.

Samples random frames from the training dataset, runs DINO player detection,
draws bounding boxes, and saves annotated images for visual inspection.

Usage:
  python tools/dino_diagnostic.py --num-frames 100 --output-dir data/output/dino_diag
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

# Add repo root to path so imports work from tools/ subdirectory
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.player_dino import DINOTracker, TRACKED_CLASSES


def load_coco_dataset(annotations_file):
    """Load COCO dataset structure."""
    with open(annotations_file, 'r') as f:
        coco = json.load(f)
    
    # Map image_id to image info
    images_by_id = {img['id']: img for img in coco['images']}
    
    # Map image_id to annotations
    anns_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
    
    # Map category_id to name
    categories = {c['id']: c['name'] for c in coco['categories']}
    
    return images_by_id, anns_by_image, categories


def draw_boxes_on_image(image, detections, ground_truth_boxes=None):
    """Draw DINO detections and ground truth on image.
    
    Args:
        image: PIL Image
        detections: list of (name, conf, x, y, w, h) normalized to [0, 1]
        ground_truth_boxes: dict of player_name -> (x, y, w, h) or None
    
    Returns:
        annotated PIL Image
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    w_px, h_px = image.size
    
    # Colors for each player
    colors = {
        'player_1': 'lime',      # Green
        'player_2': 'cyan',      # Cyan
    }
    
    # Draw DINO detections (solid boxes)
    for name, conf, cx_norm, cy_norm, w_norm, h_norm in detections:
        cx_px = int(cx_norm * w_px)
        cy_px = int(cy_norm * h_px)
        w_px_det = int(w_norm * w_px)
        h_px_det = int(h_norm * h_px)
        
        x1 = max(0, cx_px - w_px_det // 2)
        y1 = max(0, cy_px - h_px_det // 2)
        x2 = min(w_px, cx_px + w_px_det // 2)
        y2 = min(h_px, cy_px + h_px_det // 2)
        
        color = colors.get(name, 'white')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - 15), f"{name}\n{conf:.2f}", fill=color)
    
    # Draw ground truth boxes (dashed - we'll approximate with thinner lines)
    if ground_truth_boxes:
        gt_colors = {
            'player_1': 'orange',
            'player_2': 'magenta',
        }
        for name, (x, y, w, h) in ground_truth_boxes.items():
            if h <= 0 or w <= 0:
                continue
            x_px = int(x)
            y_px = int(y)
            w_px_gt = int(w)
            h_px_gt = int(h)
            x2 = min(w_px, x_px + w_px_gt)
            y2 = min(h_px, y_px + h_px_gt)
            
            color = gt_colors.get(name, 'yellow')
            draw.rectangle([x_px, y_px, x2, y2], outline=color, width=2)
            draw.text((x_px, y2 + 5), f"GT_{name}", fill=color)
    
    return img_copy


def main():
    parser = argparse.ArgumentParser(description='DINO diagnostic visualization')
    parser.add_argument('--num-frames', type=int, default=100, help='Number of random frames to process')
    parser.add_argument('--data-dir', type=str, default='data/input/train/player2', help='Training data directory')
    parser.add_argument('--output-dir', type=str, default='data/output/dino_diag', help='Output directory for visualizations')
    parser.add_argument('--checkpoint', type=str, default='models/dino_player.pt', help='Path to DINO checkpoint')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DINO_DIAG] Loading COCO dataset...")
    annotations_file = os.path.join(args.data_dir, '_annotations.coco.json')
    images_by_id, anns_by_image, categories = load_coco_dataset(annotations_file)
    
    print(f"[DINO_DIAG] Total images: {len(images_by_id)}")
    
    # Sample random frames
    image_ids = list(images_by_id.keys())
    sampled_ids = random.sample(image_ids, min(args.num_frames, len(image_ids)))
    print(f"[DINO_DIAG] Sampled {len(sampled_ids)} random frames")
    
    # Load DINO model
    print(f"[DINO_DIAG] Loading DINO from {args.checkpoint}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINOTracker(device=device)
    model.load_checkpoint(args.checkpoint)
    model.eval()
    
    print(f"[DINO_DIAG] Processing frames...")
    results = []
    
    for i, img_id in enumerate(sampled_ids):
        if (i + 1) % 10 == 0:
            print(f"[DINO_DIAG]   {i + 1}/{len(sampled_ids)}")
        
        img_info = images_by_id[img_id]
        img_path = os.path.join(args.data_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"[DINO_DIAG] WARNING: Image not found: {img_path}")
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        w_px, h_px = image.size
        
        # Resize to be divisible by patch size (14)
        patch_size = 14
        new_h = (h_px // patch_size) * patch_size
        new_w = (w_px // patch_size) * patch_size
        image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Run DINO inference
        with torch.no_grad():
            img_tensor = torch.tensor(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            pred = model.forward_detect(img_tensor)
        
        # Extract detections (still in resized space, we'll map back to original)
        detections = []
        scale_x = w_px / new_w
        scale_y = h_px / new_h
        
        for player_idx, player_name in enumerate(TRACKED_CLASSES):
            conf = float(pred[0, player_idx, 0].cpu())
            cx_norm = float(pred[0, player_idx, 1].cpu())
            cy_norm = float(pred[0, player_idx, 2].cpu())
            w_norm = float(pred[0, player_idx, 3].cpu())
            h_norm = float(pred[0, player_idx, 4].cpu())
            
            # Scale back to original image space
            cx_norm_orig = cx_norm * scale_x
            cy_norm_orig = cy_norm * scale_y
            w_norm_orig = w_norm * scale_x
            h_norm_orig = h_norm * scale_y
            
            # Clamp to [0, 1]
            cx_norm_orig = max(0.0, min(1.0, cx_norm_orig))
            cy_norm_orig = max(0.0, min(1.0, cy_norm_orig))
            w_norm_orig = max(0.0, min(1.0, w_norm_orig))
            h_norm_orig = max(0.0, min(1.0, h_norm_orig))
            
            detections.append((player_name, conf, cx_norm_orig, cy_norm_orig, w_norm_orig, h_norm_orig))
        
        # Get ground truth boxes
        gt_boxes = {}
        anns = anns_by_image.get(img_id, [])
        person_anns = [a for a in anns if categories.get(a['category_id']) == 'person']
        
        # Sort by area (largest first) to match training logic
        person_anns_sorted = sorted(person_anns, key=lambda a: a.get('area', 0), reverse=True)
        
        for player_idx, player_name in enumerate(TRACKED_CLASSES):
            if player_idx < len(person_anns_sorted):
                bbox = person_anns_sorted[player_idx].get('bbox', [])
                if bbox and len(bbox) == 4:
                    # Convert all bbox values to float (some may be strings in the JSON)
                    bbox_float = tuple(float(v) for v in bbox)
                    gt_boxes[player_name] = bbox_float
        
        # Draw boxes
        annotated = draw_boxes_on_image(image, detections, gt_boxes)
        
        # Save annotated image
        output_path = output_dir / f"frame_{i:04d}_id{img_id}.png"
        annotated.save(output_path)
        
        # Store result for summary
        results.append({
            'frame_idx': i,
            'image_id': img_id,
            'filename': img_info['file_name'],
            'detections': [(name, conf, cx, cy, w, h) for name, conf, cx, cy, w, h in detections],
            'has_gt': len(gt_boxes) > 0,
        })
    
    # Write summary CSV
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Frame Index | Image ID | Filename | Player_1 Conf | Player_2 Conf | GT Available\n")
        f.write("-" * 100 + "\n")
        for r in results:
            p1_conf = r['detections'][0][1] if len(r['detections']) > 0 else 0
            p2_conf = r['detections'][1][1] if len(r['detections']) > 1 else 0
            f.write(f"{r['frame_idx']:4d} | {r['image_id']:8d} | {r['filename']:50s} | {p1_conf:8.4f} | {p2_conf:8.4f} | {r['has_gt']}\n")
    
    print(f"\n[DINO_DIAG] Complete!")
    print(f"[DINO_DIAG] Output directory: {output_dir}")
    print(f"[DINO_DIAG] Annotated images: {len(results)} frames")
    print(f"[DINO_DIAG] Summary file: {summary_path}")
    
    # Print confidence statistics
    all_p1_confs = [r['detections'][0][1] for r in results]
    all_p2_confs = [r['detections'][1][1] for r in results]
    
    print(f"\n[DINO_DIAG] Confidence Statistics:")
    print(f"  Player_1: min={min(all_p1_confs):.4f}, max={max(all_p1_confs):.4f}, mean={np.mean(all_p1_confs):.4f}, std={np.std(all_p1_confs):.4f}")
    print(f"  Player_2: min={min(all_p2_confs):.4f}, max={max(all_p2_confs):.4f}, mean={np.mean(all_p2_confs):.4f}, std={np.std(all_p2_confs):.4f}")
    
    print(f"\n[DINO_DIAG] To view results:")
    print(f"  ls -lh {output_dir}/*.png | head -20")


if __name__ == '__main__':
    main()
