"""
shadow_prior_car.py
Direction A-2: car-shadow geometric shadow prior

The car-shadow DAVIS sequence has a car with a prominent shadow. 
SAM3 text prompts focus on "car" and miss the shadow (JM drops from 0.975 to 0.891).
This script extends any car body mask with a parametric shadow region to
recover the shadow coverage.

Strategy:
  1. Load car body mask (from SAM3 or GDINO+SAM2)
  2. Estimate car bounding box + shadow direction from image structure
  3. Project a parametric shadow ellipse below/beside the car
  4. Union car + shadow
  5. Sigma ablation: test shadow_scale values [0.5, 0.8, 1.0, 1.2, 1.5]

Usage:
  python part3/shadow_prior_car.py \
    --input_dir /data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/car-shadow \
    --image_dir /home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/car-shadow \
    --gt_dir /home/jli657/shared_data/project3/DAVIS/Annotations/480p/car-shadow \
    --output_root /data3/jli657/project3/part3/outputs/direction_a/shadow_geom/car-shadow
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Shadow estimation from image (heuristic sun-direction)
# ---------------------------------------------------------------------------

def estimate_shadow_params(image: np.ndarray, car_mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate shadow direction and length from the image.
    
    Looks for the dark region adjacent to the car mask (the shadow region in the image).
    Returns: (shadow_angle_deg, shadow_scale, shadow_offset_x_frac)
    """
    if car_mask.sum() == 0:
        return 270.0, 0.8, 0.0  # default: shadow points right

    rows, cols = np.where(car_mask > 0)
    car_bottom = int(rows.max())
    car_top = int(rows.min())
    car_left = int(cols.min())
    car_right = int(cols.max())
    car_cx = int((car_left + car_right) / 2)
    car_h = car_bottom - car_top
    car_w = car_right - car_left

    # Look for dark (shadow) region in the area below/beside the car
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Sample a region below the car
    search_h = min(car_h, H - car_bottom - 1)
    if search_h <= 0:
        return 270.0, 0.8, 0.0

    below_region = gray[car_bottom:car_bottom + search_h, max(0, car_left-car_w//2):min(W, car_right+car_w//2)]
    if below_region.size == 0:
        return 270.0, 0.8, 0.0

    # Find darkest column in region (likely where shadow is)
    col_means = below_region.mean(axis=0)
    shadow_col_local = int(np.argmin(col_means))
    shadow_region_x_start = max(0, car_left - car_w // 2)
    shadow_cx = shadow_region_x_start + shadow_col_local

    # Shadow offset relative to car center
    offset_frac = (shadow_cx - car_cx) / (car_w + 1e-6)

    return 270.0, 0.8, float(offset_frac)


def extend_car_mask_with_shadow(
    car_mask: np.ndarray,
    image: np.ndarray,
    shadow_scale: float = 1.0,
    shadow_direction_deg: float = 270.0,
    auto_detect_shadow: bool = True,
    shadow_threshold: float = 48.0,
) -> np.ndarray:
    """
    Extend car mask to include shadow region.

    Parameters:
    - shadow_scale: multiplier for shadow size relative to car
    - shadow_direction_deg: angle of shadow (270 = below, 315 = below-right, etc.)
    - auto_detect_shadow: use image to detect actual shadow position
    """
    H, W = car_mask.shape
    nz = np.nonzero(car_mask)
    if len(nz[0]) == 0:
        return car_mask.copy()

    rows, cols = nz
    car_top = int(rows.min())
    car_bottom = int(rows.max())
    car_left = int(cols.min())
    car_right = int(cols.max())
    car_h = car_bottom - car_top + 1
    car_w = car_right - car_left + 1
    car_cx = (car_left + car_right) // 2
    car_cy = (car_top + car_bottom) // 2

    # Shadow ellipse axes
    shadow_semi_major = int(car_w * 0.65 * shadow_scale)  # horizontal
    shadow_semi_minor = int(car_h * 0.35 * shadow_scale)  # vertical

    if auto_detect_shadow and image is not None:
        _, auto_scale, offset_frac = estimate_shadow_params(image, car_mask)
        shadow_scale_eff = shadow_scale * auto_scale
        shadow_semi_major = int(car_w * 0.65 * shadow_scale_eff)
        shadow_semi_minor = int(car_h * 0.35 * shadow_scale_eff)
    else:
        offset_frac = 0.1  # slight offset to the right by default for car-shadow

    # Shadow center: below the car, slightly offset
    direction_rad = np.radians(shadow_direction_deg)
    shadow_cx = int(car_cx + car_w * offset_frac)
    shadow_cy = int(car_bottom + shadow_semi_minor * 0.6)  # just below the car

    # Clamp to image bounds
    shadow_cx = max(shadow_semi_major, min(W - shadow_semi_major, shadow_cx))
    shadow_cy = max(shadow_semi_minor, min(H - shadow_semi_minor, shadow_cy))

    # Draw shadow ellipse
    shadow_canvas = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(
        shadow_canvas,
        center=(shadow_cx, shadow_cy),
        axes=(shadow_semi_major, shadow_semi_minor),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    # Only keep shadow BELOW the car body (not above)
    shadow_canvas[:car_bottom, :] = 0

    # Soften edges
    shadow_blurred = cv2.GaussianBlur(shadow_canvas, (0, 0), sigmaX=8.0, sigmaY=4.0)
    shadow_binary = (shadow_blurred > shadow_threshold).astype(np.uint8) * 255

    # Union: car mask + shadow
    extended = np.maximum(car_mask, shadow_binary)
    return extended


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_jm(pred_dir: Path, gt_dir: Path) -> float:
    if not gt_dir.exists():
        return -1.0
    ious = []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = np.array(Image.open(gt_path).convert("L")) > 0
        pred = np.array(Image.open(pred_path).convert("L")) > 127
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        ious.append(inter / (union + 1e-8))
    return float(np.mean(ious)) if ious else 0.0


def compute_jm_from_dict(pred_masks: Dict[str, np.ndarray], gt_dir: Path) -> float:
    if not gt_dir.exists():
        return -1.0
    ious = []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        if gt_path.stem not in pred_masks:
            continue
        gt = np.array(Image.open(gt_path).convert("L")) > 0
        pred = pred_masks[gt_path.stem].astype(bool)
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        ious.append(inter / (union + 1e-8))
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Direction A-2: car-shadow geometric shadow prior")
    p.add_argument("--input_dir",
                   default="/data3/jli657/project3/part3/gdino_vlm/masks/stage1/car-shadow",
                   help="Input car mask directory (best available: GDINO+SAM2 or SAM3)")
    p.add_argument("--image_dir",
                   default="/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/car-shadow")
    p.add_argument("--gt_dir",
                   default="/home/jli657/shared_data/project3/DAVIS/Annotations/480p/car-shadow")
    p.add_argument("--output_root",
                   default="/data3/jli657/project3/part3/outputs/direction_a/shadow_geom/car-shadow")
    p.add_argument("--shadow_scales", nargs="+", type=float,
                   default=[0.5, 0.8, 1.0, 1.2, 1.5],
                   help="Shadow scale values for ablation")
    p.add_argument("--auto_detect", action="store_true", default=True,
                   help="Auto-detect shadow direction from image")
    p.add_argument("--out_csv",
                   default="/home/jli657/my_storage2_1T/project3/eval/direction_a_shadow_ablation.csv")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    output_root = Path(args.output_root)

    # Load all masks
    mask_files = sorted(input_dir.glob("*.png"), key=lambda p: p.stem)
    if not mask_files:
        print(f"[error] No masks found in {input_dir}")
        return

    # Baseline JM
    baseline_jm = compute_jm(input_dir, gt_dir)
    print(f"Baseline JM (input masks): {baseline_jm:.4f}")

    ablation_rows = []
    best_jm = baseline_jm
    best_scale = None
    best_dir = None

    for scale in args.shadow_scales:
        out_dir = output_root / f"scale_{scale:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)

        extended_masks: Dict[str, np.ndarray] = {}
        for mpath in mask_files:
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = (mask > 127).astype(np.uint8) * 255

            # Load corresponding image
            img_path = image_dir / mpath.name
            if not img_path.exists():
                img_path = image_dir / (mpath.stem + ".jpg")
            image = cv2.imread(str(img_path)) if img_path.exists() else None

            extended = extend_car_mask_with_shadow(
                mask, image,
                shadow_scale=scale,
                auto_detect_shadow=args.auto_detect,
            )
            Image.fromarray(extended).save(str(out_dir / mpath.name))
            extended_masks[mpath.stem] = extended

        jm = compute_jm_from_dict(extended_masks, gt_dir)
        print(f"  scale={scale:.2f} -> JM={jm:.4f} (baseline={baseline_jm:.4f})")

        row = {"shadow_scale": scale, "JM": jm, "delta_JM": jm - baseline_jm, "output_dir": str(out_dir)}
        ablation_rows.append(row)

        if jm > best_jm:
            best_jm = jm
            best_scale = scale
            best_dir = out_dir

    # Save best as final output
    if best_dir is not None:
        final_dir = output_root / "best"
        final_dir.mkdir(parents=True, exist_ok=True)
        for f in best_dir.glob("*.png"):
            import shutil
            shutil.copy(str(f), str(final_dir / f.name))
        print(f"\nBest scale={best_scale:.2f} JM={best_jm:.4f} -> {final_dir}")
    else:
        print(f"\nNo improvement over baseline ({baseline_jm:.4f}), keeping original")

    # Save ablation CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shadow_scale", "JM", "delta_JM", "output_dir"])
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(f"[saved] {out_csv}")

    # Summary
    print("\n=== Shadow Prior Ablation (car-shadow) ===")
    print(f"{'Scale':>8} {'JM':>10} {'Delta':>10}")
    print(f"{'baseline':>8} {baseline_jm:>10.4f}")
    for r in ablation_rows:
        print(f"  {r['shadow_scale']:>6.2f} {r['JM']:>10.4f} {r['delta_JM']:>+10.4f}")


if __name__ == "__main__":
    main()
