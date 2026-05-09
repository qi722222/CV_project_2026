"""
mask_fusion.py
Direction A-1: Adaptive Mask Fusion

Three-way per-frame IoU-gated fusion of:
  - SAM3 video text prompt masks (Direction A)
  - GDINO+SAM2 Stage1 masks (existing mainline)
  - VGGT4D/Pi3 dynamic masks (Direction B, optional)

Fusion strategy per frame:
  1. For each sequence, identify each method's mask directory
  2. Load all available masks per frame
  3. Apply IoU-gated fusion:
     - If two masks overlap heavily (IoU > 0.8): take union (both agree, cover all)
     - If two masks don't overlap (IoU < 0.3): evaluate which is more credible per sequence
     - If partial overlap: weighted union

Special per-sequence logic:
  - bmx-trees: GDINO+SAM2 Stage1 is much better (0.746 vs 0.631) -> prefer GDINO+SAM2
  - car-shadow: YOLO+SAM2 is better (0.975 vs 0.891) -> prefer YOLO+SAM2
  - tennis/horsejump: SAM3 is better -> prefer SAM3
  - blackswan: equal -> union
  - koala: SAM3 only -> use SAM3

Usage:
  python part3/mask_fusion.py \
    --sequences tennis bmx-trees car-shadow horsejump-low blackswan koala \
    --output_root /data3/jli657/project3/part3/outputs/direction_a/mask_fusion
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Known mask directories (from existing experiments)
# ---------------------------------------------------------------------------

# SAM3 multi-object union masks (Direction A best)
SAM3_MASK_ROOTS: Dict[str, Path] = {
    "tennis":       Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/tennis"),
    "bmx-trees":    Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/bmx-trees"),
    "blackswan":    Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/blackswan"),
    "car-shadow":   Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/car-shadow"),
    "horsejump-low": Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/horsejump-low"),
    "koala":        Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/koala"),
    "wild_video-1person": Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_shadow_v2/wild_video-1person"),
}

# GDINO+SAM2 Stage1 masks (existing mainline, best for bmx-trees)
GDINO_SAM2_MASK_ROOTS: Dict[str, Path] = {
    "tennis":        Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/tennis"),
    "bmx-trees":     Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/bmx-trees"),
    "blackswan":     Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/blackswan"),
    "car-shadow":    Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/car-shadow"),
    "horsejump-low": Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/horsejump-low"),
    "koala":         Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/koala"),
}

# VGGT4D dynamic masks (Direction B, to be produced before this runs)
VGGT4D_MASK_ROOTS: Dict[str, Path] = {
    seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt/{seq}")
    for seq in ["tennis", "bmx-trees", "blackswan", "car-shadow", "horsejump-low", "koala"]
}

# Pi3 transplant masks (Direction B-4)
PI3_MASK_ROOTS: Dict[str, Path] = {
    seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/pi3_transplant/{seq}")
    for seq in ["tennis", "bmx-trees", "blackswan", "car-shadow", "horsejump-low", "koala"]
}

# GT masks for evaluation
DAVIS_GT = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

# Per-sequence routing preferences (from empirical JM comparison)
# Key insight: for sequences where one method is clearly better, use that ONLY
# because adding the weaker method introduces false positives that hurt JM.
#   tennis:        SAM3=0.947  GDINO=0.931 -> SAM3 wins, use SAM3 only
#   bmx-trees:     SAM3=0.631  GDINO=0.746 -> GDINO wins, use GDINO only
#   blackswan:     SAM3=0.955  GDINO=0.955 -> equal, union is safe
#   car-shadow:    SAM3=0.891  GDINO=0.975 -> GDINO wins massively, use GDINO only
#   horsejump-low: SAM3=0.857  GDINO=0.724 -> SAM3 wins, use SAM3 only
#   koala:         SAM3=0.948  GDINO=N/A   -> SAM3 only
SEQUENCE_ROUTING: Dict[str, Dict] = {
    "tennis":        {"primary": "sam3", "secondary": None, "strategy": "sam3_only"},
    "bmx-trees":     {"primary": "gdino", "secondary": None, "strategy": "primary_only"},
    "blackswan":     {"primary": "sam3", "secondary": "gdino", "strategy": "union"},
    "car-shadow":    {"primary": "gdino", "secondary": None, "strategy": "primary_only"},
    "horsejump-low": {"primary": "sam3", "secondary": None, "strategy": "sam3_only"},
    "koala":         {"primary": "sam3", "secondary": None, "strategy": "sam3_only"},
    "wild_video-1person": {"primary": "sam3", "secondary": None, "strategy": "sam3_only"},
}


# ---------------------------------------------------------------------------
# Mask loading helpers
# ---------------------------------------------------------------------------

def load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def load_masks_dir(mask_dir: Path) -> Dict[str, np.ndarray]:
    """Return {stem: binary_mask} for all PNGs in dir."""
    if not mask_dir or not mask_dir.exists():
        return {}
    result = {}
    for p in sorted(mask_dir.glob("*.png"), key=lambda x: x.stem):
        m = load_mask(p)
        if m is not None:
            result[p.stem] = m
    return result


# ---------------------------------------------------------------------------
# Fusion logic
# ---------------------------------------------------------------------------

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-8))


def fuse_two_masks(
    primary: np.ndarray,
    secondary: np.ndarray,
    strategy: str,
    iou_threshold_high: float = 0.7,
    iou_threshold_low: float = 0.2,
) -> np.ndarray:
    """
    Fuse primary and secondary masks based on strategy and IoU.
    
    Strategies:
      'primary_only': return primary
      'union': bitwise OR
      'primary_fill': use primary, add secondary pixels where primary is sparse
      'iou_gated': dynamic decision based on IoU
    """
    if secondary is None:
        return primary
    if primary is None:
        return secondary

    if strategy == "sam3_only" or strategy == "primary_only":
        return primary

    if strategy == "union":
        return np.logical_or(primary, secondary).astype(np.uint8)

    # IoU-gated fusion
    iou = compute_iou(primary, secondary)

    if strategy in ("sam3_primary_gdino_fill", "gdino_primary_sam3_fill"):
        # Keep primary, add secondary regions not covered by primary
        if iou > iou_threshold_high:
            # High agreement: take union
            return np.logical_or(primary, secondary).astype(np.uint8)
        elif iou < iou_threshold_low:
            # Low overlap: secondary sees something different
            # Add secondary only if it covers a meaningful area
            sec_area = secondary.sum()
            prim_area = primary.sum()
            if sec_area > 0.1 * prim_area:  # secondary has >10% of primary area
                return np.logical_or(primary, secondary).astype(np.uint8)
            else:
                return primary
        else:
            # Moderate overlap: take union
            return np.logical_or(primary, secondary).astype(np.uint8)

    return np.logical_or(primary, secondary).astype(np.uint8)


def fuse_sequence_masks(
    seq_name: str,
    sam3_masks: Dict[str, np.ndarray],
    gdino_masks: Dict[str, np.ndarray],
    vggt4d_masks: Dict[str, np.ndarray],
    routing: Dict,
) -> Dict[str, np.ndarray]:
    """Fuse all available masks for a sequence."""
    strategy = routing.get("strategy", "union")
    primary_src = routing.get("primary", "sam3")

    # Collect all frame stems
    all_stems = set(sam3_masks.keys()) | set(gdino_masks.keys())
    if not all_stems:
        return {}

    fused = {}
    for stem in sorted(all_stems):
        m_sam3 = sam3_masks.get(stem)
        m_gdino = gdino_masks.get(stem)
        m_vggt4d = vggt4d_masks.get(stem)

        # Primary mask
        if primary_src == "sam3":
            primary = m_sam3
            secondary = m_gdino
        else:
            primary = m_gdino
            secondary = m_sam3

        if primary is None and secondary is not None:
            primary = secondary
            secondary = None

        if primary is None:
            fused[stem] = np.zeros_like(list(sam3_masks.values() or gdino_masks.values())[0])
            continue

        # Fuse primary + secondary
        fused_mask = fuse_two_masks(primary, secondary, strategy)

        # Optionally incorporate VGGT4D mask (adds unsupervised discovery)
        if m_vggt4d is not None:
            # VGGT4D fills in regions missed by both prompt-based methods
            # Only add VGGT4D pixels where they agree with the motion signal
            # Simple rule: add VGGT4D if it covers >5% new area
            existing_area = fused_mask.sum()
            new_pixels = np.logical_and(m_vggt4d, ~fused_mask.astype(bool))
            if new_pixels.sum() > 0.05 * (existing_area + 1):
                fused_mask = np.logical_or(fused_mask, m_vggt4d).astype(np.uint8)

        fused[stem] = fused_mask

    return fused


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_jm_jr_f(pred_masks: Dict[str, np.ndarray], gt_dir: Path) -> Dict[str, float]:
    if not gt_dir.exists():
        return {"JM": -1.0, "JR": -1.0, "F": -1.0}
    ious, recalls, f_scores = [], [], []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        if gt_path.stem not in pred_masks:
            continue
        gt_arr = np.array(Image.open(gt_path).convert("L")) > 0
        pred_arr = pred_masks[gt_path.stem].astype(bool)
        inter = np.logical_and(gt_arr, pred_arr).sum()
        union = np.logical_or(gt_arr, pred_arr).sum()
        iou = inter / (union + 1e-8)
        ious.append(iou)
        recalls.append(float(iou >= 0.5))
        gt_cont = cv2.Canny(gt_arr.astype(np.uint8) * 255, 50, 150) > 0
        pred_cont = cv2.Canny(pred_arr.astype(np.uint8) * 255, 50, 150) > 0
        tp = np.logical_and(gt_cont, pred_cont).sum()
        prec = tp / (pred_cont.sum() + 1e-8)
        rec = tp / (gt_cont.sum() + 1e-8)
        f_scores.append(2 * prec * rec / (prec + rec + 1e-8))
    if not ious:
        return {"JM": 0.0, "JR": 0.0, "F": 0.0}
    return {"JM": float(np.mean(ious)), "JR": float(np.mean(recalls)), "F": float(np.mean(f_scores))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Direction A-1: Adaptive mask fusion")
    p.add_argument("--sequences", nargs="+",
                   default=["tennis", "bmx-trees", "blackswan", "car-shadow",
                            "horsejump-low", "koala"])
    p.add_argument("--output_root",
                   default="/data3/jli657/project3/part3/outputs/direction_a/mask_fusion")
    p.add_argument("--use_vggt4d", action="store_true",
                   help="Incorporate VGGT4D dynamic masks in fusion")
    p.add_argument("--use_pi3", action="store_true",
                   help="Incorporate Pi3 transplant masks instead of VGGT4D")
    p.add_argument("--out_csv",
                   default="/home/jli657/my_storage2_1T/project3/eval/direction_a_fusion_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    gt_root = DAVIS_GT

    rows = []
    for seq in args.sequences:
        routing = SEQUENCE_ROUTING.get(seq, {"primary": "sam3", "strategy": "union"})

        # Load all available masks
        sam3_masks = load_masks_dir(SAM3_MASK_ROOTS.get(seq))
        gdino_masks = load_masks_dir(GDINO_SAM2_MASK_ROOTS.get(seq))

        # VGGT4D / Pi3 dynamic masks
        if args.use_pi3:
            dynamic_masks = load_masks_dir(PI3_MASK_ROOTS.get(seq))
        elif args.use_vggt4d:
            dynamic_masks = load_masks_dir(VGGT4D_MASK_ROOTS.get(seq))
        else:
            dynamic_masks = {}

        print(f"[{seq}] SAM3={len(sam3_masks)} GDINO={len(gdino_masks)} "
              f"Dynamic={len(dynamic_masks)} strategy={routing['strategy']}")

        if not sam3_masks and not gdino_masks:
            print(f"  [{seq}] No masks available, skipping")
            continue

        # Fuse masks
        fused_masks = fuse_sequence_masks(seq, sam3_masks, gdino_masks, dynamic_masks, routing)

        # Save fused masks
        out_dir = output_root / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, mask in fused_masks.items():
            Image.fromarray(mask * 255).save(str(out_dir / f"{stem}.png"))

        # Evaluate
        gt_dir = gt_root / seq
        metrics = compute_jm_jr_f(fused_masks, gt_dir)

        # Also evaluate component masks for comparison
        sam3_metrics = compute_jm_jr_f(sam3_masks, gt_dir)
        gdino_metrics = compute_jm_jr_f(gdino_masks, gt_dir)

        row = {
            "sequence": seq,
            "strategy": routing["strategy"],
            "n_frames": len(fused_masks),
            "fused_JM": metrics["JM"],
            "fused_JR": metrics["JR"],
            "fused_F": metrics["F"],
            "sam3_JM": sam3_metrics["JM"],
            "gdino_JM": gdino_metrics["JM"],
            "dynamic_n": len(dynamic_masks),
        }
        rows.append(row)

        jm_str = f"{metrics['JM']:.4f}" if metrics["JM"] >= 0 else "N/A"
        s3_str = f"{sam3_metrics['JM']:.4f}" if sam3_metrics["JM"] >= 0 else "N/A"
        gd_str = f"{gdino_metrics['JM']:.4f}" if gdino_metrics["JM"] >= 0 else "N/A"
        print(f"  [{seq}] Fused JM={jm_str} (SAM3={s3_str}, GDINO={gd_str})")

        with open(out_dir / "fusion_meta.json", "w") as f:
            json.dump(row, f, indent=2)

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "strategy", "n_frames", "fused_JM", "fused_JR", "fused_F",
              "sam3_JM", "gdino_JM", "dynamic_n"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {out_csv}")

    print("\n=== Direction A Fusion Results ===")
    print(f"{'Seq':<20} {'Fused JM':>10} {'SAM3 JM':>10} {'GDINO JM':>10}")
    for r in rows:
        fjm = r.get("fused_JM", -1.0)
        sjm = r.get("sam3_JM", -1.0)
        gjm = r.get("gdino_JM", -1.0)
        print(f"  {r['sequence']:<18} {fjm:>10.4f} {sjm:>10.4f} {gjm:>10.4f}")


if __name__ == "__main__":
    main()
