"""
run_direction_b_comparison.py
Direction B-6: Full comparison table
VGGT4D-VGGT vs VGGT4D-Pi3 vs SAM3-text vs VGGT4D+SAM3 fusion
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

DAVIS_GT = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

SEQUENCES = ["tennis", "blackswan", "horsejump-low", "koala", "bmx-trees", "car-shadow"]
DAVIS5 = ["tennis", "blackswan", "horsejump-low", "bmx-trees", "car-shadow"]

METHODS = {
    "B-1 VGGT4D (VGGT backbone)": {
        seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt/{seq}")
        for seq in SEQUENCES
    },
    "B-4 Pi3 Transplant": {
        seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/pi3_transplant_v3/{seq}")
        for seq in SEQUENCES
    },
    "B-5 VGGT4D+SAM3 Refine": {
        seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/{seq}")
        for seq in SEQUENCES
    },
    "A-1 SAM3 Text Only": {
        "tennis":       Path("/data3/jli657/project3/part3/outputs/direction_a/mask_fusion/tennis"),
        "blackswan":    Path("/data3/jli657/project3/part3/outputs/direction_a/mask_fusion/blackswan"),
        "horsejump-low":Path("/data3/jli657/project3/part3/outputs/direction_a/mask_fusion/horsejump-low"),
        "koala":        Path("/data3/jli657/project3/part3/outputs/direction_a/mask_fusion/koala"),
        "bmx-trees":    Path("/data3/jli657/project3/part3/outputs/direction_a/mask_fusion/bmx-trees"),
        "car-shadow":   Path("/data3/jli657/project3/part3/outputs/direction_a/mask_fusion/car-shadow"),
    },
}

# Pre-tabulated from CSV files (to avoid re-running)
PRECOMPUTED = {
    "A-1 SAM3 Text Only": {
        "tennis": 0.9468, "blackswan": 0.9549, "horsejump-low": 0.8574,
        "koala": 0.9482, "bmx-trees": 0.7455, "car-shadow": 0.9749,
    },
    "Part2 YOLO+SAM2 (baseline)": {
        "tennis": 0.932, "blackswan": 0.955, "horsejump-low": 0.723,
        "koala": None, "bmx-trees": 0.640, "car-shadow": 0.975,
    },
}


def compute_jm(pred_dir: Path, gt_dir: Path) -> float:
    if not pred_dir.exists() or not gt_dir.exists():
        return -1.0
    ious = []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt_arr = np.array(Image.open(gt_path).convert("L")) > 0
        pred_img = Image.open(pred_path).convert("L")
        if pred_img.size != (gt_arr.shape[1], gt_arr.shape[0]):
            pred_img = pred_img.resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
        pred_arr = np.array(pred_img) > 127
        inter = np.logical_and(gt_arr, pred_arr).sum()
        union = np.logical_or(gt_arr, pred_arr).sum()
        ious.append(inter / (union + 1e-8))
    return float(np.mean(ious)) if ious else -1.0


def main():
    print("\n" + "="*90)
    print("Direction B-6: Full Method Comparison Table")
    print("="*90)

    results = {}  # method -> seq -> JM

    # Compute from disk
    for method, dirs in METHODS.items():
        results[method] = {}
        for seq in SEQUENCES:
            gt_dir = DAVIS_GT / seq
            jm = compute_jm(dirs[seq], gt_dir)
            results[method][seq] = jm

    # Add pre-computed
    for method, vals in PRECOMPUTED.items():
        results[method] = vals

    # Build A+B best-per-seq
    results["A+B Best Fusion"] = {}
    for seq in SEQUENCES:
        a_jm = results["A-1 SAM3 Text Only"].get(seq, -1.0) or -1.0
        b_jm = results["B-5 VGGT4D+SAM3 Refine"].get(seq, -1.0)
        results["A+B Best Fusion"][seq] = max(a_jm, b_jm)

    # Print table
    col_w = 28
    print(f"\n{'Method':<{col_w}}", end="")
    for seq in SEQUENCES:
        print(f"{seq:>14}", end="")
    print(f"{'DAVIS5 Avg':>14}")
    print("-" * (col_w + 14 * len(SEQUENCES) + 14))

    method_order = [
        "Part2 YOLO+SAM2 (baseline)",
        "A-1 SAM3 Text Only",
        "B-1 VGGT4D (VGGT backbone)",
        "B-4 Pi3 Transplant",
        "B-5 VGGT4D+SAM3 Refine",
        "A+B Best Fusion",
    ]

    rows_for_csv = []
    for method in method_order:
        if method not in results:
            continue
        d = results[method]
        row = {"method": method}
        print(f"{method:<{col_w}}", end="")
        davis5_vals = []
        for seq in SEQUENCES:
            v = d.get(seq, None)
            if v is None or v < 0:
                print(f"{'N/A':>14}", end="")
                row[seq] = "N/A"
            else:
                print(f"{v:>14.4f}", end="")
                row[seq] = f"{v:.4f}"
                if seq in DAVIS5:
                    davis5_vals.append(v)
        macro = float(np.mean(davis5_vals)) if davis5_vals else -1.0
        print(f"{macro:>14.4f}")
        row["DAVIS5_macro"] = f"{macro:.4f}"
        rows_for_csv.append(row)

    print()

    # Save CSV
    out_csv = Path("/home/jli657/my_storage2_1T/project3/eval/direction_b_comparison_v2.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method"] + SEQUENCES + ["DAVIS5_macro"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"[saved] {out_csv}")

    # Per-sequence best analysis
    print("\n=== Per-Sequence Winner Analysis ===")
    for seq in SEQUENCES:
        best_method = None
        best_jm = -1.0
        for method in method_order:
            jm = results.get(method, {}).get(seq, None)
            if jm is not None and jm > 0 and jm > best_jm:
                best_jm = jm
                best_method = method
        print(f"  {seq:<20}: best={best_method} (JM={best_jm:.4f})")


if __name__ == "__main__":
    main()
