"""
diagnose_weak_sequences.py — Task 2: Direction A 弱序列系统诊断

对 bmx-trees 和 car-shadow 的 SAM3 预测 mask 做逐帧 IoU 分析，
标记崩溃帧（IoU < 0.5），生成可视化对比图和诊断 JSON。

失败类型分类:
  (a) 目标整体漏检 - pred 区域 < GT 的 10%
  (b) 部分覆盖不足 - IoU < 0.5 但 pred 非空
  (c) 边界粗糙     - IoU 在 [0.5, 0.75) 区间（中等质量）
  (d) 精确追踪     - IoU >= 0.75
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SEQUENCES = {
    "bmx-trees": {
        "pred_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/bmx-trees",
        "gt_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/bmx-trees",
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
        "part2_mask_dir": "/home/jli657/my_storage2_1T/project3/part2/masks_cache/bmx-trees_dilated",
    },
    "car-shadow": {
        "pred_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/car-shadow",
        "gt_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/car-shadow",
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/car-shadow",
        "part2_mask_dir": "/home/jli657/my_storage2_1T/project3/part2/masks_cache/car-shadow",
    },
}

OUT_ROOT = Path("/home/jli657/my_storage2_1T/project3/report_assets/diagnosis")
CRASH_THRESHOLD = 0.5


def classify_failure(iou: float, pred_area: float, gt_area: float) -> str:
    """Classify frame quality."""
    if pred_area < gt_area * 0.10:
        return "leakage_miss"       # (a) 目标整体漏检
    elif iou < 0.5:
        return "partial_coverage"   # (b) 部分覆盖不足
    elif iou < 0.75:
        return "rough_boundary"     # (c) 边界粗糙
    else:
        return "good"               # (d) 精确追踪


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    """Compute IoU, pred area fraction, gt area fraction."""
    total = pred.shape[0] * pred.shape[1]
    pred_bin = pred > 127
    gt_bin = gt > 0  # DAVIS GT: instance ID > 0 = foreground
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    iou = inter / union if union > 0 else 0.0
    return iou, pred_bin.sum() / total, gt_bin.sum() / total


def load_mask(path: Path, target_shape=None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(target_shape or (480, 854), dtype=np.uint8)
    if target_shape and (m.shape[0] != target_shape[0] or m.shape[1] != target_shape[1]):
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def make_overlay(orig_bgr: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Create visualization: orig + green (TP pred) + red (FN gt only) + blue (FP pred only)."""
    h, w = orig_bgr.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bin = pred_mask > 127
    gt_bin = gt_mask > 0

    overlay = orig_bgr.copy()
    tp = pred_bin & gt_bin   # green: correct
    fp = pred_bin & ~gt_bin  # blue: over-segmented
    fn = ~pred_bin & gt_bin  # red: missed

    overlay[tp] = (overlay[tp] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
    overlay[fp] = (overlay[fp] * 0.5 + np.array([200, 0, 0]) * 0.5).astype(np.uint8)
    overlay[fn] = (overlay[fn] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)
    return overlay


def diagnose_sequence(seq_name: str, cfg: dict) -> dict:
    pred_dir = Path(cfg["pred_dir"])
    gt_dir = Path(cfg["gt_dir"])
    orig_dir = Path(cfg["orig_dir"])
    part2_dir = Path(cfg["part2_mask_dir"])

    out_dir = OUT_ROOT / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*.png"), key=lambda p: p.stem)
    per_frame = []
    crash_frames = []

    for gt_path in gt_files:
        fname = gt_path.stem + ".png"
        pred_path = pred_dir / fname
        orig_path = list(orig_dir.glob(f"{gt_path.stem}.*"))
        orig_path = orig_path[0] if orig_path else None

        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue

        target_shape = gt_img.shape

        pred_img = load_mask(pred_path, target_shape) if pred_path.exists() else np.zeros(target_shape, dtype=np.uint8)
        iou, pred_area, gt_area = compute_iou(pred_img, gt_img)
        failure_type = classify_failure(iou, pred_area, gt_area)

        record = {
            "frame": gt_path.stem,
            "iou": round(iou, 4),
            "pred_area_frac": round(pred_area, 4),
            "gt_area_frac": round(gt_area, 4),
            "failure_type": failure_type,
        }
        per_frame.append(record)

        if iou < CRASH_THRESHOLD and orig_path is not None:
            crash_frames.append(record)
            # Create visualization
            orig_bgr = cv2.imread(str(orig_path))
            if orig_bgr is not None:
                vis = make_overlay(orig_bgr, pred_img, gt_img)
                # Also show Part2 mask for comparison
                part2_path = part2_dir / fname
                if part2_path.exists():
                    part2_mask = load_mask(part2_path, target_shape)
                    vis_p2 = make_overlay(orig_bgr, part2_mask, gt_img)
                    vis = np.hstack([vis, vis_p2])
                    cv2.putText(vis, f"Part3 IoU={iou:.3f} ({failure_type})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(vis, "Part2 (left=Part3, right=Part2)",
                                (orig_bgr.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(vis, f"Part3 IoU={iou:.3f} ({failure_type})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                vis_path = out_dir / f"crash_{gt_path.stem}.png"
                cv2.imwrite(str(vis_path), vis)

    ious = [r["iou"] for r in per_frame]
    failure_counts = {}
    for r in per_frame:
        ft = r["failure_type"]
        failure_counts[ft] = failure_counts.get(ft, 0) + 1

    result = {
        "sequence": seq_name,
        "num_frames": len(per_frame),
        "jm_mean": round(float(np.mean(ious)), 4) if ious else 0.0,
        "jr_at_0.5": round(sum(1 for iou in ious if iou >= 0.5) / len(ious), 4) if ious else 0.0,
        "crash_frames_count": len(crash_frames),
        "crash_frame_indices": [r["frame"] for r in crash_frames[:10]],
        "failure_type_distribution": failure_counts,
        "per_frame": per_frame,
    }

    # Save JSON
    out_json = out_dir / f"diagnosis_{seq_name}.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\n[{seq_name}] Diagnosis:")
    print(f"  JM mean: {result['jm_mean']:.4f}")
    print(f"  JR@0.5: {result['jr_at_0.5']:.4f}")
    print(f"  Crash frames (IoU<0.5): {result['crash_frames_count']}/{result['num_frames']}")
    print(f"  Failure distribution: {failure_counts}")
    print(f"  Saved to: {out_dir}")

    return result


# Need typing import
from typing import Tuple


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for seq_name, cfg in SEQUENCES.items():
        print(f"\n{'='*60}\nDiagnosing: {seq_name}\n{'='*60}")

        pred_dir = Path(cfg["pred_dir"])
        if not pred_dir.exists():
            print(f"  WARN: pred_dir not found: {pred_dir}")
            # Try to find in official_sam3_best or prompt_search
            alt = Path("/data3/jli657/project3/part3/outputs/official_sam3_best") / seq_name
            if alt.exists():
                print(f"  Using alt: {alt}")
                cfg = dict(cfg)
                cfg["pred_dir"] = str(alt)
            else:
                print(f"  SKIP: no pred_dir found")
                continue

        all_results[seq_name] = diagnose_sequence(seq_name, cfg)

    # Summary table
    print("\n\n=== DIAGNOSIS SUMMARY ===")
    print(f"{'Sequence':<15} {'JM':<8} {'JR@0.5':<8} {'Crashes':<10} {'Top Failure Type'}")
    for seq, res in all_results.items():
        top_failure = max(res["failure_type_distribution"].items(), key=lambda x: x[1])[0] if res["failure_type_distribution"] else "N/A"
        print(f"{seq:<15} {res['jm_mean']:<8.4f} {res['jr_at_0.5']:<8.4f} {res['crash_frames_count']:<10} {top_failure}")

    # Save combined JSON
    out_combined = OUT_ROOT / "diagnosis_combined.json"
    with open(out_combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] {out_combined}")


if __name__ == "__main__":
    main()
