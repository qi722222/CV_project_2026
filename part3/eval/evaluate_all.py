"""
evaluate_all.py — Phase 4: 统一量化评估

对 results/ 下每个 <seq>/<method>/inpaint_out.mp4 计算:
  - PSNR_proxy: 非 mask 区域 (保留区域) PSNR
  - PSNR_synthetic: mask 区域 PSNR（对比 GT clean background；若有 GT mask 则计算）
  - SSIM: 全帧 SSIM

输出: results/evaluation_summary.csv

用法:
  conda run -n controlnet_env python3 part3/evaluate_all.py
  conda run -n controlnet_env python3 part3/evaluate_all.py --seqs koala tennis
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

RESULTS           = Path("/data3/jli657/project3/part3/results")
DAVIS_FRAMES      = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS    = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
WILD_FRAMES       = Path("/data3/jli657/project3/wild_frames")

# Override frame dirs for non-DAVIS sequences
CUSTOM_FRAME_DIRS = {
    "wild_video-1person": WILD_FRAMES / "wild_video-1person",
}

# Parts2 baselines exist at:
PART2_RESULT_DIRS = {
    "tennis":            "/data3/jli657/project3/part2/outputs/tennis_v3/tennis",
    "koala":             None,  # Part2 has no koala
    "wild_video-1person": "/data3/jli657/project3/part2/outputs/wild_video-1person/wild_video-1person",
    "bmx-trees":         "/data3/jli657/project3/part2/outputs/bmx-trees_v2/bmx-trees",
    "blackswan":         None,
    "horsejump-low":     None,
    "car-shadow":        None,
}

ALL_SEQUENCES = [
    "tennis", "koala", "wild_video-1person",
    "bmx-trees", "blackswan", "horsejump-low", "car-shadow"
]

# Wild video: use shadow-enhanced masks
WILD_SEQ_MAP = {"wild_video-1person": "wild_video-1person"}


def load_video_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def load_sorted_imgs(d: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def compute_psnr(img1: np.ndarray, img2: np.ndarray,
                 roi_mask: Optional[np.ndarray] = None) -> float:
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    if roi_mask is not None:
        roi = roi_mask > 127
        if roi.sum() == 0:
            return float("nan")
        diff = diff[roi]
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(
            img1, img2, multichannel=True, data_range=255, channel_axis=2
        ))
    except Exception:
        return float("nan")


def evaluate_method(pred_mp4: Path, seq: str, gt_seq_name: Optional[str] = None) -> dict:
    if not pred_mp4.exists():
        return {}

    orig_seq_name = gt_seq_name or seq
    # Use custom frame dir if available (e.g. wild_video-1person)
    if seq in CUSTOM_FRAME_DIRS:
        orig_dir = CUSTOM_FRAME_DIRS[seq]
    else:
        orig_dir = DAVIS_FRAMES / orig_seq_name
    gt_mask_dir = DAVIS_GT_MASKS / orig_seq_name

    pred_frames = load_video_frames(pred_mp4)
    if not pred_frames:
        return {"error": "empty video"}

    orig_paths = load_sorted_imgs(orig_dir) if orig_dir.exists() else []
    gt_mask_paths = sorted(gt_mask_dir.glob("*.png"), key=lambda p: p.stem) if gt_mask_dir.exists() else []

    has_gt = len(gt_mask_paths) > 0
    n = min(len(pred_frames), len(orig_paths), len(gt_mask_paths) if has_gt else len(pred_frames))

    proxy_list, synth_list, ssim_list = [], [], []

    for i in range(n):
        pred = pred_frames[i]
        orig = cv2.imread(str(orig_paths[i]))
        if pred is None or orig is None:
            continue

        H, W = pred.shape[:2]
        if orig.shape[:2] != (H, W):
            orig = cv2.resize(orig, (W, H))

        if has_gt:
            gt_m = cv2.imread(str(gt_mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            if gt_m is None:
                continue
            if gt_m.shape != (H, W):
                gt_m = cv2.resize(gt_m, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt_m > 0).astype(np.uint8) * 255
            proxy_mask = 255 - gt_bin  # non-masked region

            psnr_p = compute_psnr(pred, orig, roi_mask=proxy_mask)
            psnr_s = compute_psnr(pred, orig, roi_mask=gt_bin)
        else:
            psnr_p = compute_psnr(pred, orig)
            psnr_s = float("nan")

        proxy_list.append(psnr_p)
        synth_list.append(psnr_s)
        ssim_list.append(compute_ssim(pred, orig))

    def safe_mean(lst):
        valid = [v for v in lst if v == v and not np.isinf(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "PSNR_proxy":     safe_mean(proxy_list),
        "PSNR_synthetic": safe_mean(synth_list),
        "SSIM":           safe_mean(ssim_list),
        "n_frames":       n,
    }


def get_methods_for_seq(seq_dir: Path) -> List[tuple]:
    """Returns (method_name, mp4_path, display_name) tuples for all methods.

    GT-mask variants (davis_gt protocol) are named with suffix _gtmask for clarity.
    Legacy SAM3-mask variants are still collected for backward traceability.
    """
    methods = []

    # Part2 baseline
    p2_mp4 = seq_dir / "part2_baseline" / "inpaint_out.mp4"
    if p2_mp4.exists() or p2_mp4.is_symlink():
        methods.append(("part2_baseline", p2_mp4, "Part2 Baseline (YOLO+SAM2+PP)"))

    # Direction A
    da_mp4 = seq_dir / "direction_a" / "sam3_propainter" / "inpaint_out.mp4"
    if da_mp4.exists() or da_mp4.is_symlink():
        methods.append(("part3_dir_a", da_mp4, "Part3 Dir-A (SAM3+PP)"))

    # Direction C: pure ProPainter — GT mask (fair inpaint-only comparison)
    dc_pp_gt = seq_dir / "direction_c" / "pure_propainter_gtmask" / "inpaint_out.mp4"
    if dc_pp_gt.exists() or dc_pp_gt.is_symlink():
        methods.append(("part3_pure_pp_gtmask", dc_pp_gt,
                        "Part3 Dir-C Pure PP [GT mask]"))

    # Direction C: pure ProPainter — legacy SAM3 mask
    dc_pp = seq_dir / "direction_c" / "pure_propainter" / "inpaint_out.mp4"
    if dc_pp.exists() or dc_pp.is_symlink():
        methods.append(("part3_pure_pp", dc_pp, "Part3 Dir-C Pure PP [SAM3 mask]"))

    # Direction C: SDXL kf5 + ProPainter — GT mask
    dc_sdxl_gt = seq_dir / "direction_c" / "sdxl_kf5_gtmask_propainter" / "inpaint_out.mp4"
    if dc_sdxl_gt.exists() or dc_sdxl_gt.is_symlink():
        methods.append(("part3_sdxl_kf5_gtmask", dc_sdxl_gt,
                        "Part3 Dir-C SDXL kf5+PP [GT mask]"))

    # Direction C: SDXL kf5 + ProPainter — legacy SAM3 mask
    dc_sdxl = seq_dir / "direction_c" / "sdxl_kf5_propainter" / "inpaint_out.mp4"
    if dc_sdxl.exists() or dc_sdxl.is_symlink():
        methods.append(("part3_sdxl_kf5", dc_sdxl, "Part3 Dir-C SDXL kf5+PP [SAM3 mask]"))

    # Direction C: LaMa + ProPainter — GT mask
    dc_lama_gt = seq_dir / "direction_c" / "lama_gtmask_propainter" / "inpaint_out.mp4"
    if dc_lama_gt.exists() or dc_lama_gt.is_symlink():
        methods.append(("part3_lama_gtmask", dc_lama_gt,
                        "Part3 Dir-C LaMa kf5+PP [GT mask]"))

    # Direction C: LaMa + ProPainter — legacy SAM3 mask
    dc_lama = seq_dir / "direction_c" / "lama_propainter" / "inpaint_out.mp4"
    if dc_lama.exists() or dc_lama.is_symlink():
        methods.append(("part3_lama", dc_lama, "Part3 Dir-C LaMa kf5+PP [SAM3 mask]"))

    # Direction C: DiffuEraser — GT mask (versioned, pattern: diffueraser_gtmask_v*)
    dc_dir = seq_dir / "direction_c"
    if dc_dir.exists():
        for de_dir in sorted(dc_dir.iterdir()):
            if de_dir.name.startswith("diffueraser_") and de_dir.is_dir():
                de_mp4 = de_dir / "inpaint_out.mp4"
                if de_mp4.exists() or de_mp4.is_symlink():
                    ver_tag = de_dir.name  # e.g. diffueraser_gtmask_v1
                    methods.append((f"part3_{ver_tag}", de_mp4,
                                    f"Part3 Dir-C DiffuEraser [GT mask / {ver_tag}]"))

    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", default=ALL_SEQUENCES)
    parser.add_argument("--output_csv", default=str(RESULTS / "evaluation_summary.csv"))
    args = parser.parse_args()

    rows = []
    per_seq_results = {}

    for seq in args.seqs:
        seq_dir = RESULTS / seq
        if not seq_dir.exists():
            print(f"[SKIP] {seq}: results dir missing")
            continue

        # Determine the real DAVIS sequence name for frame loading
        # wild_video-1person uses same name for DAVIS
        orig_seq_name = seq

        print(f"\n=== {seq} ===")
        methods = get_methods_for_seq(seq_dir)
        seq_results = {}

        for method_key, mp4_path, display in methods:
            print(f"  {display}...")
            metrics = evaluate_method(mp4_path, seq, orig_seq_name)
            if not metrics or "error" in metrics:
                print(f"    [ERROR] {metrics}")
                continue

            row = {
                "sequence":       seq,
                "method":         method_key,
                "display_name":   display,
                "PSNR_proxy":     f"{metrics.get('PSNR_proxy', float('nan')):.4f}",
                "PSNR_synthetic": f"{metrics.get('PSNR_synthetic', float('nan')):.4f}",
                "SSIM":           f"{metrics.get('SSIM', float('nan')):.4f}",
                "n_frames":       metrics.get("n_frames", 0),
            }
            rows.append(row)
            seq_results[method_key] = metrics

            print(f"    PSNR_proxy={metrics.get('PSNR_proxy', float('nan')):.3f}  "
                  f"PSNR_synth={metrics.get('PSNR_synthetic', float('nan')):.3f}  "
                  f"SSIM={metrics.get('SSIM', float('nan')):.4f}")

        per_seq_results[seq] = seq_results

        # Compute deltas vs Part2 baseline
        if "part2_baseline" in seq_results:
            p2 = seq_results["part2_baseline"]
            for mk, mr in seq_results.items():
                if mk == "part2_baseline":
                    continue
                delta_p = mr.get("PSNR_proxy", float("nan")) - p2.get("PSNR_proxy", float("nan"))
                delta_s = mr.get("PSNR_synthetic", float("nan")) - p2.get("PSNR_synthetic", float("nan"))
                print(f"    Δ({mk} vs Part2): proxy={delta_p:+.3f}  synth={delta_s:+.3f}")

    # Write CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[CSV saved] {output_csv} ({len(rows)} rows)")

    # Write per-seq JSON
    json_path = RESULTS / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(per_seq_results, f, indent=2, default=str)
    print(f"[JSON saved] {json_path}")

    # Print summary table
    print("\n" + "="*90)
    print(f"{'Sequence':<22} {'Method':<22} {'PSNR_proxy':>12} {'PSNR_synth':>12} {'SSIM':>8}")
    print("="*90)
    for row in rows:
        print(f"{row['sequence']:<22} {row['method']:<22} "
              f"{row['PSNR_proxy']:>12} {row['PSNR_synthetic']:>12} {row['SSIM']:>8}")


if __name__ == "__main__":
    main()
