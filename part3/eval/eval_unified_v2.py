"""
eval_unified_v2.py — Unified Evaluation: Direction A vs B vs A+B Fusion

Computes PSNR_proxy + SSIM for the A+B best pipeline (results_v2),
then merges with existing evaluation_summary.csv (Part2, Dir-A) and
JM data from direction_b_comparison_v2.csv.

Output:
  /home/jli657/my_storage2_1T/project3/eval/unified_eval_v2.csv
  /home/jli657/my_storage2_1T/project3/eval/unified_eval_v2_print.txt

Usage:
  conda run -n controlnet_env python3 part3/eval_unified_v2.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_V2     = Path("/data3/jli657/project3/part3/results_v2")
DAVIS_FRAMES   = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
EVAL_DIR       = Path("/home/jli657/my_storage2_1T/project3/eval")
EXISTING_CSV   = EVAL_DIR / "evaluation_summary.csv" if False else Path(
    "/data3/jli657/project3/part3/results/evaluation_summary.csv"
)
JM_CSV         = EVAL_DIR / "direction_b_comparison_v2.csv"

SEQUENCES = ["tennis", "horsejump-low", "car-shadow", "blackswan", "bmx-trees", "koala"]
DAVIS5    = ["tennis", "horsejump-low", "car-shadow", "blackswan", "bmx-trees"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def evaluate_mp4(mp4: Path, seq: str) -> dict:
    """Compute PSNR_proxy, PSNR_synthetic, SSIM using evaluate_all.py approach."""
    if not mp4.exists():
        print(f"  [WARN] {mp4} not found")
        return {}
    orig_dir   = DAVIS_FRAMES   / seq
    gt_mask_dir = DAVIS_GT_MASKS / seq

    pred_frames  = load_video_frames(mp4)
    if not pred_frames:
        return {"error": "empty video"}
    orig_paths    = load_sorted_imgs(orig_dir)   if orig_dir.exists()    else []
    gt_mask_paths = sorted(gt_mask_dir.glob("*.png"), key=lambda p: p.stem) \
                    if gt_mask_dir.exists() else []

    has_gt = len(gt_mask_paths) > 0
    n = min(len(pred_frames), len(orig_paths),
            len(gt_mask_paths) if has_gt else len(pred_frames))

    proxy_list, synth_list, ssim_list = [], [], []
    for i in range(n):
        pred = pred_frames[i]
        orig = cv2.imread(str(orig_paths[i])) if i < len(orig_paths) else None
        if pred is None or orig is None:
            continue
        H, W = pred.shape[:2]
        if orig.shape[:2] != (H, W):
            orig = cv2.resize(orig, (W, H))

        if has_gt and i < len(gt_mask_paths):
            gt_m = cv2.imread(str(gt_mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            if gt_m is None:
                continue
            if gt_m.shape != (H, W):
                gt_m = cv2.resize(gt_m, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_bin    = (gt_m > 0).astype(np.uint8) * 255
            proxy_mask = 255 - gt_bin
            proxy_list.append(compute_psnr(pred, orig, roi_mask=proxy_mask))
            synth_list.append(compute_psnr(pred, orig, roi_mask=gt_bin))
        else:
            proxy_list.append(compute_psnr(pred, orig))
            synth_list.append(float("nan"))
        ssim_list.append(compute_ssim(pred, orig))

    def safe_mean(lst):
        valid = [v for v in lst if v == v and not np.isinf(v)]
        return round(float(np.mean(valid)), 4) if valid else float("nan")

    return {
        "psnr_proxy":  safe_mean(proxy_list),
        "psnr_synth":  safe_mean(synth_list),
        "ssim":        safe_mean(ssim_list),
        "n_frames":    n,
    }


# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------

def load_existing_csv():
    """Load evaluation_summary.csv -> {(seq, method): {PSNR_proxy, SSIM}}"""
    data = {}
    if not EXISTING_CSV.exists():
        print(f"[WARN] {EXISTING_CSV} not found")
        return data
    with open(EXISTING_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sequence"], row["method"])
            data[key] = {
                "psnr_proxy": float(row["PSNR_proxy"]) if row.get("PSNR_proxy") else float("nan"),
                "psnr_synth": float(row["PSNR_synthetic"]) if row.get("PSNR_synthetic") else float("nan"),
                "ssim":       float(row["SSIM"]) if row.get("SSIM") else float("nan"),
            }
    return data


def load_jm_csv():
    """Load direction_b_comparison_v2.csv -> {method: {seq: JM}}"""
    data = {}
    if not JM_CSV.exists():
        print(f"[WARN] {JM_CSV} not found")
        return data
    with open(JM_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            data[method] = {}
            for seq in SEQUENCES:
                val = row.get(seq, "N/A")
                try:
                    data[method][seq] = float(val)
                except (ValueError, TypeError):
                    data[method][seq] = float("nan")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Unified Evaluation v2 ===\n")

    existing = load_existing_csv()
    jm_data  = load_jm_csv()

    # --- Evaluate A+B best pipeline ---
    print("Evaluating A+B best pipeline (PSNR_proxy)...")
    ab_metrics = {}
    for seq in SEQUENCES:
        mp4 = RESULTS_V2 / seq / "a_plus_b_best" / "propainter" / seq / "inpaint_out.mp4"
        print(f"  [{seq}] {mp4}")
        m = evaluate_mp4(mp4, seq)
        ab_metrics[seq] = m
        if m:
            print(f"  [{seq}] PSNR_proxy={m.get('psnr_proxy'):.2f}  "
                  f"PSNR_synth={m.get('psnr_synth', float('nan')):.2f}  "
                  f"SSIM={m.get('ssim'):.4f}")

    # ---------------------------------------------------------------------------
    # Build unified table
    # ---------------------------------------------------------------------------

    # JM lookup shortcuts
    def jm(method_key, seq):
        return jm_data.get(method_key, {}).get(seq, float("nan"))

    # PSNR/SSIM from existing evaluation_summary
    def ep(seq, method):
        return existing.get((seq, method), {})

    # Build rows: one row per (seq, method)
    rows = []

    for seq in SEQUENCES:
        p2   = ep(seq, "part2_baseline")
        da   = ep(seq, "part3_dir_a")
        ab_m = ab_metrics.get(seq, {})

        # Part2 baseline
        rows.append({
            "sequence":    seq,
            "method":      "Part2 YOLO+SAM2+PP",
            "mask_JM":     jm("Part2 YOLO+SAM2", seq),
            "psnr_proxy":  p2.get("psnr_proxy", float("nan")),
            "psnr_synth":  p2.get("psnr_synth", float("nan")),
            "ssim":        p2.get("ssim", float("nan")),
        })

        # Direction A: SAM3+PP
        rows.append({
            "sequence":    seq,
            "method":      "Dir-A SAM3+PP",
            "mask_JM":     jm("A-1 SAM3 Fusion", seq),
            "psnr_proxy":  da.get("psnr_proxy", float("nan")),
            "psnr_synth":  da.get("psnr_synth", float("nan")),
            "ssim":        da.get("ssim", float("nan")),
        })

        # Direction B: VGGT4D only (no SAM3 refinement)
        rows.append({
            "sequence":    seq,
            "method":      "Dir-B VGGT4D (VGGT)",
            "mask_JM":     jm("B-1 VGGT4D (VGGT)", seq),
            "psnr_proxy":  float("nan"),   # no inpainting run for VGGT4D-only
            "psnr_synth":  float("nan"),
            "ssim":        float("nan"),
        })

        # Direction B: VGGT4D+SAM3 refined
        rows.append({
            "sequence":    seq,
            "method":      "Dir-B VGGT4D+SAM3",
            "mask_JM":     jm("B-5 VGGT4D+SAM3", seq),
            "psnr_proxy":  float("nan"),   # no standalone inpainting run; uses A+B fusion
            "psnr_synth":  float("nan"),
            "ssim":        float("nan"),
        })

        # A+B Best Fusion (best masks through ProPainter)
        rows.append({
            "sequence":    seq,
            "method":      "A+B Best Fusion+PP",
            "mask_JM":     jm("A+B Best Fusion", seq),
            "psnr_proxy":  ab_m.get("psnr_proxy", float("nan")),
            "psnr_synth":  ab_m.get("psnr_synth", float("nan")),
            "ssim":        ab_m.get("ssim", float("nan")),
        })

    # Save CSV
    out_csv = EVAL_DIR / "unified_eval_v2.csv"
    fields = ["sequence", "method", "mask_JM", "psnr_proxy", "psnr_synth", "ssim"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) and v == v else v)
                             for k, v in r.items()})
    print(f"\n[saved] {out_csv}")

    # ---------------------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------------------

    def fmt(v):
        if isinstance(v, float):
            return "  N/A  " if v != v else f"{v:7.4f}"
        return f"{str(v):>7}"

    def fmt2(v):
        if isinstance(v, float):
            return "   N/A  " if v != v else f"{v:8.2f}"
        return f"{str(v):>8}"

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("UNIFIED EVALUATION: Direction A vs B vs A+B Fusion")
    lines.append("=" * 90)
    header = f"{'Seq':<18} {'Method':<25} {'JM':>8} {'PSNR_prx':>10} {'PSNR_syn':>10} {'SSIM':>8}"
    lines.append(header)
    lines.append("-" * 90)

    prev_seq = None
    for r in rows:
        if r["sequence"] != prev_seq and prev_seq is not None:
            lines.append("")
        prev_seq = r["sequence"]
        jm_v    = r["mask_JM"]
        pp_v    = r["psnr_proxy"]
        ps_v    = r["psnr_synth"]
        ss_v    = r["ssim"]
        lines.append(
            f"{r['sequence']:<18} {r['method']:<25} "
            f"{fmt(jm_v)} {fmt2(pp_v)} {fmt2(ps_v)} {fmt(ss_v)}"
        )

    lines.append("\n" + "=" * 90)
    lines.append("DAVIS-5 MACRO AVERAGES (tennis/horsejump-low/car-shadow/blackswan/bmx-trees)")
    lines.append("=" * 90)
    lines.append(f"{'Method':<25} {'JM avg':>8} {'PSNR_prx avg':>14} {'SSIM avg':>10}")
    lines.append("-" * 60)

    methods_order = [
        "Part2 YOLO+SAM2+PP",
        "Dir-A SAM3+PP",
        "Dir-B VGGT4D (VGGT)",
        "Dir-B VGGT4D+SAM3",
        "A+B Best Fusion+PP",
    ]
    for mth in methods_order:
        jm_vals    = [r["mask_JM"]    for r in rows if r["method"] == mth and r["sequence"] in DAVIS5]
        psnr_vals  = [r["psnr_proxy"] for r in rows if r["method"] == mth and r["sequence"] in DAVIS5]
        ssim_vals  = [r["ssim"]       for r in rows if r["method"] == mth and r["sequence"] in DAVIS5]

        def avg(lst):
            valid = [v for v in lst if isinstance(v, float) and v == v]
            return float(np.mean(valid)) if valid else float("nan")

        lines.append(f"{mth:<25} {fmt(avg(jm_vals))} {fmt2(avg(psnr_vals))} {fmt(avg(ssim_vals))}")

    txt = "\n".join(lines)
    print(txt)

    out_txt = EVAL_DIR / "unified_eval_v2_print.txt"
    with open(out_txt, "w") as f:
        f.write(txt + "\n")
    print(f"\n[saved] {out_txt}")


if __name__ == "__main__":
    main()
