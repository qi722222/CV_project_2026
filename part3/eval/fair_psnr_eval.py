"""
fair_psnr_eval.py — Task 1: 公平 PSNR/SSIM 评估

使用统一的 DAVIS GT mask（而非各自的 prediction mask）对 Part2/Part3 inpaint 输出
计算 PSNR_proxy 和 PSNR_synthetic，消除 mask ROI 差异导致的虚假增益。

评估分辨率: 统一使用 EVAL_H x EVAL_W (864x480) —— Part3 的原生分辨率，
Part2 视频会被双线性插值放大到该分辨率（体现 Part2 低分辨率的实际局限）。

PSNR_proxy: 非 mask 区域（frame 内 mask=0 的像素）原始帧 vs inpaint 帧之间的 PSNR
PSNR_synthetic: 用 cv2.inpaint 在 GT mask 区域生成"合成 GT"，与 inpaint 帧全局比 PSNR
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

EVAL_H, EVAL_W = 480, 864   # 统一评估分辨率

SEQUENCES = ["tennis", "bmx-trees"]

PART2_VIDEOS = {
    "tennis": "/data3/jli657/project3/part2/outputs/tennis_v3/tennis/inpaint_out.mp4",
    "bmx-trees": "/data3/jli657/project3/part2/outputs/bmx-trees_v2/bmx-trees/inpaint_out.mp4",
}

PART3_VIDEOS = {
    "tennis": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/tennis/propainter_pure/tennis/inpaint_out.mp4",
    "bmx-trees": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/bmx-trees/propainter_pure/bmx-trees/inpaint_out.mp4",
}

DAVIS_FRAMES = {
    "tennis": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
    "bmx-trees": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
}

DAVIS_GT_MASKS = {
    "tennis": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/tennis",
    "bmx-trees": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/bmx-trees",
}


def load_video_frames(video_path: str, target_h: int, target_w: int) -> List[np.ndarray]:
    """Load all frames from video, resize to target size."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames


def load_dir_frames(dir_path: str, target_h: int, target_w: int,
                    exts=(".jpg", ".jpeg", ".png")) -> List[np.ndarray]:
    """Load sorted frames from directory, resize to target size."""
    p = Path(dir_path)
    paths = sorted([f for f in p.iterdir() if f.suffix.lower() in exts], key=lambda x: x.stem)
    frames = []
    for fp in paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
    return frames


def load_gt_masks(mask_dir: str, target_h: int, target_w: int) -> List[np.ndarray]:
    """Load DAVIS GT annotation masks. DAVIS uses instance ID > 0 as foreground."""
    p = Path(mask_dir)
    paths = sorted([f for f in p.iterdir() if f.suffix.lower() == ".png"], key=lambda x: x.stem)
    masks = []
    for fp in paths:
        m = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        binary = (m > 0).astype(np.uint8) * 255
        if binary.shape[0] != target_h or binary.shape[1] != target_w:
            binary = cv2.resize(binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            binary = (binary > 127).astype(np.uint8) * 255
        masks.append(binary)
    return masks


def compute_psnr(img1: np.ndarray, img2: np.ndarray,
                 roi_mask: Optional[np.ndarray] = None) -> float:
    """Compute PSNR between img1 and img2 within roi_mask region (or full frame if None)."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    if roi_mask is not None:
        roi = roi_mask > 127
        if roi.sum() == 0:
            return float("nan")
        diff = diff[roi]
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute mean SSIM over all channels."""
    from skimage.metrics import structural_similarity
    if img1.ndim == 3:
        return float(structural_similarity(img1, img2, multichannel=True, data_range=255,
                                           channel_axis=2))
    return float(structural_similarity(img1, img2, data_range=255))


def evaluate_sequence(seq: str, inpaint_frames: List[np.ndarray],
                      orig_frames: List[np.ndarray],
                      gt_masks: List[np.ndarray]) -> dict:
    """Compute per-frame and aggregate PSNR/SSIM for one method on one sequence."""
    n = min(len(inpaint_frames), len(orig_frames), len(gt_masks))
    psnr_proxy_list, psnr_synth_list, ssim_proxy_list, ssim_synth_list = [], [], [], []

    for i in range(n):
        pred = inpaint_frames[i]
        orig = orig_frames[i]
        mask = gt_masks[i]  # 255=foreground (inpaint region), 0=background

        # proxy mask: non-inpaint region
        proxy_mask = (255 - mask)  # 255 where mask=0

        # PSNR_proxy: measure on background pixels (where mask=0)
        psnr_p = compute_psnr(pred, orig, roi_mask=proxy_mask)
        if not np.isnan(psnr_p):
            psnr_proxy_list.append(psnr_p)

        # Synthetic GT: cv2.inpaint the original frame in the GT mask region
        synth_gt = cv2.inpaint(orig, (mask > 127).astype(np.uint8), inpaintRadius=3,
                               flags=cv2.INPAINT_TELEA)
        psnr_s = compute_psnr(pred, synth_gt)
        if not np.isnan(psnr_s):
            psnr_synth_list.append(psnr_s)

        # SSIM_proxy (compute on whole frame but masked - simpler: compute full SSIM)
        # For proxy: apply mask to zero out inpaint region before SSIM
        pred_proxy = pred.copy()
        orig_proxy = orig.copy()
        pred_proxy[mask > 127] = 0
        orig_proxy[mask > 127] = 0
        ssim_p = compute_ssim(pred_proxy, orig_proxy)
        ssim_proxy_list.append(ssim_p)

        ssim_s = compute_ssim(pred, synth_gt)
        ssim_synth_list.append(ssim_s)

    return {
        "num_frames": n,
        "PSNR_proxy": float(np.mean(psnr_proxy_list)) if psnr_proxy_list else float("nan"),
        "SSIM_proxy": float(np.mean(ssim_proxy_list)) if ssim_proxy_list else float("nan"),
        "PSNR_synthetic": float(np.mean(psnr_synth_list)) if psnr_synth_list else float("nan"),
        "SSIM_synthetic": float(np.mean(ssim_synth_list)) if ssim_synth_list else float("nan"),
    }


def main():
    out_csv = "/home/jli657/my_storage2_1T/project3/report_assets/final_delivery/table2_video_quality_fair.csv"
    out_json = "/home/jli657/my_storage2_1T/project3/report_assets/final_delivery/fair_psnr_eval.json"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_results = {}

    for seq in SEQUENCES:
        print(f"\n{'='*60}\n[{seq}] Evaluating fair PSNR\n{'='*60}")

        # Load original DAVIS frames
        print(f"  Loading DAVIS orig frames from {DAVIS_FRAMES[seq]}")
        orig_frames = load_dir_frames(DAVIS_FRAMES[seq], EVAL_H, EVAL_W)
        print(f"  Loaded {len(orig_frames)} orig frames")

        # Load GT masks
        print(f"  Loading GT masks from {DAVIS_GT_MASKS[seq]}")
        gt_masks = load_gt_masks(DAVIS_GT_MASKS[seq], EVAL_H, EVAL_W)
        print(f"  Loaded {len(gt_masks)} GT masks, coverage: {np.mean([np.sum(m>127)/(EVAL_H*EVAL_W) for m in gt_masks]):.3f}")

        all_results[seq] = {}

        for method_tag, video_path in [("part2", PART2_VIDEOS[seq]), ("part3_sam3", PART3_VIDEOS[seq])]:
            print(f"\n  [{method_tag}] Loading {video_path}")
            if not Path(video_path).exists():
                print(f"  SKIP: {video_path} not found")
                continue
            inpaint_frames = load_video_frames(video_path, EVAL_H, EVAL_W)
            print(f"  Loaded {len(inpaint_frames)} inpaint frames")

            metrics = evaluate_sequence(seq, inpaint_frames, orig_frames, gt_masks)
            all_results[seq][method_tag] = metrics
            print(f"  {method_tag}: PSNR_proxy={metrics['PSNR_proxy']:.3f} "
                  f"SSIM_proxy={metrics['SSIM_proxy']:.4f} "
                  f"PSNR_synth={metrics['PSNR_synthetic']:.3f} "
                  f"SSIM_synth={metrics['SSIM_synthetic']:.4f}")

            rows.append({
                "sequence": seq,
                "method": method_tag,
                "eval_resolution": f"{EVAL_W}x{EVAL_H}",
                "mask_source": "DAVIS_GT",
                "num_frames": metrics["num_frames"],
                "PSNR_proxy": f"{metrics['PSNR_proxy']:.4f}",
                "SSIM_proxy": f"{metrics['SSIM_proxy']:.6f}",
                "PSNR_synthetic": f"{metrics['PSNR_synthetic']:.4f}",
                "SSIM_synthetic": f"{metrics['SSIM_synthetic']:.6f}",
            })

    # Write CSV
    fieldnames = ["sequence", "method", "eval_resolution", "mask_source",
                  "num_frames", "PSNR_proxy", "SSIM_proxy", "PSNR_synthetic", "SSIM_synthetic"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[save] {out_csv}")

    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[save] {out_json}")


if __name__ == "__main__":
    main()
