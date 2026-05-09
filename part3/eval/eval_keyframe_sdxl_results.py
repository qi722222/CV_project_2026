"""
eval_keyframe_sdxl_results.py — 评估已有的 keyframe SDXL + ProPainter 输出

专为 Task 8 的评估步骤设计，对已存在的输出目录计算 PSNR/SSIM。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

SEQUENCES = {
    "tennis": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/tennis",
        "keyframe_sdxl_output": "/data3/jli657/project3/part3/outputs/keyframe_sdxl/tennis/interval10/propainter_output/frames/inpaint_out.mp4",
        "pure_propainter_video": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/tennis/propainter_pure/tennis/inpaint_out.mp4",
    },
    "bmx-trees": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/bmx-trees",
        "keyframe_sdxl_output": "/data3/jli657/project3/part3/outputs/keyframe_sdxl/bmx-trees/interval10/propainter_output/frames/inpaint_out.mp4",
        "pure_propainter_video": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/bmx-trees/propainter_pure/bmx-trees/inpaint_out.mp4",
    },
}

EVAL_H, EVAL_W = 480, 864
OUT_CSV = "/home/jli657/my_storage2_1T/project3/report_assets/final_delivery/direction_c_keyframe_metrics.csv"


def load_video(video_path: str, h: int, w: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return frames


def load_dir(dir_path: str, h: int, w: int) -> List[np.ndarray]:
    p = Path(dir_path)
    paths = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
                   key=lambda x: x.stem)
    frames = []
    for fp in paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
    return frames


def load_gt_masks(mask_dir: str, h: int, w: int) -> List[np.ndarray]:
    p = Path(mask_dir)
    paths = sorted([f for f in p.iterdir() if f.suffix.lower() == ".png"], key=lambda x: x.stem)
    masks = []
    for fp in paths:
        m = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        binary = (m > 0).astype(np.uint8) * 255
        if binary.shape[0] != h or binary.shape[1] != w:
            binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = (binary > 127).astype(np.uint8) * 255
        masks.append(binary)
    return masks


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
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(img1, img2, multichannel=True,
                                       data_range=255, channel_axis=2))


def evaluate(pred_frames: List[np.ndarray], orig_frames: List[np.ndarray],
             gt_masks: List[np.ndarray]) -> dict:
    n = min(len(pred_frames), len(orig_frames), len(gt_masks))
    psnr_proxy_list, psnr_synth_list, ssim_synth_list = [], [], []

    for i in range(n):
        pred = pred_frames[i]
        orig = orig_frames[i]
        mask = gt_masks[i]

        proxy_mask = (255 - mask)
        psnr_p = compute_psnr(pred, orig, roi_mask=proxy_mask)
        if not np.isnan(psnr_p):
            psnr_proxy_list.append(psnr_p)

        synth = cv2.inpaint(orig, (mask > 127).astype(np.uint8), inpaintRadius=3,
                            flags=cv2.INPAINT_TELEA)
        psnr_s = compute_psnr(pred, synth)
        if not np.isnan(psnr_s):
            psnr_synth_list.append(psnr_s)

        try:
            ssim_s = compute_ssim(pred, synth)
            ssim_synth_list.append(ssim_s)
        except Exception:
            pass

    return {
        "num_frames": n,
        "PSNR_proxy": float(np.mean(psnr_proxy_list)) if psnr_proxy_list else float("nan"),
        "PSNR_synthetic": float(np.mean(psnr_synth_list)) if psnr_synth_list else float("nan"),
        "SSIM_synthetic": float(np.mean(ssim_synth_list)) if ssim_synth_list else float("nan"),
    }


def main():
    rows = []
    all_results = {}

    for seq_name, cfg in SEQUENCES.items():
        print(f"\n{'='*60}\n[{seq_name}]\n{'='*60}")

        orig_frames = load_dir(cfg["orig_dir"], EVAL_H, EVAL_W)
        gt_masks = load_gt_masks(cfg["gt_mask_dir"], EVAL_H, EVAL_W)
        print(f"  orig: {len(orig_frames)} frames, gt_masks: {len(gt_masks)}")

        # Load keyframe_sdxl + ProPainter output
        kf_path = cfg["keyframe_sdxl_output"]
        if not Path(kf_path).exists():
            print(f"  SKIP keyframe_sdxl: {kf_path} not found")
            kf_metrics = None
        else:
            kf_frames = load_video(kf_path, EVAL_H, EVAL_W)
            print(f"  keyframe_sdxl: {len(kf_frames)} frames")
            kf_metrics = evaluate(kf_frames, orig_frames, gt_masks)
            print(f"  keyframe_sdxl: PSNR_proxy={kf_metrics['PSNR_proxy']:.3f} "
                  f"PSNR_synth={kf_metrics['PSNR_synthetic']:.3f} "
                  f"SSIM_synth={kf_metrics['SSIM_synthetic']:.4f}")

        # Load pure_propainter reference
        pp_path = cfg["pure_propainter_video"]
        if not Path(pp_path).exists():
            print(f"  SKIP pure_propainter: {pp_path} not found")
            pp_metrics = None
        else:
            pp_frames = load_video(pp_path, EVAL_H, EVAL_W)
            pp_metrics = evaluate(pp_frames, orig_frames, gt_masks)
            print(f"  pure_propainter: PSNR_proxy={pp_metrics['PSNR_proxy']:.3f} "
                  f"PSNR_synth={pp_metrics['PSNR_synthetic']:.3f} "
                  f"SSIM_synth={pp_metrics['SSIM_synthetic']:.4f}")

        if kf_metrics and pp_metrics:
            delta_proxy = kf_metrics["PSNR_proxy"] - pp_metrics["PSNR_proxy"]
            delta_synth = kf_metrics["PSNR_synthetic"] - pp_metrics["PSNR_synthetic"]
            print(f"\n  DELTA: proxy={delta_proxy:+.3f} dB,  synth={delta_synth:+.3f} dB")

            verdict = "POSITIVE" if delta_synth >= 0 else "NEGATIVE"
            print(f"  VERDICT: {verdict}")

            all_results[seq_name] = {
                "keyframe_sdxl": kf_metrics,
                "pure_propainter": pp_metrics,
                "delta_proxy": delta_proxy,
                "delta_synthetic": delta_synth,
                "verdict": verdict,
            }

            rows.append({
                "sequence": seq_name,
                "method": "keyframe_sdxl_propainter",
                "keyframe_interval": 10,
                "guidance_scale": 3.0,
                "PSNR_proxy": f"{kf_metrics['PSNR_proxy']:.4f}",
                "PSNR_synthetic": f"{kf_metrics['PSNR_synthetic']:.4f}",
                "SSIM_synthetic": f"{kf_metrics['SSIM_synthetic']:.6f}",
                "pure_pp_PSNR_proxy": f"{pp_metrics['PSNR_proxy']:.4f}",
                "pure_pp_PSNR_synthetic": f"{pp_metrics['PSNR_synthetic']:.4f}",
                "delta_proxy": f"{delta_proxy:+.4f}",
                "delta_synthetic": f"{delta_synth:+.4f}",
                "verdict": verdict,
            })

    # Save CSV
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "method", "keyframe_interval", "guidance_scale",
              "PSNR_proxy", "PSNR_synthetic", "SSIM_synthetic",
              "pure_pp_PSNR_proxy", "pure_pp_PSNR_synthetic",
              "delta_proxy", "delta_synthetic", "verdict"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[save] {OUT_CSV}")

    # Save JSON
    out_json = "/home/jli657/my_storage2_1T/project3/report_assets/final_delivery/direction_c_keyframe_eval.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[save] {out_json}")

    return all_results


if __name__ == "__main__":
    results = main()
    print("\n=== TASK 8 SUMMARY ===")
    for seq, r in results.items():
        print(f"  {seq}: delta_proxy={r['delta_proxy']:+.3f}  delta_synth={r['delta_synthetic']:+.3f}  [{r['verdict']}]")
