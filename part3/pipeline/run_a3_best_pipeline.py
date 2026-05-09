"""
run_a3_best_pipeline.py
Direction A-3: Assemble A+B best-per-sequence masks, run ProPainter, evaluate.

Per-sequence winner (JM):
  tennis:        A-1 SAM3 (0.9468) > B-5 (0.8735) -> A
  horsejump-low: B-5 (0.9331) > A-1 (0.8574)      -> B
  car-shadow:    B-5 (0.9785) > A-1 (0.9749)       -> B
  blackswan:     B-5 (0.9558) > A-1 (0.9549)       -> B (marginal)
  bmx-trees:     A-1 (0.7455) > B-5 (0.6887)       -> A
  koala:         A-1 (0.9482) > B-5 (0.9373)       -> A
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROPAINTER_PY = "/data2/jli657/envs/propainter_env/bin/python"
PROPAINTER_DIR = "/data2/jli657/ProPainter"
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

# A+B best mask sources per sequence
BEST_MASK_SOURCES = {
    "tennis": {
        "dir": Path("/data3/jli657/project3/part3/outputs/sam3_rebuild_v1/masks/davis5/tennis"),
        "JM": 0.9468, "method": "A-1 SAM3",
    },
    "horsejump-low": {
        "dir": Path("/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/horsejump-low"),
        "JM": 0.9331, "method": "B-5 VGGT4D+SAM3",
    },
    "car-shadow": {
        "dir": Path("/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/car-shadow"),
        "JM": 0.9785, "method": "B-5 VGGT4D+SAM3",
    },
    "blackswan": {
        "dir": Path("/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/blackswan"),
        "JM": 0.9558, "method": "B-5 VGGT4D+SAM3",
    },
    "bmx-trees": {
        "dir": Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1/bmx-trees"),
        "JM": 0.7455, "method": "A-1 GDINO+SAM2",
    },
    "koala": {
        "dir": Path("/data3/jli657/project3/part3/outputs/sam3_rebuild_v1/masks/davis5/koala"),
        "JM": 0.9482, "method": "A-1 SAM3",
    },
}

OUTPUT_ROOT = Path("/data3/jli657/project3/part3/results_v2")


def dilate_masks(src_dir: Path, dst_dir: Path, kernel_size: int = 9):
    dst_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_files = sorted(src_dir.glob("*.png"))
    for mp in mask_files:
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        dilated = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(str(dst_dir / mp.name), dilated)
    return len(mask_files)


def run_propainter(seq: str, frames_dir: Path, mask_dir: Path, out_dir: Path, gpu_id: int = 0) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    # ProPainter creates {out_dir}/{video_name}/inpaint_out.mp4
    mp4_out = out_dir / seq / "inpaint_out.mp4"
    if mp4_out.exists():
        print(f"  [{seq}] skipping (output already exists at {mp4_out})")
        return True

    dilated_dir = out_dir / "masks_dilated"
    n = dilate_masks(mask_dir, dilated_dir)
    print(f"  [{seq}] dilated {n} masks; running ProPainter...")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    script = Path(PROPAINTER_DIR) / "inference_propainter.py"
    pp_log = out_dir / f"propainter_{seq}.log"
    cmd = [
        PROPAINTER_PY, str(script),
        "--video", str(frames_dir),
        "--mask", str(dilated_dir),
        "--output", str(out_dir),
        "--resize_ratio", "1.0",
        "--neighbor_length", "10",
        "--ref_stride", "10",
    ]
    with open(pp_log, "w") as lf:
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=lf, text=True,
                                cwd=PROPAINTER_DIR)  # ensure 'weights/' resolves correctly
    if result.returncode != 0:
        print(f"  [{seq}] ProPainter failed (see {pp_log})")
        tail = open(pp_log).read()[-3000:]
        print(tail)
        return False
    print(f"  [{seq}] ProPainter done: {mp4_out}")
    return True


def compute_psnr_ssim_from_mp4(mp4_path: Path, gt_frames_dir: Path) -> dict:
    """Compute PSNR and SSIM between inpainted video and original frames."""
    if not mp4_path.exists():
        return {"psnr": -1.0, "ssim": -1.0}
    cap = cv2.VideoCapture(str(mp4_path))
    gt_files = sorted(gt_frames_dir.glob("*.jpg")) + sorted(gt_frames_dir.glob("*.png"))
    psnrs, ssims = [], []
    idx = 0
    while cap.isOpened() and idx < len(gt_files):
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= len(gt_files):
            break
        gt = cv2.imread(str(gt_files[idx]))
        if gt is None:
            idx += 1
            continue
        if frame.shape != gt.shape:
            frame = cv2.resize(frame, (gt.shape[1], gt.shape[0]))
        mse = np.mean((frame.astype(float) - gt.astype(float)) ** 2)
        psnr = 10 * np.log10(255 ** 2 / (mse + 1e-10)) if mse > 0 else 100.0
        psnrs.append(psnr)
        # SSIM
        from skimage.metrics import structural_similarity as ssim_fn
        s = ssim_fn(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
            data_range=255
        )
        ssims.append(s)
        idx += 1
    cap.release()
    return {
        "psnr": float(np.mean(psnrs)) if psnrs else -1.0,
        "ssim": float(np.mean(ssims)) if ssims else -1.0,
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
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+", default=list(BEST_MASK_SOURCES.keys()))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip_propainter", action="store_true")
    p.add_argument("--out_csv", default="/home/jli657/my_storage2_1T/project3/eval/a3_best_pipeline_results.csv")
    args = p.parse_args()

    print("\n=== Direction A-3: Best Pipeline (A+B Fusion) ===")
    rows = []

    for seq in args.sequences:
        src = BEST_MASK_SOURCES.get(seq)
        if src is None:
            print(f"[skip] {seq}: no source defined")
            continue

        mask_dir = src["dir"]
        frames_dir = DAVIS_FRAMES / seq

        if not mask_dir.exists():
            print(f"[{seq}] mask dir missing: {mask_dir}")
            continue
        if not frames_dir.exists():
            print(f"[{seq}] frames dir missing: {frames_dir}")
            continue

        out_dir = OUTPUT_ROOT / seq / "a_plus_b_best" / "propainter"

        print(f"\n[{seq}] source={src['method']}, JM={src['JM']:.4f}")
        print(f"       masks={mask_dir}")
        print(f"       output={out_dir}")

        # Run ProPainter
        if not args.skip_propainter:
            ok = run_propainter(seq, frames_dir, mask_dir, out_dir, gpu_id=args.gpu)
        else:
            ok = (out_dir / "inpaint_out.mp4").exists()

        # Evaluate
        row = {
            "sequence": seq,
            "method": src["method"],
            "mask_JM": src["JM"],
            "propainter_ok": ok,
        }

        if ok:
            mp4_path = out_dir / seq / "inpaint_out.mp4"
            metrics = compute_psnr_ssim_from_mp4(mp4_path, frames_dir)
            row.update(metrics)
            print(f"  [{seq}] PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")
        else:
            row.update({"psnr": -1.0, "ssim": -1.0})

        rows.append(row)

    # Summary
    print("\n=== A+B Best Pipeline Results ===")
    print(f"{'Seq':<20} {'Source':<22} {'Mask JM':>8} {'PSNR':>8} {'SSIM':>8}")
    print("-" * 70)
    for r in rows:
        psnr = r.get("psnr", -1.0)
        ssim = r.get("ssim", -1.0)
        print(f"  {r['sequence']:<18} {r['method']:<22} {r['mask_JM']:>8.4f} "
              f"{psnr:>8.2f} {ssim:>8.4f}")

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "method", "mask_JM", "propainter_ok", "psnr", "ssim"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
