"""
run_phase1_4seq_propainter.py — Phase 1:  4  ProPainter

: blackswan (50), bmx-trees (80), horsejump-low (60), car-shadow (40)
 GPU:  2  OOM
"""

import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np
import cv2
import imageio
import scipy.ndimage
from pathlib import Path

PROPAINTER_PY = "/data2/jli657/envs/propainter_env/bin/python"
PROPAINTER_DIR = "/data2/jli657/ProPainter"
RESULTS = Path("/data3/jli657/project3/part3/results")
MASKS_FINAL = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final")
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
PART3_SCRIPTS = Path("/home/jli657/my_storage2_1T/project3/part3")

SEQUENCES = ["blackswan", "bmx-trees", "horsejump-low", "car-shadow"]


def dilate_masks(src_mask_dir: Path, dst_mask_dir: Path, kernel_size: int = 9):
    dst_mask_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_files = sorted(src_mask_dir.glob("*.png"))
    print(f"  [dilate] {len(mask_files)} masks, kernel={kernel_size}")
    for mp in mask_files:
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        dilated = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(str(dst_mask_dir / mp.name), dilated)


def run_propainter(seq: str, gpu_id: int):
    frames_dir = DAVIS_FRAMES / seq
    masks_dir = MASKS_FINAL / seq
    out_dir = RESULTS / seq / "direction_a" / "sam3_propainter"
    out_dir.mkdir(parents=True, exist_ok=True)

    inpaint_mp4 = out_dir / "inpaint_out.mp4"
    if inpaint_mp4.exists():
        print(f"  [skip] {seq}: inpaint_out.mp4 already exists")
        return True

    # Step 1: dilate masks to a temp dir
    dilated_dir = out_dir / "masks_dilated"
    dilate_masks(masks_dir, dilated_dir, kernel_size=9)

    # Step 2: run ProPainter
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    script = Path(PROPAINTER_DIR) / "inference_propainter.py"
    cmd = [
        PROPAINTER_PY, str(script),
        "--video", str(frames_dir),
        "--mask", str(dilated_dir),
        "--output", str(out_dir),
        "--resize_ratio", "1.0",
        "--neighbor_length", "10",
        "--ref_stride", "10",
    ]
    print(f"\n[ProPainter] {seq} on GPU {gpu_id}")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROPAINTER_DIR, env=env)
    if result.returncode != 0:
        print(f"[ERROR] ProPainter failed for {seq}")
        return False

    # Step 3: generate masked_in.mp4
    generate_masked_preview(seq, masks_dir, frames_dir, out_dir / "masked_in.mp4")
    print(f"[OK] {seq} -> {out_dir}")
    return True


def generate_masked_preview(seq: str, masks_dir: Path, frames_dir: Path, output_mp4: Path):
    if output_mp4.exists():
        return
    frame_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
                          key=lambda p: p.stem)
    mask_paths = sorted(masks_dir.glob("*.png"), key=lambda p: p.stem)
    n = min(len(frame_paths), len(mask_paths))
    result_frames = []
    for i in range(n):
        frame = cv2.imread(str(frame_paths[i]))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mask_img = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            mask_img = np.zeros((h, w), np.uint8)
        else:
            mask_img = (mask_img > 127).astype(np.uint8)
            if mask_img.shape != (h, w):
                mask_img = cv2.resize(mask_img, (w, h), cv2.INTER_NEAREST)
        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=5).astype(np.uint8)
        mask_3ch = np.expand_dims(mask_img, 2).repeat(3, axis=2).astype(np.float32)
        frame_f = frame.astype(np.float32)
        green = np.zeros([h, w, 3], np.float32)
        green[:, :, 1] = 255.0
        fused = 0.4 * frame_f + 0.6 * green
        composite = mask_3ch * fused + (1 - mask_3ch) * frame_f
        result_frames.append(composite.astype(np.uint8))
    if result_frames:
        imageio.mimwrite(str(output_mp4), result_frames, fps=25.0, quality=7)
        print(f"  [preview] {len(result_frames)} frames -> {output_mp4.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument("--seqs", nargs="+", default=SEQUENCES, help="Which sequences to run")
    args = parser.parse_args()

    for seq in args.seqs:
        success = run_propainter(seq, args.gpu)
        if not success:
            print(f"[WARN] {seq} failed, continuing...")


if __name__ == "__main__":
    main()
