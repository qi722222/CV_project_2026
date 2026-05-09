"""
run_phase3_lama_all7.py — LaMa inpainting (GT mask 公平比较协议)

LaMa (Large Mask Inpainting, WACV 2022) — 专为大面积遮挡设计, Fast Fourier Convolution.
管线: 原始帧 + mask -> LaMa 逐帧修复 keyframes -> ProPainter 传播全序列

mask 来源:
  - DAVIS 序列: 统一使用 DAVIS annotation / GT mask (公平 inpaint-only 对比)
  - wild_video-1person: 保留现有 shadow/SAM3 mask (demo 分组)

输出目录:
  - DAVIS: results/<seq>/direction_c/lama_gtmask_propainter/
  - wild:  results/<seq>/direction_c/lama_propainter/  (保持原有路径)

注意: LaMa 的 JIT 模型在当前 cuDNN 版本下有兼容性问题.
     使用 CUDA_VISIBLE_DEVICES='' 强制 CPU 模式, 在 240p 下处理后 resize 到原始分辨率.
     CPU 模式约 8s/帧 (240p), 80 帧 ≈ 11 分钟/序列.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

# Force CPU for LaMa (JIT cuDNN compat issue on this system)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "/data3/jli657/hf_cache"

PROPAINTER_PYTHON = "/data2/jli657/envs/propainter_env/bin/python"
PROPAINTER_DIR    = "/data2/jli657/ProPainter"
RESULTS           = Path("/data3/jli657/project3/part3/results")
MASKS_FINAL       = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final")
DAVIS_FRAMES      = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS    = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

WILD_FRAMES = Path("/data3/jli657/project3/wild_frames")


def resolve_mask_dir(seq: str, cfg: dict) -> tuple[Path, str]:
    """Return (mask_dir, mask_protocol) for a sequence.

    DAVIS sequences: use DAVIS GT annotations (fair inpaint-only comparison).
    wild_video-1person: use existing shadow/SAM3 mask (demo group, no GT available).
    """
    if cfg.get("wild"):
        wild_mask = MASKS_FINAL / "wild_video-1person_with_shadow"
        if not wild_mask.exists():
            wild_mask = MASKS_FINAL / "wild_video-1person"
        return wild_mask, "wild_existing_mask"
    gt_dir = DAVIS_GT_MASKS / seq
    if gt_dir.exists():
        return gt_dir, "davis_gt"
    fallback = MASKS_FINAL / seq
    return fallback, "sam3_fallback"

SEQUENCES_CFG = {
    "tennis":           {"wild": False, "frames_override": None},
    "koala":            {"wild": False, "frames_override": None},
    "wild_video-1person": {"wild": True, "frames_override": str(WILD_FRAMES / "wild_video-1person")},
    "bmx-trees":        {"wild": False, "frames_override": None},
    "blackswan":        {"wild": False, "frames_override": None},
    "horsejump-low":    {"wild": False, "frames_override": None},
    "car-shadow":       {"wild": False, "frames_override": None},
}

# Process at 240p then resize to original for speed (4x faster than 480p)
LAMA_PROCESS_H = 256
LAMA_PROCESS_W = 448


def load_sorted_imgs(d: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def generate_masked_preview(masks_dir: Path, frames_dir: Path, output_mp4: Path):
    if output_mp4.exists():
        return
    import imageio
    frame_paths = load_sorted_imgs(frames_dir)
    mask_paths  = sorted(masks_dir.glob("*.png"), key=lambda p: p.stem)
    n = min(len(frame_paths), len(mask_paths))
    result_frames = []
    for i in range(n):
        frame = cv2.imread(str(frame_paths[i]))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        msk = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.zeros((h, w), np.uint8)
        else:
            msk = (msk > 127).astype(np.uint8)
            if msk.shape != (h, w):
                msk = cv2.resize(msk, (w, h), cv2.INTER_NEAREST)
        msk = scipy.ndimage.binary_dilation(msk, iterations=5).astype(np.uint8)
        msk3 = np.expand_dims(msk, 2).repeat(3, axis=2).astype(np.float32)
        ff = frame.astype(np.float32)
        g = np.zeros([h, w, 3], np.float32); g[:, :, 1] = 255.0
        composite = msk3 * (0.4 * ff + 0.6 * g) + (1 - msk3) * ff
        result_frames.append(composite.astype(np.uint8))
    if result_frames:
        imageio.mimwrite(str(output_mp4), result_frames, fps=25.0, quality=7)
        print(f"  [preview] {len(result_frames)} frames -> {output_mp4.name}")


def lama_inpaint_frame(lama, frame_bgr: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    """Inpaint one frame with LaMa. Resizes to LAMA_PROCESS resolution and back."""
    H, W = frame_bgr.shape[:2]

    # Resize to process resolution
    frame_small = cv2.resize(frame_bgr, (LAMA_PROCESS_W, LAMA_PROCESS_H), cv2.INTER_LINEAR)
    mask_small  = cv2.resize(mask_bin,  (LAMA_PROCESS_W, LAMA_PROCESS_H), cv2.INTER_NEAREST)

    pil_img  = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_small)

    result_pil = lama(pil_img, pil_mask)
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # Resize back to original resolution
    result_full = cv2.resize(result_bgr, (W, H), cv2.INTER_LINEAR)
    return result_full


def run_propainter(frames_dir: str, masks_dir: str, output_dir: str) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # let propainter use GPU 0
    cmd = [PROPAINTER_PYTHON, "inference_propainter.py",
           "--video", frames_dir, "--mask", masks_dir, "--output", output_dir,
           "--resize_ratio", "1.0", "--neighbor_length", "10", "--ref_stride", "10"]
    print(f"  [ProPainter] {frames_dir} -> {output_dir}")
    r = subprocess.run(cmd, cwd=PROPAINTER_DIR, env=env, timeout=900)
    return r.returncode == 0


def process_sequence(seq: str, cfg: dict, lama, interval: int = 5):
    print(f"\n{'='*60}")
    print(f"[{seq}] LaMa kf{interval} + ProPainter")
    print('='*60)

    orig_seq_name = "wild_video-1person" if cfg.get("wild") else seq

    if cfg.get("frames_override"):
        frames_dir = Path(cfg["frames_override"])
    else:
        frames_dir = DAVIS_FRAMES / orig_seq_name

    masks_dir, mask_protocol = resolve_mask_dir(seq, cfg)
    print(f"  [mask_protocol={mask_protocol}] {masks_dir}")

    # Output directory: GT-mask runs get a distinct subdirectory to preserve old results
    if mask_protocol == "davis_gt":
        out_subdir = "lama_gtmask_propainter"
    else:
        out_subdir = "lama_propainter"

    out_dir = RESULTS / seq / "direction_c" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    inpaint_mp4 = out_dir / "inpaint_out.mp4"
    if inpaint_mp4.exists():
        print(f"  [skip] {out_subdir}/inpaint_out.mp4 already exists")
        return True

    orig_frames = load_sorted_imgs(frames_dir)
    n_frames = len(orig_frames)
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: p.stem)

    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]
    print(f"  {n_frames} frames, {W}x{H}, processing at {LAMA_PROCESS_W}x{LAMA_PROCESS_H}")

    keyframes = list(range(0, n_frames, interval))
    print(f"  {len(keyframes)} keyframes (interval={interval})")

    # Step 1: LaMa inpaint keyframes
    kf_dir = out_dir / "lama_keyframes"
    kf_dir.mkdir(exist_ok=True)
    inpaint_cache = {}

    for idx in keyframes:
        frame = cv2.imread(str(orig_frames[idx]))
        mf = mask_files[idx] if idx < len(mask_files) else None
        if mf is None or not mf.exists():
            mask_bin = np.zeros((H, W), np.uint8)
        else:
            m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            mask_bin = (m > 127).astype(np.uint8) * 255 if m is not None else np.zeros((H, W), np.uint8)

        if mask_bin.sum() == 0:
            inpaint_cache[idx] = frame
            continue

        repaired = lama_inpaint_frame(lama, frame, mask_bin)
        inpaint_cache[idx] = repaired
        cv2.imwrite(str(kf_dir / f"{orig_frames[idx].stem}.png"), repaired)
        if idx % 5 == 0:
            print(f"  frame {idx:04d}: lama inpainted (mask {mask_bin.mean()/255*100:.1f}%)")

    print(f"  LaMa keyframes saved: {len([p for p in kf_dir.glob('*.png')])}")

    # Step 2: Build ProPainter input
    pp_frames_dir = out_dir / "pp_input" / "frames"
    pp_masks_dir  = out_dir / "pp_input" / "masks"
    pp_frames_dir.mkdir(parents=True, exist_ok=True)
    pp_masks_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(orig_frames):
        stem = fp.stem
        if i in inpaint_cache:
            cv2.imwrite(str(pp_frames_dir / f"{stem}.png"), inpaint_cache[i])
            cv2.imwrite(str(pp_masks_dir  / f"{stem}.png"), np.zeros((H, W), np.uint8))
        else:
            shutil.copy2(fp, pp_frames_dir / f"{stem}.png")
            mf = mask_files[i] if i < len(mask_files) else None
            if mf and mf.exists():
                m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                mask_bin = (m > 127).astype(np.uint8) * 255 if m is not None else np.zeros((H, W), np.uint8)
            else:
                mask_bin = np.zeros((H, W), np.uint8)
            cv2.imwrite(str(pp_masks_dir / f"{stem}.png"), mask_bin)

    # Step 3: ProPainter propagation
    pp_out = out_dir / "propainter_output"
    pp_out.mkdir(exist_ok=True)
    success = run_propainter(str(pp_frames_dir), str(pp_masks_dir), str(pp_out))
    if not success:
        print(f"  [ERROR] ProPainter failed for {seq}")
        return False

    # Find output video
    candidates = [pp_out / "inpaint_out.mp4", pp_out / orig_seq_name / "inpaint_out.mp4",
                  pp_out / "frames" / "inpaint_out.mp4"]
    pp_video = None
    for c in candidates:
        if c.exists():
            pp_video = c; break
    if pp_video is None:
        mp4s = list(pp_out.rglob("inpaint_out.mp4"))
        if mp4s:
            pp_video = mp4s[0]

    if pp_video and pp_video != inpaint_mp4:
        shutil.copy2(pp_video, inpaint_mp4)
        print(f"  [copy] inpaint_out.mp4 -> {inpaint_mp4}")
    elif inpaint_mp4.exists():
        pass
    else:
        print(f"  [ERROR] Could not find ProPainter output video for {seq}")
        return False

    # Step 4: masked_in.mp4
    generate_masked_preview(masks_dir, frames_dir, out_dir / "masked_in.mp4")
    print(f"  [OK] {seq} -> {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--seqs", nargs="+", default=list(SEQUENCES_CFG.keys()))
    args = parser.parse_args()

    from simple_lama_inpainting import SimpleLama
    import torch
    print("Loading LaMa model (CPU mode)...")
    lama = SimpleLama(device=torch.device("cpu"))
    print("LaMa ready (CPU)")

    for seq in args.seqs:
        if seq not in SEQUENCES_CFG:
            print(f"[WARN] unknown sequence: {seq}")
            continue
        try:
            process_sequence(seq, SEQUENCES_CFG[seq], lama, args.interval)
        except Exception as e:
            print(f"[ERROR] {seq}: {e}")

    print("\n[Phase 3 complete]")


if __name__ == "__main__":
    main()
