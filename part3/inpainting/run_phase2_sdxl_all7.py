"""
run_phase2_sdxl_all7.py — SDXL kf5 + ProPainter (GT mask 公平比较协议)

mask 来源:
  - DAVIS 序列: 统一使用 DAVIS annotation / GT mask (公平 inpaint-only 对比)
  - wild_video-1person: 继续使用现有 shadow/SAM3 mask (demo 分组, 不参与 GT 对比)

输出目录:
  - DAVIS: results/<seq>/direction_c/sdxl_kf5_gtmask_propainter/
  - wild:  results/<seq>/direction_c/sdxl_kf5_propainter/  (保持原有路径)

用法:
  # GPU0 — DAVIS 序列
  PYTHONUNBUFFERED=1 conda run -n controlnet_env python3 -u \
    part3/run_phase2_sdxl_all7.py --gpu 0 \
    --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

  # GPU1 (或单独跑) wild demo
  PYTHONUNBUFFERED=1 conda run -n controlnet_env python3 -u \
    part3/run_phase2_sdxl_all7.py --gpu 1 --seqs wild_video-1person
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

# ── env setup ──────────────────────────────────────────────────────────────────
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data3/jli657/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data3/jli657/hf_cache"

import torch
torch.backends.cudnn.enabled = False

# ── paths ──────────────────────────────────────────────────────────────────────
PROPAINTER_PYTHON = "/data2/jli657/envs/propainter_env/bin/python"
PROPAINTER_DIR    = "/data2/jli657/ProPainter"
RESULTS           = Path("/data3/jli657/project3/part3/results")
MASKS_FINAL       = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final")
DAVIS_FRAMES      = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS    = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

WILD_FRAMES = Path("/data3/jli657/project3/wild_frames")

SEQUENCES_CFG = {
    "tennis": {
        "prompt": "tennis court background with no players, clean green surface",
        "wild": False,
        "frames_override": None,
    },
    "koala": {
        "prompt": "forest background with eucalyptus trees, no koala, natural scene",
        "wild": False,
        "frames_override": None,
    },
    "wild_video-1person": {
        "prompt": "clean city sidewalk, no person, no bag, no shadow, urban background",
        "wild": True,  # use wild_video-1person_with_shadow masks
        "frames_override": str(WILD_FRAMES / "wild_video-1person"),
    },
    "bmx-trees": {
        "prompt": "forest path with trees and dirt, no cyclist, natural outdoor background",
        "wild": False,
        "frames_override": None,
    },
    "blackswan": {
        "prompt": "calm lake water with reflections, no swan, natural water scene",
        "wild": False,
        "frames_override": None,
    },
    "horsejump-low": {
        "prompt": "outdoor field with grass, no horse, no rider, natural background",
        "wild": False,
        "frames_override": None,
    },
    "car-shadow": {
        "prompt": "road surface with no car, no shadow, clean street background",
        "wild": False,
        "frames_override": None,
    },
}


def load_sdxl_pipe(gpu: int):
    from diffusers import AutoPipelineForInpainting
    print(f"[SDXL] Loading {SDXL_MODEL_ID} ...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload(gpu_id=gpu)
    print(f"[SDXL] Loaded, CPU offload -> cuda:{gpu}")
    return pipe


def select_keyframes(n: int, interval: int) -> List[int]:
    return list(range(0, n, interval))


def load_sorted_imgs(d: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def load_mask_bin(path: Path, H: int, W: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((H, W), np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((H, W), np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def inpaint_sdxl(pipe, frame_bgr: np.ndarray, mask_bin: np.ndarray,
                 prompt: str, guidance: float = 3.0, steps: int = 20,
                 strength: float = 0.99) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    pil_img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_bin)
    gen = torch.Generator(device="cpu").manual_seed(42)
    result = pipe(
        prompt=prompt,
        negative_prompt="person, human, distortion, artifact, blur",
        image=pil_img,
        mask_image=pil_mask,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    ).images[0]
    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    if result_bgr.shape[:2] != (H, W):
        result_bgr = cv2.resize(result_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    return result_bgr


def run_propainter(frames_dir: str, masks_dir: str, output_dir: str) -> bool:
    env = os.environ.copy()
    cmd = [PROPAINTER_PYTHON, "inference_propainter.py",
           "--video", frames_dir, "--mask", masks_dir, "--output", output_dir,
           "--resize_ratio", "1.0", "--neighbor_length", "10", "--ref_stride", "10"]
    print(f"  [ProPainter] {frames_dir} -> {output_dir}")
    r = subprocess.run(cmd, cwd=PROPAINTER_DIR, env=env, timeout=900)
    if r.returncode != 0:
        print(f"  [ERROR] ProPainter returned {r.returncode}")
        return False
    return True


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
    # Fallback to SAM3 masks if GT not found (should not happen for DAVIS seqs)
    fallback = MASKS_FINAL / seq
    return fallback, "sam3_fallback"


def process_sequence(seq: str, cfg: dict, pipe, gpu: int, interval: int = 5):
    print(f"\n{'='*60}")
    print(f"[{seq}] SDXL kf{interval} + ProPainter")
    print(f"  prompt: {cfg['prompt']}")
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
        out_subdir = "sdxl_kf5_gtmask_propainter"
    else:
        out_subdir = "sdxl_kf5_propainter"

    out_dir = RESULTS / seq / "direction_c" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    inpaint_mp4 = out_dir / "inpaint_out.mp4"
    if inpaint_mp4.exists():
        print(f"  [skip] {seq}: {out_subdir}/inpaint_out.mp4 already exists")
        return True

    orig_frames = load_sorted_imgs(frames_dir)
    n_frames = len(orig_frames)
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: p.stem)

    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]
    print(f"  {n_frames} frames, {W}x{H}")

    keyframes = select_keyframes(n_frames, interval)
    print(f"  {len(keyframes)} keyframes (interval={interval})")

    # Step 1: SDXL inpaint keyframes
    kf_dir = out_dir / "sdxl_keyframes"
    kf_dir.mkdir(exist_ok=True)
    inpaint_cache = {}

    for idx in keyframes:
        frame = cv2.imread(str(orig_frames[idx]))
        mf = mask_files[idx] if idx < len(mask_files) else None
        mask = load_mask_bin(mf, H, W)

        if mask.sum() == 0:
            inpaint_cache[idx] = frame
            print(f"  frame {idx:04d}: no mask, skip")
            continue

        repaired = inpaint_sdxl(pipe, frame, mask, cfg["prompt"])
        inpaint_cache[idx] = repaired
        out_kf = kf_dir / f"{orig_frames[idx].stem}.png"
        cv2.imwrite(str(out_kf), repaired)
        print(f"  frame {idx:04d}: inpainted (mask {mask.mean()/255*100:.1f}%)")

    kf_count = len([p for p in kf_dir.glob("*.png")])
    print(f"  SDXL keyframes saved: {kf_count}")

    # Step 2: Build ProPainter input directory
    pp_frames_dir = out_dir / "pp_input" / "frames"
    pp_masks_dir  = out_dir / "pp_input" / "masks"
    pp_frames_dir.mkdir(parents=True, exist_ok=True)
    pp_masks_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(orig_frames):
        stem = fp.stem
        if i in inpaint_cache:
            # Keyframe: use SDXL result, set mask to 0 (anchor frame)
            cv2.imwrite(str(pp_frames_dir / f"{stem}.png"), inpaint_cache[i])
            cv2.imwrite(str(pp_masks_dir  / f"{stem}.png"), np.zeros((H, W), np.uint8))
        else:
            # Non-keyframe: copy original, keep mask
            shutil.copy2(fp, pp_frames_dir / f"{stem}.png")
            mf = mask_files[i] if i < len(mask_files) else None
            mask = load_mask_bin(mf, H, W)
            cv2.imwrite(str(pp_masks_dir / f"{stem}.png"), mask)

    # Step 3: Run ProPainter
    pp_out = out_dir / "propainter_output"
    pp_out.mkdir(exist_ok=True)
    success = run_propainter(str(pp_frames_dir), str(pp_masks_dir), str(pp_out))
    if not success:
        print(f"  [ERROR] ProPainter failed for {seq}")
        return False

    # Find output video and copy to standardized location
    candidates = [pp_out / "inpaint_out.mp4", pp_out / orig_seq_name / "inpaint_out.mp4"]
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

    # Step 4: Generate masked_in.mp4
    generate_masked_preview(masks_dir, frames_dir, out_dir / "masked_in.mp4")

    print(f"  [OK] {seq} -> {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--seqs", nargs="+", default=list(SEQUENCES_CFG.keys()))
    args = parser.parse_args()

    pipe = load_sdxl_pipe(args.gpu)

    for seq in args.seqs:
        if seq not in SEQUENCES_CFG:
            print(f"[WARN] unknown sequence: {seq}")
            continue
        process_sequence(seq, SEQUENCES_CFG[seq], pipe, args.gpu, args.interval)

    print("\n[Phase 2 complete]")


if __name__ == "__main__":
    main()
