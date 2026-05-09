"""
run_koala_diffusion_inpaint.py — Direction C: Large-Mask Diffusion Inpainting

Motivation:
  koala 序列 mask 平均覆盖 26.3%（最高帧 57.6%），ProPainter 因传播锚点严重不足
  导致视觉效果差。本脚本实验两种 SDXL 策略:

  1. sdxl_kf5  — SDXL keyframe (interval=5) + ProPainter propagation
     对每隔5帧的关键帧用 SDXL 生成背景，其余帧交由 ProPainter 传播
  2. sdxl_perframe — SDXL inpaint on every frame independently
     逐帧 SDXL 修复，无 ProPainter 传播，评估"语义生成 vs 时序一致性"权衡

对比:
  - Baseline: pure ProPainter (已有)
  - sdxl_kf5: SDXL 关键帧 + ProPainter 传播（期望: 提升大遮罩帧语义质量）
  - sdxl_perframe: 全帧 SDXL（期望: 最强单帧语义但无时序）

用法:
  conda run -n controlnet_env python3 part3/run_koala_diffusion_inpaint.py \
    --mode kf5 --gpu 1

  conda run -n controlnet_env python3 part3/run_koala_diffusion_inpaint.py \
    --mode perframe --gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("HF_HOME", "/data3/jli657/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data3/jli657/hf_cache")

import torch

torch.backends.cudnn.enabled = False

PROPAINTER_DIR = "/data2/jli657/ProPainter"
PROPAINTER_PYTHON = "/data2/jli657/envs/propainter_env/bin/python"

SEQ_CFG = {
    "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/koala",
    "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/koala",
    "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/koala",
    "pure_pp_video": (
        "/data3/jli657/project3/part3/outputs/sam3_multiobj/propainter/koala/koala/inpaint_out.mp4"
    ),
    # Semantically accurate prompt for what should be behind the koala
    "inpaint_prompt": (
        "eucalyptus tree branches with leaves, natural forest background, "
        "no animals, empty branch, outdoors, high quality, natural lighting"
    ),
    "negative_prompt": (
        "koala, bear, animal, fur, claws, distortion, blur, artifact, "
        "low quality, watermark"
    ),
}

OUTPUT_ROOT = Path("/data3/jli657/project3/part3/outputs/koala_diffusion")
EVAL_OUT = Path("/data3/jli657/project3/part3/outputs/koala_diffusion/eval_results.json")

GENERATE_PREVIEW = Path("/home/jli657/my_storage2_1T/project3/part3/generate_mask_preview.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["kf5", "kf10", "perframe"],
        default="kf5",
        help="kf5=keyframe interval 5, kf10=interval 10, perframe=every frame",
    )
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--strength", type=float, default=0.99)
    return p.parse_args()


def load_sdxl(gpu: int):
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    print(f"[SDXL loaded] device=cuda:{gpu}")
    return pipe


def load_frames(d: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def load_mask(path: Path, H: int, W: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((H, W), dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((H, W), dtype=np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def inpaint_with_sdxl(
    pipe,
    bgr: np.ndarray,
    mask_bin: np.ndarray,
    prompt: str,
    neg_prompt: str,
    guidance: float,
    steps: int,
    strength: float,
    seed: int = 42,
) -> np.ndarray:
    H, W = bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_bin)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
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


def run_propainter(frames_dir: str, masks_dir: str, output_dir: str) -> Path | None:
    cmd = [
        PROPAINTER_PYTHON, "inference_propainter.py",
        "--video", frames_dir,
        "--mask", masks_dir,
        "--output", output_dir,
        "--fp16",
    ]
    r = subprocess.run(cmd, cwd=PROPAINTER_DIR, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"  ProPainter FAILED:\n{r.stderr[-500:]}")
        return None
    # Find output
    for c in [
        Path(output_dir) / "inpaint_out.mp4",
        Path(output_dir) / "koala" / "inpaint_out.mp4",
    ]:
        if c.exists():
            return c
    mp4s = list(Path(output_dir).rglob("inpaint_out.mp4"))
    return mp4s[0] if mp4s else None


def generate_preview(orig_dir: Path, mask_dir: Path, out_mp4: Path) -> bool:
    cmd = [
        sys.executable, str(GENERATE_PREVIEW),
        "--frames_dir", str(orig_dir),
        "--masks_dir", str(mask_dir),
        "--output_mp4", str(out_mp4),
        "--fps", "24",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  [WARN] Preview failed: {r.stderr[-200:]}")
        return False
    return True


def save_video(frames: List[np.ndarray], out_path: Path, fps: int = 24):
    """Save frame list as mp4."""
    import imageio

    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimwrite(str(out_path), rgb_frames, fps=fps, quality=8)
    print(f"  [saved] {out_path} ({len(frames)} frames)")


def psnr(a: np.ndarray, b: np.ndarray, roi: Optional[np.ndarray] = None) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    if roi is not None:
        r = roi > 127
        if r.sum() == 0:
            return float("nan")
        diff = diff[r]
    mse = np.mean(diff ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


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


def evaluate_vs_baseline(
    pred_frames: List[np.ndarray],
    orig_paths: List[Path],
    gt_mask_paths: List[Path],
    pp_frames: List[np.ndarray],
    eval_H: int,
    eval_W: int,
) -> dict:
    """
    Compute PSNR_proxy (non-masked), masked_diff_mean, and compare vs ProPainter.
    Use GT mask from DAVIS annotations.
    """
    n = min(len(pred_frames), len(orig_paths), len(gt_mask_paths), len(pp_frames))
    proxy_ours, proxy_pp = [], []
    masked_diff_ours, masked_diff_pp = [], []

    for i in range(n):
        pred = cv2.resize(pred_frames[i], (eval_W, eval_H)) if pred_frames[i].shape[:2] != (eval_H, eval_W) else pred_frames[i]
        orig = cv2.imread(str(orig_paths[i]))
        pp = cv2.resize(pp_frames[i], (eval_W, eval_H)) if pp_frames[i].shape[:2] != (eval_H, eval_W) else pp_frames[i]
        if orig is None:
            continue
        orig = cv2.resize(orig, (eval_W, eval_H))

        gt_m = cv2.imread(str(gt_mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if gt_m is None:
            continue
        gt_m = cv2.resize(gt_m, (eval_W, eval_H), interpolation=cv2.INTER_NEAREST)
        mask_bin = (gt_m > 0).astype(np.uint8) * 255
        proxy_mask = 255 - mask_bin

        proxy_ours.append(psnr(pred, orig, roi=proxy_mask))
        proxy_pp.append(psnr(pp, orig, roi=proxy_mask))

        diff_ours = np.abs(pred.astype(float) - orig.astype(float))
        diff_pp = np.abs(pp.astype(float) - orig.astype(float))
        if mask_bin.mean() > 0:
            masked_diff_ours.append(diff_ours[mask_bin > 127].mean())
            masked_diff_pp.append(diff_pp[mask_bin > 127].mean())

    def safe_mean(lst):
        lst = [x for x in lst if not np.isnan(x)]
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "n": n,
        "PSNR_proxy_ours": safe_mean(proxy_ours),
        "PSNR_proxy_pp": safe_mean(proxy_pp),
        "delta_proxy": safe_mean(proxy_ours) - safe_mean(proxy_pp),
        "masked_diff_ours": safe_mean(masked_diff_ours),
        "masked_diff_pp": safe_mean(masked_diff_pp),
        "masked_diff_improvement": safe_mean(masked_diff_pp) - safe_mean(masked_diff_ours),
    }


def run_kf_mode(args, pipe, interval: int, mode_name: str) -> dict:
    """SDXL keyframe inpainting + ProPainter propagation."""
    print(f"\n{'='*65}")
    print(f"  Mode: {mode_name} (interval={interval})")
    print(f"{'='*65}")

    orig_dir = Path(SEQ_CFG["orig_dir"])
    mask_dir = Path(SEQ_CFG["mask_dir"])
    out_dir = OUTPUT_ROOT / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_frames = load_frames(orig_dir)
    mask_files = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)
    n = len(orig_frames)
    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]

    keyframes = list(range(0, n, interval))
    print(f"  {n} frames, {W}x{H}, {len(keyframes)} keyframes")

    # Step 1: SDXL inpaint keyframes
    kf_dir = out_dir / "sdxl_keyframes"
    kf_dir.mkdir(exist_ok=True)
    kf_results: dict[int, np.ndarray] = {}

    for idx in keyframes:
        frame = cv2.imread(str(orig_frames[idx]))
        mf = mask_files[idx] if idx < len(mask_files) else None
        mask = load_mask(mf, H, W)
        coverage = (mask > 127).mean() * 100

        if coverage < 1.0:
            kf_results[idx] = frame
            print(f"  frame {idx:3d}: skip (mask={coverage:.1f}%)")
            continue

        repaired = inpaint_with_sdxl(
            pipe, frame, mask,
            SEQ_CFG["inpaint_prompt"],
            SEQ_CFG["negative_prompt"],
            args.guidance, args.steps, args.strength,
            seed=42 + idx,
        )
        kf_results[idx] = repaired
        cv2.imwrite(str(kf_dir / f"{orig_frames[idx].stem}.png"), repaired)
        print(f"  frame {idx:3d}: SDXL inpainted (mask={coverage:.1f}%)")

    # Step 2: Build ProPainter input
    pp_in_frames = out_dir / "pp_input" / "frames"
    pp_in_masks = out_dir / "pp_input" / "masks"
    pp_in_frames.mkdir(parents=True, exist_ok=True)
    pp_in_masks.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(orig_frames):
        stem = fp.stem
        if i in kf_results:
            cv2.imwrite(str(pp_in_frames / f"{stem}.png"), kf_results[i])
            # Zero mask → ProPainter treats as known content
            cv2.imwrite(str(pp_in_masks / f"{stem}.png"), np.zeros((H, W), dtype=np.uint8))
        else:
            shutil.copy2(fp, pp_in_frames / f"{stem}.png")
            mf = mask_files[i] if i < len(mask_files) else None
            cv2.imwrite(str(pp_in_masks / f"{stem}.png"), load_mask(mf, H, W))

    # Step 3: ProPainter propagation
    pp_out = str(out_dir / "propainter_output")
    Path(pp_out).mkdir(exist_ok=True)
    print("  Running ProPainter propagation...")
    pp_video = run_propainter(str(pp_in_frames), str(pp_in_masks), pp_out)
    if pp_video is None:
        print("  [ERROR] ProPainter failed")
        return {"mode": mode_name, "error": "ProPainter failed"}

    # Copy outputs to final_output dir
    final_dir = out_dir / "final_output"
    final_dir.mkdir(exist_ok=True)
    shutil.copy2(pp_video, final_dir / "inpaint_out.mp4")

    # Generate masked_in.mp4 preview
    preview_mp4 = final_dir / "masked_in.mp4"
    generate_preview(orig_dir, mask_dir, preview_mp4)

    print(f"  [outputs] {final_dir}")
    print(f"    inpaint_out.mp4 : {(final_dir / 'inpaint_out.mp4').exists()}")
    print(f"    masked_in.mp4   : {preview_mp4.exists()}")

    return {
        "mode": mode_name,
        "interval": interval,
        "inpaint_out": str(final_dir / "inpaint_out.mp4"),
        "masked_in": str(preview_mp4),
    }


def run_perframe_mode(args, pipe, mode_name: str = "sdxl_perframe") -> dict:
    """SDXL inpainting on every frame independently (no ProPainter)."""
    print(f"\n{'='*65}")
    print(f"  Mode: {mode_name} (all {args.steps} steps per frame)")
    print(f"{'='*65}")

    orig_dir = Path(SEQ_CFG["orig_dir"])
    mask_dir = Path(SEQ_CFG["mask_dir"])
    out_dir = OUTPUT_ROOT / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_frames = load_frames(orig_dir)
    mask_files = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)
    n = len(orig_frames)
    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]
    print(f"  {n} frames, {W}x{H}")

    result_frames = []
    for i, fp in enumerate(orig_frames):
        frame = cv2.imread(str(fp))
        mf = mask_files[i] if i < len(mask_files) else None
        mask = load_mask(mf, H, W)
        coverage = (mask > 127).mean() * 100

        if coverage < 1.0:
            result_frames.append(frame)
            continue

        repaired = inpaint_with_sdxl(
            pipe, frame, mask,
            SEQ_CFG["inpaint_prompt"],
            SEQ_CFG["negative_prompt"],
            args.guidance, args.steps, args.strength,
            seed=42,  # fixed seed for consistency
        )
        result_frames.append(repaired)
        if i % 10 == 0:
            print(f"  frame {i:3d}/{n}: mask={coverage:.1f}%")

    # Save output video
    final_dir = out_dir / "final_output"
    final_dir.mkdir(exist_ok=True)
    save_video(result_frames, final_dir / "inpaint_out.mp4")

    # Generate masked_in.mp4 preview
    preview_mp4 = final_dir / "masked_in.mp4"
    generate_preview(orig_dir, mask_dir, preview_mp4)

    print(f"  [outputs] {final_dir}")
    return {
        "mode": mode_name,
        "inpaint_out": str(final_dir / "inpaint_out.mp4"),
        "masked_in": str(preview_mp4),
    }


def do_evaluation(results: list[dict]) -> dict:
    """Compare all methods vs ProPainter baseline at unified resolution."""
    print(f"\n{'='*65}")
    print("  Evaluation (all methods vs pure ProPainter baseline)")
    print(f"{'='*65}")

    orig_dir = Path(SEQ_CFG["orig_dir"])
    mask_dir = Path(SEQ_CFG["mask_dir"])
    gt_mask_dir = Path(SEQ_CFG["gt_mask_dir"])
    pure_pp_path = Path(SEQ_CFG["pure_pp_video"])

    orig_paths = load_frames(orig_dir)
    gt_mask_paths = sorted(gt_mask_dir.glob("*.png"), key=lambda p: p.stem)
    pp_frames = load_video_frames(pure_pp_path)

    # Unified eval resolution (ProPainter internal: 368x640)
    eval_H, eval_W = 368, 640

    print(f"  Pure ProPainter video: {len(pp_frames)} frames @ {pp_frames[0].shape[:2] if pp_frames else 'N/A'}")

    eval_results = {}
    for res in results:
        mode = res.get("mode", "unknown")
        video_path = Path(res.get("inpaint_out", ""))
        if not video_path.exists():
            print(f"  [skip] {mode}: no video at {video_path}")
            continue

        pred_frames = load_video_frames(video_path)
        n_eval = min(len(pred_frames), len(orig_paths), len(gt_mask_paths), len(pp_frames))
        print(f"  {mode}: {len(pred_frames)} frames (eval on {n_eval})")

        ev = evaluate_vs_baseline(pred_frames, orig_paths, gt_mask_paths, pp_frames, eval_H, eval_W)
        eval_results[mode] = ev
        print(f"    PSNR_proxy_ours={ev['PSNR_proxy_ours']:.2f}  PP={ev['PSNR_proxy_pp']:.2f}  delta={ev['delta_proxy']:+.2f}")
        print(f"    masked_diff: ours={ev['masked_diff_ours']:.1f}  PP={ev['masked_diff_pp']:.1f}  improvement={ev['masked_diff_improvement']:+.1f}")

    return eval_results


def main():
    args = parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  koala Large-Mask Diffusion Inpainting Experiment")
    print(f"  Mode: {args.mode}  GPU: {args.gpu}")
    print(f"  SDXL: guidance={args.guidance}, steps={args.steps}, strength={args.strength}")
    print(f"{'='*65}")

    # Print baseline info
    pure_pp = Path(SEQ_CFG["pure_pp_video"])
    if pure_pp.exists():
        pp_frames = load_video_frames(pure_pp)
        print(f"\n  [Baseline] Pure ProPainter: {len(pp_frames)} frames @ {pp_frames[0].shape if pp_frames else 'N/A'}")
        print(f"    Note: PSNR_proxy was incorrectly reported as 10.87 due to resolution mismatch")
        print(f"    Correct PSNR_proxy (non-masked) ≈ 31.3 dB")
        print(f"    Issue: masked region (avg 26.3%, max 57.6%) generated from scratch by ProPainter")

    # Load SDXL
    pipe = load_sdxl(args.gpu)

    # Run selected mode
    results = []
    if args.mode in ("kf5", "kf10"):
        interval = 5 if args.mode == "kf5" else 10
        res = run_kf_mode(args, pipe, interval, args.mode)
        results.append(res)
    elif args.mode == "perframe":
        res = run_perframe_mode(args)
        results.append(res)

    # Evaluation
    if results:
        eval_results = do_evaluation(results)

        # Load existing eval file if present, merge
        if EVAL_OUT.exists():
            with open(EVAL_OUT) as f:
                existing = json.load(f)
        else:
            existing = {"mode_results": {}, "run_log": []}

        existing["mode_results"].update(eval_results)
        existing["run_log"].append({
            "mode": args.mode,
            "steps": args.steps,
            "guidance": args.guidance,
            "outputs": results,
        })

        EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(EVAL_OUT, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\n[saved] {EVAL_OUT}")

    print(f"\n{'='*65}")
    print("  All outputs include inpaint_out.mp4 + masked_in.mp4")
    print(f"  Results dir: {OUTPUT_ROOT}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
