"""
run_void_koala.py — VOID (Netflix) inference on koala sequence

VOID: Video Object and Interaction Deletion
  Base: CogVideoX-Fun-V1.5-5b-InP (5B params, BF16)
  Fine-tuned: void_pass1.safetensors for video object removal

此脚本:
  1. 将 koala 原始视频裁剪为 VOID 所支持格式 (384x672, ≤72帧)
  2. 将 ProPainter 二值 mask 转为 VOID quadmask (0=remove, 255=keep)
  3. 通过 VOID pipeline 生成 koala-removed 视频
  4. 输出 inpaint_out.mp4 + masked_in.mp4 并与 ProPainter 对比

用法:
  conda run -n controlnet_env python3 part3/run_void_koala.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# cuDNN 9.x on this system has a bug where CUDNN_STATUS_NOT_INITIALIZED is raised
# for 3D convolutions. Disable cuDNN to use PyTorch's built-in fallback kernels.
# This is slower but fully functional for inference.
torch.backends.cudnn.enabled = False

os.environ.setdefault("HF_HOME", "/data3/jli657/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── paths ─────────────────────────────────────────────────────────────────────
VOID_CODE = Path("/data3/jli657/void-model/code")
BASE_MODEL = Path("/data3/jli657/void-model/CogVideoX-Fun-V1.5-5b-InP")
VOID_CKPT  = Path("/data3/jli657/void-model/checkpoints/void_pass1.safetensors")

KOALA_ORIG  = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/koala")
KOALA_MASK  = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/koala")
KOALA_GT    = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p/koala")
PP_BASELINE = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/propainter/koala/koala/inpaint_out.mp4")

OUTPUT_DIR  = Path("/data3/jli657/project3/part3/outputs/koala_diffusion/void/final_output")
PREVIEW_SCRIPT = Path("/home/jli657/my_storage2_1T/project3/part3/generate_mask_preview.py")

# VOID inference constants
SAMPLE_SIZE  = (384, 672)  # H × W (VOID native resolution)
TEMPORAL_WIN = 72           # must give even T_lat: T_lat=ceil(ceil(T/2)/2). 72→18 (even) ✓
FPS          = 12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_sorted_frames(d: Path, exts=(".jpg", ".jpeg", ".png")) -> list[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def frames_to_tensor(frame_paths: list[Path], max_frames: int = TEMPORAL_WIN) -> torch.Tensor:
    """Load frames as (1, C, T, H, W) float32 [0,1] resized to SAMPLE_SIZE."""
    frames = []
    for fp in frame_paths[:max_frames]:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
    t = torch.from_numpy(frames_np).permute(3, 0, 1, 2)             # (C, T, H, W)
    t = F.interpolate(
        t.unsqueeze(0),  # (1, C, T, H, W)  - needs (N, C, *) for 3D interpolate
        size=None, scale_factor=None,
        # Use 2D interpolation per frame
    ).squeeze(0)
    # Resize spatially using cv2 per frame
    return t  # placeholder, actual resize below


def load_video_tensor(frame_paths: list[Path], max_frames: int = TEMPORAL_WIN) -> torch.Tensor:
    """Return (1, C, T, H, W) float32 [0,1] at VOID resolution."""
    H_out, W_out = SAMPLE_SIZE
    frames = []
    for fp in frame_paths[:max_frames]:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.resize(img, (W_out, H_out), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(img)
    t = torch.from_numpy(np.stack(frames, 0))   # (T, H, W, C)
    t = t.permute(3, 0, 1, 2).unsqueeze(0)      # (1, C, T, H, W)
    return t


def load_mask_tensor(mask_paths: list[Path], max_frames: int = TEMPORAL_WIN) -> torch.Tensor:
    """
    Return quadmask as (1, 1, T, H, W) float32 [0,1] at VOID resolution.
    Our mask: 255=koala(remove), 0=background(keep)
    VOID quadmask convention: 0=remove, 255=keep  →  invert
    Then normalize to [0,1].
    """
    H_out, W_out = SAMPLE_SIZE
    frames = []
    for mp in mask_paths[:max_frames]:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros((H_out, W_out), dtype=np.uint8)
        else:
            m = cv2.resize(m, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
        # Convert: 255(koala→remove) → 0, 0(bg→keep) → 255
        quadmask = 255 - (m > 127).astype(np.uint8) * 255
        frames.append(quadmask.astype(np.float32) / 255.0)

    t = torch.from_numpy(np.stack(frames, 0))    # (T, H, W)
    t = t.unsqueeze(0).unsqueeze(0)              # (1, 1, T, H, W)
    return t


def temporal_pad(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad temporal dimension if shorter than target (repeat last frame)."""
    T = tensor.shape[2]
    if T >= target_frames:
        return tensor[:, :, :target_frames]
    pad_n = target_frames - T
    last = tensor[:, :, -1:].repeat(1, 1, pad_n, 1, 1)
    return torch.cat([tensor, last], dim=2)


def load_void_pipeline(gpu: int):
    """Load VOID pipeline from local weights."""
    sys.path.insert(0, str(VOID_CODE))
    from videox_fun.models import (
        AutoencoderKLCogVideoX,
        CogVideoXTransformer3DModel,
        T5EncoderModel,
        T5Tokenizer,
    )
    from videox_fun.pipeline import CogVideoXFunInpaintPipeline
    from safetensors.torch import load_file

    WEIGHT_DTYPE = torch.bfloat16
    base = str(BASE_MODEL)

    print(f"  Loading transformer from {BASE_MODEL}/transformer (void_pass1 symlinked)...")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        base,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=WEIGHT_DTYPE,
        use_vae_mask=True,
        stack_mask=False,
    ).to(WEIGHT_DTYPE)
    print("  Transformer loaded OK")

    # Load VOID fine-tuned weights (void_pass1.safetensors is already linked as
    # diffusion_pytorch_model.safetensors, so weights are already loaded above.
    # Re-apply in case of architecture patch mismatch.)
    print("  Verifying VOID weights (re-applying from void_pass1.safetensors)...")
    state_dict = load_file(str(VOID_CKPT))
    state_dict = state_dict.get("state_dict", state_dict)

    param_name = "patch_embed.proj.weight"
    if state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1):
        feat_dim   = 16 * 8
        new_weight = transformer.state_dict()[param_name].clone()
        new_weight[:, :feat_dim]  = state_dict[param_name][:, :feat_dim]
        new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
        state_dict[param_name] = new_weight
        print(f"  patch_embed patched")

    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"  VOID state_dict applied: missing={len(missing)}, unexpected={len(unexpected)}")

    print("  Loading VAE, tokenizer, text_encoder, scheduler...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        base, subfolder="vae"
    ).to(WEIGHT_DTYPE)
    tokenizer    = T5Tokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE
    )
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained(base, subfolder="scheduler")

    pipeline = CogVideoXFunInpaintPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    # CPU offloading: keeps components in RAM, moves to GPU only when needed.
    # Required since 5B model (BF16 ~20 GB) + 72-frame activations is large.
    pipeline.enable_model_cpu_offload(gpu_id=gpu)
    print(f"  VOID pipeline ready (CPU offload → cuda:{gpu})")
    return pipeline


def run_void_inference(pipeline, frame_paths, mask_paths, prompt, neg_prompt,
                       steps, guidance, seed):
    """Run VOID on a batch of frames, return output frames (list of np.ndarray BGR)."""
    from videox_fun.utils.utils import temporal_padding as tp_fn

    video_tensor = load_video_tensor(frame_paths, TEMPORAL_WIN)   # (1, C, T, H, W)
    mask_tensor  = load_mask_tensor(mask_paths, TEMPORAL_WIN)     # (1, 1, T, H, W)

    # temporal_padding to at least TEMPORAL_WIN frames
    video_tensor = temporal_pad(video_tensor, TEMPORAL_WIN)
    mask_tensor  = temporal_pad(mask_tensor, TEMPORAL_WIN)

    T_actual = min(len(frame_paths), TEMPORAL_WIN)

    gen = torch.Generator(device="cpu").manual_seed(seed)

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=SAMPLE_SIZE[0],
            width=SAMPLE_SIZE[1],
            num_frames=TEMPORAL_WIN,
            video=video_tensor,
            mask_video=mask_tensor,
            generator=gen,
            guidance_scale=guidance,
            num_inference_steps=steps,
            strength=1.0,
            use_trimask=True,
            use_vae_mask=True,
            stack_mask=False,
            zero_out_mask_region=False,
        ).videos

    # result: (1, C, T, H, W) float32 [0,1]
    frames = result[0].permute(1, 2, 3, 0).cpu().float().numpy()[:T_actual]
    frames = (frames * 255).clip(0, 255).astype(np.uint8)  # (T, H, W, C) RGB

    # Resize back to original koala resolution (480, 854)
    orig_ref = cv2.imread(str(frame_paths[0]))
    H_orig, W_orig = orig_ref.shape[:2]
    out_frames = []
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (W_orig, H_orig), interpolation=cv2.INTER_LANCZOS4)
        out_frames.append(bgr)
    return out_frames


def compute_psnr(a, b, roi=None):
    diff = a.astype(np.float64) - b.astype(np.float64)
    if roi is not None:
        r = roi > 127
        if r.sum() == 0:
            return float("nan")
        diff = diff[r]
    mse = np.mean(diff**2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def evaluate(pred_frames, orig_paths, gt_mask_paths):
    proxy_psnrs = []
    for i in range(min(len(pred_frames), len(orig_paths), len(gt_mask_paths))):
        pred = pred_frames[i]
        orig = cv2.imread(str(orig_paths[i]))
        if orig is None:
            continue
        if orig.shape[:2] != pred.shape[:2]:
            orig = cv2.resize(orig, (pred.shape[1], pred.shape[0]))
        gt_m = cv2.imread(str(gt_mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if gt_m is None:
            continue
        if gt_m.shape != pred.shape[:2]:
            gt_m = cv2.resize(gt_m, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        proxy_mask = 255 - (gt_m > 0).astype(np.uint8) * 255
        proxy_psnrs.append(compute_psnr(pred, orig, roi=proxy_mask))
    proxy_psnrs = [x for x in proxy_psnrs if not np.isnan(x)]
    return float(np.mean(proxy_psnrs)) if proxy_psnrs else float("nan")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  VOID koala inference")
    print(f"  GPU={args.gpu}, steps={args.steps}, guidance={args.guidance}")
    print("=" * 65)

    orig_paths = load_sorted_frames(KOALA_ORIG)
    mask_paths = sorted(KOALA_MASK.glob("*.png"), key=lambda p: p.stem)
    gt_paths   = sorted(KOALA_GT.glob("*.png"),   key=lambda p: p.stem)

    n = len(orig_paths)
    print(f"  {n} frames, processing in batches of {TEMPORAL_WIN}")

    # Load VOID pipeline
    pipeline = load_void_pipeline(args.gpu)

    prompt = "eucalyptus tree branches with leaves, natural forest background, no animals, outdoors"
    neg_prompt = "koala, bear, animal, fur, blurry, low quality, watermark"

    # Process in batches of TEMPORAL_WIN frames
    all_result_frames = []
    batch_start = 0
    while batch_start < n:
        batch_end = min(batch_start + TEMPORAL_WIN, n)
        batch_frames = orig_paths[batch_start:batch_end]
        batch_masks  = mask_paths[batch_start:batch_end]
        print(f"\n  Batch [{batch_start}:{batch_end}] ({len(batch_frames)} frames)")

        out_frames = run_void_inference(
            pipeline, batch_frames, batch_masks,
            prompt, neg_prompt, args.steps, args.guidance, args.seed + batch_start,
        )
        all_result_frames.extend(out_frames)
        batch_start = batch_end
        print(f"  Batch done: got {len(out_frames)} frames")

    print(f"\n  Total output frames: {len(all_result_frames)}")

    # Save output
    out_mp4 = OUTPUT_DIR / "inpaint_out.mp4"
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_result_frames]
    imageio.mimwrite(str(out_mp4), rgb_frames, fps=24, quality=8)
    print(f"  [saved] {out_mp4}")

    # Generate masked_in.mp4 preview
    import subprocess
    preview_mp4 = OUTPUT_DIR / "masked_in.mp4"
    if PREVIEW_SCRIPT.exists():
        r = subprocess.run([
            sys.executable, str(PREVIEW_SCRIPT),
            "--frames_dir", str(KOALA_ORIG),
            "--masks_dir", str(KOALA_MASK),
            "--output_mp4", str(preview_mp4),
        ], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print(f"  [saved] {preview_mp4}")
        else:
            print(f"  [WARN] preview failed: {r.stderr[-200:]}")

    # Evaluate vs ProPainter baseline
    print("\n  Evaluating VOID vs ProPainter baseline...")
    void_psnr_proxy = evaluate(all_result_frames, orig_paths, gt_paths)
    print(f"  VOID proxy PSNR: {void_psnr_proxy:.3f} dB")

    if PP_BASELINE.exists():
        cap = cv2.VideoCapture(str(PP_BASELINE))
        pp_frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            pp_frames.append(f)
        cap.release()

        # Resize PP frames to match original resolution
        ref = cv2.imread(str(orig_paths[0]))
        H_orig, W_orig = ref.shape[:2]
        pp_frames_rs = [cv2.resize(f, (W_orig, H_orig)) for f in pp_frames]

        pp_psnr_proxy = evaluate(pp_frames_rs, orig_paths, gt_paths)
        delta = void_psnr_proxy - pp_psnr_proxy
        print(f"  ProPainter proxy PSNR: {pp_psnr_proxy:.3f} dB")
        print(f"  Delta (VOID - PP): {delta:+.3f} dB")
    else:
        pp_psnr_proxy = float("nan")
        delta = float("nan")

    # Save results
    result = {
        "model": "VOID_pass1",
        "sequence": "koala",
        "steps": args.steps,
        "guidance": args.guidance,
        "PSNR_proxy_void": void_psnr_proxy,
        "PSNR_proxy_pp": pp_psnr_proxy,
        "delta_proxy": delta,
        "n_frames": len(all_result_frames),
        "output_dir": str(OUTPUT_DIR),
        "inpaint_out": str(out_mp4),
        "masked_in": str(preview_mp4),
    }
    result_json = OUTPUT_DIR / "void_eval.json"
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [saved] {result_json}")

    print("\n" + "=" * 65)
    print("  VOID inference complete")
    print(f"  inpaint_out.mp4: {out_mp4}")
    print(f"  masked_in.mp4:   {preview_mp4}")
    print("=" * 65)


if __name__ == "__main__":
    main()
