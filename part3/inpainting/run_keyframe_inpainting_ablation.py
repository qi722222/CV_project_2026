"""
run_keyframe_inpainting_ablation.py — Direction C: Keyframe Inpainting Model Ablation

对比两种关键帧修复模型:
  - SD1.5 Inpainting (runwayml/stable-diffusion-inpainting) — 专用 SD1.5 inpainting
  - SDXL Inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1) — 高容量通用
两者都与 ProPainter 结合，形成 "keyframe repair → ProPainter propagation" 管线。

消融设计:
  pure_propainter  — 纯 ProPainter（已有结果，作为参考基线）
  sd15_keyframe    — SD1.5 keyframe + ProPainter
  sdxl_keyframe    — SDXL keyframe + ProPainter（已有结果，但补跑以统一对比）

用法:
  conda run -n controlnet_env python3 part3/run_keyframe_inpainting_ablation.py \
    --model sd15 \
    --sequences tennis bmx-trees \
    --keyframe_interval 10 \
    --gpu 0

  conda run -n controlnet_env python3 part3/run_keyframe_inpainting_ablation.py \
    --model sdxl \
    --sequences tennis bmx-trees \
    --keyframe_interval 5 \
    --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
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

MODEL_CONFIGS = {
    "sd15": {
        "model_id": "runwayml/stable-diffusion-inpainting",
        "pipeline_class": "StableDiffusionInpaintPipeline",
        "default_guidance": 7.5,
        "default_steps": 20,
    },
    "sdxl": {
        "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "pipeline_class": "AutoPipelineForInpainting",
        "default_guidance": 3.0,
        "default_steps": 20,
    },
}

SEQUENCES_CFG = {
    "tennis": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
        "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/tennis",
        "pure_propainter_dir": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/tennis/propainter_pure/tennis",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/tennis",
        "inpaint_prompt": "tennis court background with no players, clean green surface",
    },
    "bmx-trees": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
        "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/bmx-trees",
        "pure_propainter_dir": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/bmx-trees/propainter_pure/bmx-trees",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/bmx-trees",
        "inpaint_prompt": "forest path with trees, no cyclist, natural background",
    },
    "wild_video-1person": {
        "orig_dir": "/home/jli657/my_storage2_1T/project3/wild_frames/wild_video-1person",
        "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_shadow_v1/wild_video-1person",
        "pure_propainter_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/propainter/wild_video-1person/wild_video-1person",
        "gt_mask_dir": "",  # no GT for wild video
        "inpaint_prompt": "outdoor street background, no people, clean sidewalk",
    },
    # koala: large-mask case (avg 26.3%, max 57.6%). ProPainter baseline PSNR poor
    # due to missing background behind large close-up subject.
    # SDXL expected to help by generating semantically coherent eucalyptus background.
    "koala": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/koala",
        "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/koala",
        "pure_propainter_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/propainter/koala/koala",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/koala",
        "inpaint_prompt": (
            "eucalyptus tree branches with leaves, natural forest background, "
            "no animals, empty branch, outdoors, high quality, natural lighting"
        ),
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sd15", "sdxl"], default="sd15")
    p.add_argument("--sequences", nargs="+", default=["tennis", "bmx-trees"])
    p.add_argument("--keyframe_interval", type=int, default=10)
    p.add_argument("--perframe", action="store_true",
                   help="Inpaint every frame with diffusion (no ProPainter propagation)")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="Override default guidance scale")
    p.add_argument("--strength", type=float, default=0.99)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output_root", default="/data3/jli657/project3/part3/outputs/keyframe_inpaint_ablation")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_pipe(model_key: str, gpu: int):
    cfg = MODEL_CONFIGS[model_key]
    device = f"cuda:{gpu}"
    if model_key == "sd15":
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16,
        )
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload(gpu_id=gpu)
    else:
        from diffusers import AutoPipelineForInpainting
        pipe = AutoPipelineForInpainting.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload(gpu_id=gpu)
    print(f"[model loaded] {cfg['model_id']} on cuda:{gpu}")
    return pipe


def select_keyframes(n: int, interval: int) -> List[int]:
    return list(range(0, n, interval))


def load_sorted_files(d: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def load_mask_bin(path: Path, H: int, W: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((H, W), dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((H, W), dtype=np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def inpaint_frame(pipe, model_key: str, orig_bgr: np.ndarray, mask_bin: np.ndarray,
                   prompt: str, guidance: float, steps: int, strength: float) -> np.ndarray:
    H, W = orig_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_bin)

    gen = torch.Generator(device="cpu").manual_seed(42)

    if model_key == "sd15":
        # SD1.5 requires 512x512 internally; resize then resize back
        target_size = 512
        pil_img_r = pil_img.resize((target_size, target_size), Image.LANCZOS)
        pil_mask_r = pil_mask.resize((target_size, target_size), Image.NEAREST)
        result = pipe(
            prompt=prompt,
            negative_prompt="person, human, distortion, blur, artifact",
            image=pil_img_r,
            mask_image=pil_mask_r,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        ).images[0]
        result = result.resize((W, H), Image.LANCZOS)
    else:
        result = pipe(
            prompt=prompt,
            negative_prompt="person, human, distortion, artifact",
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
    cmd = [PROPAINTER_PYTHON, "inference_propainter.py",
           "--video", frames_dir, "--mask", masks_dir, "--output", output_dir, "--fp16"]
    r = subprocess.run(cmd, cwd=PROPAINTER_DIR, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"  ProPainter FAILED: {r.stderr[-500:]}")
        return False
    return True


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
    return float(structural_similarity(img1, img2, multichannel=True, data_range=255, channel_axis=2))


def load_video(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def evaluate(pred_source: Path, orig_paths: List[Path], gt_mask_paths: List[Path]) -> dict:
    if pred_source.is_file() and pred_source.suffix == ".mp4":
        pred_list = load_video(pred_source)
    elif pred_source.is_dir():
        mp4s = list(pred_source.glob("*.mp4"))
        if mp4s:
            pred_list = load_video(mp4s[0])
        else:
            pred_list = [cv2.imread(str(p)) for p in sorted(pred_source.glob("*.png"), key=lambda p: p.stem)]
    else:
        return {}

    has_gt = len(gt_mask_paths) > 0
    n = min(len(pred_list), len(orig_paths), len(gt_mask_paths) if has_gt else len(pred_list))
    proxy_list, synth_list, ssim_list = [], [], []

    for i in range(n):
        pred = pred_list[i]
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
            mask_bin = (gt_m > 0).astype(np.uint8) * 255
            proxy_mask = 255 - mask_bin
            psnr_p = compute_psnr(pred, orig, roi_mask=proxy_mask)
            if not np.isnan(psnr_p):
                proxy_list.append(psnr_p)
            synth = cv2.inpaint(orig, (mask_bin > 127).astype(np.uint8), 3, cv2.INPAINT_TELEA)
            psnr_s = compute_psnr(pred, synth)
            if not np.isnan(psnr_s):
                synth_list.append(psnr_s)
            try:
                ssim_list.append(compute_ssim(pred, synth))
            except Exception:
                pass

    return {
        "num_frames": n,
        "PSNR_proxy": float(np.mean(proxy_list)) if proxy_list else float("nan"),
        "PSNR_synthetic": float(np.mean(synth_list)) if synth_list else float("nan"),
        "SSIM_synthetic": float(np.mean(ssim_list)) if ssim_list else float("nan"),
    }


def process_sequence_perframe(seq_name: str, cfg: dict, args, pipe, model_key: str, model_cfg: dict) -> dict:
    """Inpaint every frame independently with diffusion — no ProPainter propagation."""
    guidance = args.guidance_scale or model_cfg["default_guidance"]
    steps = args.steps or model_cfg["default_steps"]
    prompt = cfg["inpaint_prompt"]

    print(f"\n{'='*60}")
    print(f"[{seq_name}] {model_key} PERFRAME (every frame inpainted, no ProPainter)")
    print(f"  prompt: {prompt[:80]}")
    print(f"  guidance={guidance}, steps={steps}, strength={args.strength}")
    print('='*60)

    orig_dir = Path(cfg["orig_dir"])
    mask_dir = Path(cfg["mask_dir"])
    out_root = Path(args.output_root) / f"{model_key}_perframe" / seq_name
    out_root.mkdir(parents=True, exist_ok=True)
    final_dir = out_root / "final_output"
    final_dir.mkdir(exist_ok=True)

    orig_frames = load_sorted_files(orig_dir)
    mask_files = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)
    n_frames = len(orig_frames)
    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]
    print(f"  {n_frames} frames, {W}x{H}")

    result_frames = []
    for i, fp in enumerate(orig_frames):
        frame = cv2.imread(str(fp))
        mf = mask_files[i] if i < len(mask_files) else None
        mask = load_mask_bin(mf, H, W)
        coverage = (mask > 127).mean() * 100

        if coverage < 1.0:
            result_frames.append(frame)
            continue

        repaired = inpaint_frame(pipe, model_key, frame, mask, prompt, guidance, steps, args.strength)
        result_frames.append(repaired)
        if i % 10 == 0:
            print(f"  frame {i:3d}/{n_frames}: mask={coverage:.1f}%")

    # Save video
    import imageio
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in result_frames]
    out_mp4 = final_dir / "inpaint_out.mp4"
    imageio.mimwrite(str(out_mp4), rgb_frames, fps=24, quality=8)
    print(f"  [saved] {out_mp4}")

    # Generate masked_in.mp4 preview
    import subprocess as sp
    gen_preview = Path("/home/jli657/my_storage2_1T/project3/part3/generate_mask_preview.py")
    if gen_preview.exists():
        r = sp.run([
            "python3", str(gen_preview),
            "--frames_dir", str(orig_dir),
            "--masks_dir", str(mask_dir),
            "--output_mp4", str(final_dir / "masked_in.mp4"),
        ], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"  [WARN] preview: {r.stderr[-100:]}")

    # Evaluate vs baseline
    gt_mask_dir = Path(cfg["gt_mask_dir"]) if cfg["gt_mask_dir"] else Path("")
    gt_masks = sorted(gt_mask_dir.glob("*.png"), key=lambda p: p.stem) if gt_mask_dir.exists() else []
    metrics = evaluate(out_mp4, orig_frames, gt_masks)

    pure_dir = Path(cfg.get("pure_propainter_dir", ""))
    if pure_dir.exists():
        pure_metrics = evaluate(pure_dir, orig_frames, gt_masks)
        metrics["pure_PSNR_proxy"] = pure_metrics.get("PSNR_proxy", float("nan"))
        metrics["delta_proxy"] = metrics.get("PSNR_proxy", 0) - pure_metrics.get("PSNR_proxy", 0)
        print(f"\n  Results ({model_key} perframe vs pure_propainter):")
        print(f"    perframe: proxy={metrics.get('PSNR_proxy',0):.3f}")
        print(f"    pure_pp:  proxy={pure_metrics.get('PSNR_proxy',0):.3f}")
        print(f"    delta:    {metrics['delta_proxy']:+.3f}")

    metrics.update({
        "sequence": seq_name, "model": f"{model_key}_perframe",
        "output_dir": str(out_root), "pp_video": str(out_mp4),
    })
    return metrics


def process_sequence(seq_name: str, cfg: dict, args, pipe, model_key: str, model_cfg: dict) -> dict:
    guidance = args.guidance_scale or model_cfg["default_guidance"]
    steps = args.steps or model_cfg["default_steps"]
    prompt = cfg["inpaint_prompt"]

    print(f"\n{'='*60}")
    print(f"[{seq_name}] {model_key} keyframe (interval={args.keyframe_interval}) + ProPainter")
    print(f"  prompt: {prompt}")
    print(f"  guidance={guidance}, steps={steps}, strength={args.strength}")
    print('='*60)

    orig_dir = Path(cfg["orig_dir"])
    mask_dir = Path(cfg["mask_dir"])
    out_root = Path(args.output_root) / model_key / seq_name / f"interval{args.keyframe_interval}"
    out_root.mkdir(parents=True, exist_ok=True)

    orig_frames = load_sorted_files(orig_dir)
    n_frames = len(orig_frames)
    mask_files = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)

    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]
    print(f"  {n_frames} frames, {W}x{H}")

    keyframes = select_keyframes(n_frames, args.keyframe_interval)
    print(f"  {len(keyframes)} keyframes")

    # Step 1: inpaint keyframes
    sdxl_dir = out_root / "inpainted_keyframes"
    sdxl_dir.mkdir(exist_ok=True)
    inpaint_results = {}

    for idx in keyframes:
        frame = cv2.imread(str(orig_frames[idx]))
        mf = mask_files[idx] if idx < len(mask_files) else None
        mask = load_mask_bin(mf, H, W)

        if mask.sum() == 0:
            inpaint_results[idx] = frame
            continue

        repaired = inpaint_frame(pipe, model_key, frame, mask, prompt, guidance, steps, args.strength)
        inpaint_results[idx] = repaired
        out_kf = sdxl_dir / f"{orig_frames[idx].stem}.png"
        cv2.imwrite(str(out_kf), repaired)
        print(f"  frame {idx}: inpainted (mask {mask.mean()/255*100:.1f}%)")

    # Step 2: prepare ProPainter input
    pp_frames = out_root / "pp_input" / "frames"
    pp_masks = out_root / "pp_input" / "masks"
    pp_frames.mkdir(parents=True, exist_ok=True)
    pp_masks.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(orig_frames):
        stem = fp.stem
        if i in inpaint_results:
            cv2.imwrite(str(pp_frames / f"{stem}.png"), inpaint_results[i])
            cv2.imwrite(str(pp_masks / f"{stem}.png"), np.zeros((H, W), dtype=np.uint8))
        else:
            shutil.copy2(fp, pp_frames / f"{stem}.png")
            mf = mask_files[i] if i < len(mask_files) else None
            mask = load_mask_bin(mf, H, W)
            cv2.imwrite(str(pp_masks / f"{stem}.png"), mask)

    # Step 3: Run ProPainter
    pp_out_root = str(out_root / "propainter_output")
    Path(pp_out_root).mkdir(exist_ok=True)
    success = run_propainter(str(pp_frames), str(pp_masks), pp_out_root)
    if not success:
        return {"sequence": seq_name, "model": model_key, "error": "ProPainter failed"}

    # Find output video
    pp_video = None
    for candidate in [
        Path(pp_out_root) / "inpaint_out.mp4",
        Path(pp_out_root) / seq_name / "inpaint_out.mp4",
    ]:
        if candidate.exists():
            pp_video = candidate
            break
    if pp_video is None:
        mp4s = list(Path(pp_out_root).rglob("inpaint_out.mp4"))
        pp_video = mp4s[0] if mp4s else Path(pp_out_root)

    print(f"  ProPainter output: {pp_video}")

    # Step 4: Evaluate
    gt_mask_dir = Path(cfg["gt_mask_dir"]) if cfg["gt_mask_dir"] else Path("")
    gt_masks = sorted(gt_mask_dir.glob("*.png"), key=lambda p: p.stem) if gt_mask_dir.exists() else []
    metrics = evaluate(pp_video, orig_frames, gt_masks)

    # Compare vs pure ProPainter
    pure_dir = Path(cfg.get("pure_propainter_dir", ""))
    if pure_dir.exists():
        pure_metrics = evaluate(pure_dir, orig_frames, gt_masks)
        metrics["pure_PSNR_proxy"] = pure_metrics.get("PSNR_proxy", float("nan"))
        metrics["pure_PSNR_synthetic"] = pure_metrics.get("PSNR_synthetic", float("nan"))
        metrics["delta_proxy"] = metrics.get("PSNR_proxy", 0) - pure_metrics.get("PSNR_proxy", 0)
        metrics["delta_synthetic"] = metrics.get("PSNR_synthetic", 0) - pure_metrics.get("PSNR_synthetic", 0)
        print(f"\n  Results ({model_key} keyframe vs pure_propainter):")
        print(f"    {model_key}: proxy={metrics.get('PSNR_proxy',0):.3f}  synth={metrics.get('PSNR_synthetic',0):.3f}")
        print(f"    pure:   proxy={pure_metrics.get('PSNR_proxy',0):.3f}  synth={pure_metrics.get('PSNR_synthetic',0):.3f}")
        print(f"    delta:  proxy={metrics['delta_proxy']:+.3f}  synth={metrics['delta_synthetic']:+.3f}")

    # Generate masked_in.mp4 preview alongside inpaint_out.mp4
    final_dir = Path(str(pp_video)).parent
    gen_preview = Path("/home/jli657/my_storage2_1T/project3/part3/generate_mask_preview.py")
    if gen_preview.exists():
        import subprocess as sp
        r = sp.run([
            "python3", str(gen_preview),
            "--frames_dir", str(orig_dir),
            "--masks_dir", str(mask_dir),
            "--output_mp4", str(final_dir / "masked_in.mp4"),
        ], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print(f"  [preview] masked_in.mp4 saved to {final_dir}")
        else:
            print(f"  [WARN] preview failed: {r.stderr[-100:]}")

    metrics.update({"sequence": seq_name, "model": model_key,
                    "keyframe_interval": args.keyframe_interval,
                    "output_dir": str(out_root),
                    "pp_video": str(pp_video),
                    "masked_in": str(final_dir / "masked_in.mp4")})
    return metrics


def main():
    args = parse_args()
    model_cfg = MODEL_CONFIGS[args.model]
    guidance = args.guidance_scale or model_cfg["default_guidance"]
    steps = args.steps or model_cfg["default_steps"]
    print(f"Loading {args.model} inpainting model: {model_cfg['model_id']}")
    pipe = load_pipe(args.model, args.gpu)

    all_results = []
    for seq in args.sequences:
        cfg = SEQUENCES_CFG.get(seq)
        if not cfg:
            print(f"[skip] {seq}: not in config")
            continue
        try:
            if args.perframe:
                res = process_sequence_perframe(seq, cfg, args, pipe, args.model, model_cfg)
            else:
                res = process_sequence(seq, cfg, args, pipe, args.model, model_cfg)
            all_results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({"sequence": seq, "model": args.model, "error": str(e)})

    out_json = Path(args.output_root) / f"ablation_{args.model}_interval{args.keyframe_interval}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {out_json}")

    print(f"\n{'='*60}\nFINAL SUMMARY ({args.model}, interval={args.keyframe_interval})")
    for r in all_results:
        if "error" in r:
            print(f"  {r['sequence']}: ERROR - {r['error']}")
        else:
            print(f"  {r['sequence']}: proxy={r.get('PSNR_proxy',0):.3f}  synth={r.get('PSNR_synthetic',0):.3f}  "
                  f"delta_proxy={r.get('delta_proxy',0):+.3f}  delta_synth={r.get('delta_synthetic',0):+.3f}")


if __name__ == "__main__":
    main()
