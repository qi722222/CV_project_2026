"""
run_keyframe_sdxl_propainter.py — Task 8: SDXL 关键帧 + ProPainter 传播

正确实现"修关键帧再传播"管线（对应课程文档要求）：
1. 选择关键帧（均匀间隔）
2. SDXL-Inpainting 对关键帧进行 mask 区域修复
3. 关键帧替换 + 对应 mask 设为全黑（0）→ ProPainter 以此为锚点传播
4. 与纯 ProPainter 对比 PSNR/SSIM

关键技术洞察:
  ProPainter 的 `masked_frames = frames * (1 - masks_dilated)`:
  - 如果某帧 mask = 0，该帧数据被完整保留作为传播锚点
  - 无需修改 ProPainter 代码，只需将关键帧 mask 文件替换为全黑图

运行环境: controlnet_env (有 diffusers/SDXL) + propainter_env (ProPainter)
脚本在 controlnet_env 中运行，ProPainter 步骤通过 subprocess 调用 propainter_env
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("HF_HOME", "/data3/jli657/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data3/jli657/hf_cache")
# Disable cuDNN for stability (same as in run_sdxl_failed_frame_repair.py)
import torch
torch.backends.cudnn.enabled = False

SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
PROPAINTER_DIR = "/data2/jli657/ProPainter"
PROPAINTER_PYTHON = "/data2/jli657/envs/propainter_env/bin/python"

SEQUENCES_CFG = {
    "tennis": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
        "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/tennis",
        "pure_propainter_dir": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/tennis/propainter_pure/tennis",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/tennis",
    },
    "bmx-trees": {
        "orig_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
        "mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/bmx-trees",
        "pure_propainter_dir": "/data3/jli657/project3/part3/outputs/controlnet/ablation_5seq/bmx-trees/propainter_pure/bmx-trees",
        "gt_mask_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/bmx-trees",
    },
}

EVAL_SCRIPT = "/home/jli657/my_storage2_1T/project3/part3/eval_controlnet_video_metrics.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+", default=["tennis", "bmx-trees"])
    p.add_argument("--keyframe_interval", type=int, default=10,
                   help="Interval between keyframes (e.g. 10 = every 10th frame)")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="Low guidance_scale reduces style drift")
    p.add_argument("--strength", type=float, default=0.99,
                   help="Inpainting strength (1.0 = full replacement)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output_root", default="/data3/jli657/project3/part3/outputs/keyframe_sdxl")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--prompt", default="",
                   help="Optional text prompt for SDXL inpainting (empty = background generation)")
    return p.parse_args()


def select_keyframes(n_frames: int, interval: int) -> List[int]:
    """Select keyframe indices at uniform intervals. Always include frame 0."""
    indices = list(range(0, n_frames, interval))
    return indices


def load_sorted_frames(frame_dir: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    return sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in exts],
                  key=lambda p: p.stem)


def load_mask_binary(mask_path: Path, target_h: int, target_w: int) -> np.ndarray:
    """Load mask as binary uint8 (0 or 255)."""
    if not mask_path.exists():
        return np.zeros((target_h, target_w), dtype=np.uint8)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    if m.shape[0] != target_h or m.shape[1] != target_w:
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def sdxl_inpaint_frame(pipe, orig_bgr: np.ndarray, mask_binary: np.ndarray,
                        prompt: str, strength: float, steps: int,
                        guidance_scale: float, seed: int = 42) -> np.ndarray:
    """Run SDXL inpainting on a single frame. Returns BGR uint8."""
    h, w = orig_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_binary)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(seed)

    result = pipe(
        prompt=prompt if prompt else "background scenery, no people, realistic",
        negative_prompt="person, human, distortion, artifact",
        image=pil_img,
        mask_image=pil_mask,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    if result_bgr.shape[:2] != (h, w):
        result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return result_bgr


def run_propainter(frames_dir: str, masks_dir: str, output_dir: str) -> bool:
    """Run ProPainter via subprocess in propainter_env."""
    cmd = [
        PROPAINTER_PYTHON,
        "inference_propainter.py",
        "--video", frames_dir,
        "--mask", masks_dir,
        "--output", output_dir,
        "--fp16",
    ]
    print(f"  Running ProPainter: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, cwd=PROPAINTER_DIR, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ProPainter failed: {result.stderr[-500:]}")
        return False
    print(f"  ProPainter done.")
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


def compute_ssim_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(img1, img2, multichannel=True,
                                       data_range=255, channel_axis=2))


def load_video_to_bgr_list(video_path: Path) -> List[np.ndarray]:
    """Load all frames from video as BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def evaluate_video_quality(pred_source: Path, orig_frames: List[Path],
                            gt_masks: List[Path]) -> dict:
    """Compute PSNR_proxy and PSNR_synthetic. pred_source can be dir of PNGs or .mp4 video."""
    # Load predicted frames
    if pred_source.is_file() and pred_source.suffix == ".mp4":
        pred_frames_list = load_video_to_bgr_list(pred_source)
    elif pred_source.is_dir():
        # Try to find frames or video
        mp4_files = list(pred_source.glob("*.mp4"))
        if mp4_files:
            pred_frames_list = load_video_to_bgr_list(mp4_files[0])
        else:
            pred_dir_frames = sorted(pred_source.glob("*.png"), key=lambda p: p.stem)
            pred_frames_list = [cv2.imread(str(p)) for p in pred_dir_frames]
    else:
        return {"num_frames": 0, "PSNR_proxy": float("nan"), "PSNR_synthetic": float("nan"), "SSIM_synthetic": float("nan")}

    n = min(len(pred_frames_list), len(orig_frames), len(gt_masks))

    psnr_proxy_list, psnr_synth_list, ssim_synth_list = [], [], []

    for i in range(n):
        pred = pred_frames_list[i]
        orig = cv2.imread(str(orig_frames[i]))
        gt_m = cv2.imread(str(gt_masks[i]), cv2.IMREAD_GRAYSCALE)

        if pred is None or orig is None or gt_m is None:
            continue

        h, w = pred.shape[:2]
        if orig.shape[:2] != (h, w):
            orig = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)
        if gt_m.shape[:2] != (h, w):
            gt_m = cv2.resize(gt_m, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_bin = (gt_m > 0).astype(np.uint8) * 255
        proxy_mask = (255 - mask_bin)

        psnr_p = compute_psnr(pred, orig, roi_mask=proxy_mask)
        if not np.isnan(psnr_p):
            psnr_proxy_list.append(psnr_p)

        synth = cv2.inpaint(orig, (mask_bin > 127).astype(np.uint8), inpaintRadius=3,
                            flags=cv2.INPAINT_TELEA)
        psnr_s = compute_psnr(pred, synth)
        if not np.isnan(psnr_s):
            psnr_synth_list.append(psnr_s)

        try:
            ssim_s = compute_ssim_simple(pred, synth)
            ssim_synth_list.append(ssim_s)
        except Exception:
            pass

    return {
        "num_frames": n,
        "PSNR_proxy": float(np.mean(psnr_proxy_list)) if psnr_proxy_list else float("nan"),
        "PSNR_synthetic": float(np.mean(psnr_synth_list)) if psnr_synth_list else float("nan"),
        "SSIM_synthetic": float(np.mean(ssim_synth_list)) if ssim_synth_list else float("nan"),
    }


def process_sequence(seq_name: str, cfg: dict, args, pipe) -> dict:
    print(f"\n{'='*60}\n[{seq_name}] SDXL Keyframe + ProPainter\n{'='*60}")

    orig_dir = Path(cfg["orig_dir"])
    mask_dir = Path(cfg["mask_dir"])
    out_root = Path(args.output_root) / seq_name / f"interval{args.keyframe_interval}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load frames
    orig_frames = load_sorted_frames(orig_dir)
    n_frames = len(orig_frames)
    print(f"  Loaded {n_frames} original frames")

    # Load masks
    mask_files = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)
    if not mask_files:
        print(f"  WARN: No masks in {mask_dir}, using zero masks")
        mask_files = [None] * n_frames

    # Reference frame size
    ref = cv2.imread(str(orig_frames[0]))
    H, W = ref.shape[:2]
    print(f"  Frame size: {W}x{H}")

    # Select keyframes
    keyframe_indices = select_keyframes(n_frames, args.keyframe_interval)
    print(f"  Keyframes ({len(keyframe_indices)} total): {keyframe_indices[:10]}{'...' if len(keyframe_indices)>10 else ''}")

    # ---- Step 1: SDXL repair on keyframes ----
    sdxl_frames_dir = out_root / "sdxl_keyframes"
    sdxl_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"  SDXL inpainting {len(keyframe_indices)} keyframes...")
    sdxl_results = {}
    for idx in keyframe_indices:
        orig_frame = cv2.imread(str(orig_frames[idx]))
        if mask_files[idx] is not None:
            mask = load_mask_binary(mask_files[idx], H, W)
        else:
            mask = np.zeros((H, W), dtype=np.uint8)

        if mask.sum() == 0:
            # No mask: keep original
            sdxl_results[idx] = orig_frame
            print(f"    Frame {idx}: no mask, keeping original")
            continue

        repaired = sdxl_inpaint_frame(
            pipe, orig_frame, mask, args.prompt,
            args.strength, args.steps, args.guidance_scale, seed=42
        )
        sdxl_results[idx] = repaired
        print(f"    Frame {idx}: SDXL repaired (mask area: {mask.mean()/255*100:.1f}%)")

    # ---- Step 2: Prepare ProPainter input ----
    # frames/: keyframes = SDXL output, others = original
    # masks/: keyframes = all-zero mask, others = SAM3 mask
    pp_frames_dir = out_root / "propainter_input" / "frames"
    pp_masks_dir = out_root / "propainter_input" / "masks"
    pp_frames_dir.mkdir(parents=True, exist_ok=True)
    pp_masks_dir.mkdir(parents=True, exist_ok=True)

    for i, orig_path in enumerate(orig_frames):
        stem = orig_path.stem
        frame_out = pp_frames_dir / f"{stem}.png"
        mask_out = pp_masks_dir / f"{stem}.png"

        if i in sdxl_results:
            # Keyframe: use SDXL output, zero mask (ProPainter treats as known)
            cv2.imwrite(str(frame_out), sdxl_results[i])
            cv2.imwrite(str(mask_out), np.zeros((H, W), dtype=np.uint8))
        else:
            # Non-keyframe: use original, keep SAM3 mask
            shutil.copy2(orig_path, frame_out)
            if mask_files[i] is not None:
                mask = load_mask_binary(mask_files[i], H, W)
                cv2.imwrite(str(mask_out), mask)
            else:
                cv2.imwrite(str(mask_out), np.zeros((H, W), dtype=np.uint8))

    print(f"  Prepared ProPainter input: {n_frames} frames, {len(keyframe_indices)} with zero mask")

    # ---- Step 3: Run ProPainter ----
    pp_output_dir = str(out_root / "propainter_output")
    Path(pp_output_dir).mkdir(parents=True, exist_ok=True)
    success = run_propainter(str(pp_frames_dir), str(pp_masks_dir), pp_output_dir)

    if not success:
        return {"error": "ProPainter failed", "sequence": seq_name}

    # Output frames are in pp_output_dir/frames/
    pp_out_frames_dir = Path(pp_output_dir) / "frames"
    if not pp_out_frames_dir.exists():
        # ProPainter might save directly in output_dir
        pp_out_frames_dir = Path(pp_output_dir)

    # ---- Step 4: Evaluate ----
    gt_masks_list = sorted(Path(cfg["gt_mask_dir"]).glob("*.png"), key=lambda p: p.stem)
    # ProPainter outputs to a subdirectory named after the sequence inside output_dir
    # Try finding the actual output location
    pp_video = None
    for candidate in [
        Path(pp_output_dir) / "inpaint_out.mp4",
        Path(pp_output_dir) / seq_name / "inpaint_out.mp4",
        Path(pp_output_dir) / "frames",
    ]:
        if candidate.exists():
            pp_video = candidate
            break
    if pp_video is None:
        # Scan for any mp4
        mp4s = list(Path(pp_output_dir).rglob("inpaint_out.mp4"))
        pp_video = mp4s[0] if mp4s else Path(pp_output_dir)

    metrics = evaluate_video_quality(pp_video, orig_frames, gt_masks_list)

    # Also compute pure_propainter reference
    pure_pp_dir = Path(cfg.get("pure_propainter_dir", ""))
    if pure_pp_dir.exists():
        pure_metrics = evaluate_video_quality(pure_pp_dir, orig_frames, gt_masks_list)
        metrics["pure_propainter_PSNR_proxy"] = pure_metrics["PSNR_proxy"]
        metrics["pure_propainter_PSNR_synthetic"] = pure_metrics["PSNR_synthetic"]
        metrics["delta_PSNR_proxy"] = metrics["PSNR_proxy"] - pure_metrics["PSNR_proxy"]
        metrics["delta_PSNR_synthetic"] = metrics["PSNR_synthetic"] - pure_metrics["PSNR_synthetic"]
        print(f"\n  Results:")
        print(f"    keyframe_sdxl: PSNR_proxy={metrics['PSNR_proxy']:.3f}  PSNR_synth={metrics['PSNR_synthetic']:.3f}")
        print(f"    pure_propainter: PSNR_proxy={pure_metrics['PSNR_proxy']:.3f}  PSNR_synth={pure_metrics['PSNR_synthetic']:.3f}")
        print(f"    delta: proxy={metrics['delta_PSNR_proxy']:+.3f}  synth={metrics['delta_PSNR_synthetic']:+.3f}")
    else:
        print(f"  keyframe_sdxl: PSNR_proxy={metrics['PSNR_proxy']:.3f}  PSNR_synth={metrics['PSNR_synthetic']:.3f}")

    metrics["sequence"] = seq_name
    metrics["keyframe_interval"] = args.keyframe_interval
    metrics["guidance_scale"] = args.guidance_scale
    metrics["output_dir"] = str(out_root)

    return metrics


def main():
    args = parse_args()

    # Load SDXL pipeline
    print(f"Loading SDXL Inpainting: {SDXL_MODEL_ID}")
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    print("SDXL loaded.")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seq in args.sequences:
        cfg = SEQUENCES_CFG.get(seq)
        if cfg is None:
            print(f"[skip] {seq}: no config")
            continue
        try:
            res = process_sequence(seq, cfg, args, pipe)
            all_results.append(res)
        except Exception as e:
            import traceback
            print(f"[ERROR] {seq}: {e}")
            traceback.print_exc()
            all_results.append({"sequence": seq, "error": str(e)})

    # Save results
    out_json = out_root / f"direction_c_keyframe_results_interval{args.keyframe_interval}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] {out_json}")

    # Also save to final_delivery
    import csv
    out_csv = "/home/jli657/my_storage2_1T/project3/report_assets/final_delivery/direction_c_keyframe_metrics.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "keyframe_interval", "guidance_scale",
              "PSNR_proxy", "PSNR_synthetic", "SSIM_synthetic",
              "pure_propainter_PSNR_proxy", "pure_propainter_PSNR_synthetic",
              "delta_PSNR_proxy", "delta_PSNR_synthetic"]
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if Path(out_csv).stat().st_size == 0:
            w.writeheader()
        for r in all_results:
            if "error" not in r:
                w.writerow({k: r.get(k, "") for k in fields})
    print(f"[append] {out_csv}")

    # Summary
    print(f"\n{'='*60}\nFINAL SUMMARY (interval={args.keyframe_interval})\n{'='*60}")
    for r in all_results:
        if "error" in r:
            print(f"  {r.get('sequence','?')}: ERROR - {r['error']}")
        else:
            delta_p = r.get("delta_PSNR_proxy", float("nan"))
            delta_s = r.get("delta_PSNR_synthetic", float("nan"))
            print(f"  {r['sequence']}: PSNR_proxy={r.get('PSNR_proxy',0):.3f} "
                  f"PSNR_synth={r.get('PSNR_synthetic',0):.3f} "
                  f"delta_proxy={delta_p:+.3f} delta_synth={delta_s:+.3f}")


if __name__ == "__main__":
    main()
