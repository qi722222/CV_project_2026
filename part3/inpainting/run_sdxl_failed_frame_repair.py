from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting

# controlnet_env on this machine has cuDNN init issues; disable cuDNN for stability.
torch.backends.cudnn.enabled = False
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDXL failed-frame repair over ProPainter video")
    p.add_argument("--orig_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--propainter_video", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt", default="clean natural background")
    p.add_argument("--negative_prompt", default="person, human, object, artifact, blur, watermark")
    p.add_argument("--model_id", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    p.add_argument("--max_repair_frames", type=int, default=8)
    p.add_argument("--mask_alpha", type=float, default=0.85)
    p.add_argument("--strength", type=float, default=0.95)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def sorted_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def ensure_video_frames(video_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.png")):
        return sorted_images(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), frame)
        idx += 1
    cap.release()
    return sorted_images(out_dir)


def read_mask(mask_path: Path, hw: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(hw, np.uint8)
    if m.shape != hw:
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def select_failed_frames(prop_frames: List[Path], mask_dir: Path, topk: int) -> List[int]:
    # Heuristic: high temporal flicker inside masked region => likely ProPainter failure.
    scores = []
    prev = None
    for i, p in enumerate(prop_frames):
        cur = cv2.imread(str(p))
        if cur is None:
            continue
        h, w = cur.shape[:2]
        m = read_mask(mask_dir / f"{p.stem}.png", (h, w)).astype(bool)
        if prev is None or m.sum() == 0:
            prev = cur
            continue
        diff = np.abs(cur.astype(np.float32) - prev.astype(np.float32)).mean(axis=2)
        score = float(diff[m].mean()) if m.any() else 0.0
        area = float(m.mean())
        # combine flicker and area, prefer meaningful masked regions
        scores.append((i, score * (1.0 + 2.0 * area)))
        prev = cur
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:topk]]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = Path(args.orig_dir)
    mask_dir = Path(args.mask_dir)
    prop_video = Path(args.propainter_video)

    orig_files = sorted_images(orig_dir)
    prop_frames_dir = out_dir / "propainter_frames"
    repaired_frames_dir = out_dir / "repaired_frames"
    repaired_frames_dir.mkdir(parents=True, exist_ok=True)

    prop_files = ensure_video_frames(prop_video, prop_frames_dir)
    if not orig_files or not prop_files:
        raise RuntimeError("missing input frames")

    n = min(len(orig_files), len(prop_files))
    orig_files = orig_files[:n]
    prop_files = prop_files[:n]

    failed_indices = select_failed_frames(prop_files, mask_dir, args.max_repair_frames)

    dtype = torch.float16
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    # Lower peak VRAM to avoid cuDNN init/OOM issues on long runs.
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    gen = torch.Generator("cuda").manual_seed(args.seed)
    repaired_set = set(failed_indices)

    for i, (orig_p, prop_p) in enumerate(zip(orig_files, prop_files)):
        orig = cv2.imread(str(orig_p))
        prop = cv2.imread(str(prop_p))
        if orig is None or prop is None:
            continue
        h, w = prop.shape[:2]
        if orig.shape[:2] != (h, w):
            orig = cv2.resize(orig, (w, h), interpolation=cv2.INTER_LINEAR)

        m = read_mask(mask_dir / f"{orig_p.stem}.png", (h, w))
        out = prop.copy()

        if i in repaired_set and m.sum() > 0:
            rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_mask = Image.fromarray((m * 255).astype(np.uint8))

            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=pil_img,
                mask_image=pil_mask,
                strength=args.strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=gen,
            ).images[0]
            torch.cuda.empty_cache()
            sdxl = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            if sdxl.shape[:2] != (h, w):
                sdxl = cv2.resize(sdxl, (w, h), interpolation=cv2.INTER_LINEAR)

            alpha = (m.astype(np.float32) * args.mask_alpha)[:, :, None]
            out = (alpha * sdxl.astype(np.float32) + (1.0 - alpha) * prop.astype(np.float32)).astype(np.uint8)

        cv2.imwrite(str(repaired_frames_dir / f"{i:05d}.png"), out)

    out_video = out_dir / "sdxl_repair_out.mp4"
    first = cv2.imread(str(repaired_frames_dir / "00000.png"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, 24.0, (first.shape[1], first.shape[0]))
    for i in range(n):
        f = cv2.imread(str(repaired_frames_dir / f"{i:05d}.png"))
        writer.write(f)
    writer.release()

    meta = {
        "num_frames": n,
        "failed_indices": failed_indices,
        "max_repair_frames": args.max_repair_frames,
        "prompt": args.prompt,
        "output_video": str(out_video),
    }
    with (out_dir / "repair_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[ok] repaired video: {out_video}")
    print(f"[ok] repaired frames: {repaired_frames_dir}")
    print(f"[ok] failed frame indices: {failed_indices}")


if __name__ == "__main__":
    main()
