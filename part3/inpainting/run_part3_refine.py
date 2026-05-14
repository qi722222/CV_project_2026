"""
run_part3_refine.py
-------------------
Part3 skeleton:  +

:
- copy_fallback: diffusersrefined

:
- sd_controlnet:  SD Inpainting + ControlNet
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class KeyframeStats:
    frame_name: str
    frame_index: int
    mask_area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part3 keyframe refine skeleton")
    parser.add_argument(
        "--config",
        default="part3/configs/default_part3.yaml",
        help="YAML",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f": {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("dict")
    return data


def sorted_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def extract_video_to_frames(video_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f": {video_path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"{idx:05d}.png"
        cv2.imwrite(str(out_path), frame)
        idx += 1
    cap.release()
    return sorted_images(out_dir)


def materialize_input_frames(input_dir: Path, output_dir: Path) -> Tuple[List[Path], Path]:
    frame_files = sorted_images(input_dir)
    if frame_files:
        return frame_files, input_dir

    # inpaint_out.mp4
    cand_videos = sorted(input_dir.glob("*.mp4"))
    preferred = [p for p in cand_videos if p.name == "inpaint_out.mp4"]
    video = preferred[0] if preferred else (cand_videos[0] if cand_videos else None)
    if video is None:
        return [], input_dir

    extracted_dir = output_dir / "_tmp_extracted_frames"
    frames = extract_video_to_frames(video, extracted_dir)
    return frames, extracted_dir


def mask_area_ratio(mask_path: Path) -> float:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0.0
    return float((m > 0).sum()) / float(m.shape[0] * m.shape[1])


def pick_uniform_indices(candidates: Sequence[KeyframeStats], k: int) -> List[KeyframeStats]:
    if len(candidates) <= k:
        return list(candidates)
    idxs = np.linspace(0, len(candidates) - 1, k, dtype=int).tolist()
    return [candidates[i] for i in idxs]


def select_keyframes(
    frame_files: List[Path],
    masks_dir: Path,
    policy: Dict,
) -> List[KeyframeStats]:
    threshold = float(policy.get("mask_area_ratio_threshold", 0.15))
    max_k = int(policy.get("max_keyframes_per_sequence", 3))
    global_k = int(policy.get("global_max_keyframes", 5))
    force_ids = policy.get("force_frame_indices", []) or []
    k = max(1, min(max_k, global_k))

    frame_by_idx = {i: p for i, p in enumerate(frame_files)}
    if force_ids:
        forced: List[KeyframeStats] = []
        for i in force_ids[:k]:
            if i in frame_by_idx:
                frame_path = frame_by_idx[i]
                stem = frame_path.stem
                mask_path = masks_dir / f"{stem}.png"
                forced.append(
                    KeyframeStats(
                        frame_name=frame_path.name,
                        frame_index=i,
                        mask_area_ratio=mask_area_ratio(mask_path) if mask_path.exists() else 0.0,
                    )
                )
        return forced

    candidates: List[KeyframeStats] = []
    for i, frame_path in enumerate(frame_files):
        stem = frame_path.stem
        mask_path = masks_dir / f"{stem}.png"
        if not mask_path.exists():
            continue
        ratio = mask_area_ratio(mask_path)
        if ratio >= threshold:
            candidates.append(KeyframeStats(frame_name=frame_path.name, frame_index=i, mask_area_ratio=ratio))

    if not candidates:
        fallback_ids = sorted(set([0, len(frame_files) // 2, max(0, len(frame_files) - 1)]))
        fallback = []
        for i in fallback_ids[:k]:
            frame_path = frame_files[i]
            stem = frame_path.stem
            mask_path = masks_dir / f"{stem}.png"
            fallback.append(
                KeyframeStats(
                    frame_name=frame_path.name,
                    frame_index=i,
                    mask_area_ratio=mask_area_ratio(mask_path) if mask_path.exists() else 0.0,
                )
            )
        return fallback

    candidates.sort(key=lambda x: x.frame_index)
    return pick_uniform_indices(candidates, k)


def draw_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    red = np.zeros_like(frame)
    red[:, :, 2] = 255
    pos = mask > 0
    vis[pos] = (0.7 * vis[pos] + 0.3 * red[pos]).astype(np.uint8)
    return vis


def run_copy_fallback(
    keyframes: List[KeyframeStats],
    input_frames_dir: Path,
    masks_dir: Path,
    out_refined: Path,
    out_debug: Path,
    save_debug: bool,
) -> None:
    out_refined.mkdir(parents=True, exist_ok=True)
    if save_debug:
        out_debug.mkdir(parents=True, exist_ok=True)

    for kf in keyframes:
        src = input_frames_dir / kf.frame_name
        img = cv2.imread(str(src))
        if img is None:
            continue
        cv2.imwrite(str(out_refined / kf.frame_name), img)

        if save_debug:
            stem = Path(kf.frame_name).stem
            mask_path = masks_dir / f"{stem}.png"
            if mask_path.exists():
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    overlay = draw_overlay(img, m)
                    cv2.imwrite(str(out_debug / f"{stem}_overlay.jpg"), overlay)


def compute_infer_size(width: int, height: int, max_side: int) -> Tuple[int, int]:
    if max_side > 0 and max(width, height) > max_side:
        scale = float(max_side) / float(max(width, height))
        width = int(round(width * scale))
        height = int(round(height * scale))
    # SD  8
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)
    return width, height


def build_control_image(image_rgb: np.ndarray, controlnet_type: str, canny_low: int, canny_high: int) -> np.ndarray:
    ctype = controlnet_type.lower().strip()
    if ctype != "canny":
        raise ValueError(f" canny : {controlnet_type}")
    edges = cv2.Canny(image_rgb, canny_low, canny_high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def save_debug_quad(
    out_debug: Path,
    stem: str,
    orig_bgr: np.ndarray,
    mask_gray: np.ndarray,
    control_rgb: np.ndarray,
    refined_bgr: np.ndarray,
) -> None:
    h, w = orig_bgr.shape[:2]
    mask_vis = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    control_bgr = cv2.cvtColor(control_rgb, cv2.COLOR_RGB2BGR)
    items = [
        cv2.resize(orig_bgr, (w, h), interpolation=cv2.INTER_AREA),
        cv2.resize(mask_vis, (w, h), interpolation=cv2.INTER_NEAREST),
        cv2.resize(control_bgr, (w, h), interpolation=cv2.INTER_AREA),
        cv2.resize(refined_bgr, (w, h), interpolation=cv2.INTER_AREA),
    ]
    quad = np.hstack(items)
    cv2.imwrite(str(out_debug / f"{stem}_quad.jpg"), quad)


def export_refined_keyframe_video(refined_dir: Path, out_path: Path, fps: float = 4.0) -> None:
    imgs = sorted(p for p in refined_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        return
    first = cv2.imread(str(imgs[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(im)
    writer.release()


def run_sd_controlnet_refine(
    keyframes: List[KeyframeStats],
    input_frames_dir: Path,
    masks_dir: Path,
    out_refined: Path,
    out_debug: Path,
    save_debug: bool,
    refine_cfg: Dict,
    runtime_cfg: Dict,
) -> None:
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
    except Exception as exc:
        raise RuntimeError(
            "sd_controlnet  diffusers/torch  controlnet_env"
        ) from exc

    sd_model_id = str(refine_cfg.get("sd_model_id", "runwayml/stable-diffusion-inpainting"))
    controlnet_model_id = str(refine_cfg.get("controlnet_model_id", "lllyasviel/control_v11p_sd15_canny"))
    controlnet_type = str(refine_cfg.get("controlnet_type", "canny"))
    prompt = str(refine_cfg.get("prompt", "clean background"))
    negative_prompt = str(refine_cfg.get("negative_prompt", "artifacts, blurry"))
    canny_low = int(refine_cfg.get("canny_low_threshold", 100))
    canny_high = int(refine_cfg.get("canny_high_threshold", 200))
    mask_dilate_px = int(refine_cfg.get("mask_dilate_px", 0))
    controlnet_scale = float(refine_cfg.get("controlnet_conditioning_scale", 0.8))
    guidance_scale = float(refine_cfg.get("guidance_scale", 7.5))
    strength = float(refine_cfg.get("strength", 0.99))
    num_steps = int(refine_cfg.get("num_inference_steps", 30))
    max_infer_side = int(refine_cfg.get("max_infer_side", 768))

    prefer_device = str(runtime_cfg.get("device", "auto")).lower()
    if prefer_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = prefer_device
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    base_seed = int(runtime_cfg.get("seed", 42))

    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        sd_model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device.startswith("cuda"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    out_refined.mkdir(parents=True, exist_ok=True)
    if save_debug:
        out_debug.mkdir(parents=True, exist_ok=True)

    for idx, kf in enumerate(keyframes):
        src = input_frames_dir / kf.frame_name
        stem = Path(kf.frame_name).stem
        mask_path = masks_dir / f"{stem}.png"
        img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if img_bgr is None or mask is None:
            continue

        if mask_dilate_px > 0:
            kernel = np.ones((mask_dilate_px, mask_dilate_px), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        control_rgb = build_control_image(img_rgb, controlnet_type, canny_low, canny_high)
        h, w = img_rgb.shape[:2]
        iw, ih = compute_infer_size(w, h, max_side=max_infer_side)

        init_pil = Image.fromarray(cv2.resize(img_rgb, (iw, ih), interpolation=cv2.INTER_AREA))
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(cv2.resize(mask_u8, (iw, ih), interpolation=cv2.INTER_NEAREST))
        control_pil = Image.fromarray(cv2.resize(control_rgb, (iw, ih), interpolation=cv2.INTER_AREA))

        if device.startswith("cuda"):
            gen = torch.Generator(device=device).manual_seed(base_seed + idx)
        else:
            gen = torch.Generator().manual_seed(base_seed + idx)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_pil,
            mask_image=mask_pil,
            control_image=control_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            controlnet_conditioning_scale=controlnet_scale,
            generator=gen,
        ).images[0]

        out_np = np.array(out.convert("RGB"))
        out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        if out_bgr.shape[:2] != (h, w):
            out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_refined / kf.frame_name), out_bgr)

        if save_debug:
            save_debug_quad(out_debug, stem, img_bgr, mask_u8, control_rgb, out_bgr)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    paths = cfg.get("paths", {})
    policy = cfg.get("keyframe_policy", {})
    refine = cfg.get("refine", {})
    runtime = cfg.get("runtime", {})

    input_frames_dir = Path(paths["input_frames_dir"])
    masks_dir = Path(paths["masks_dir"])
    output_dir = Path(paths["output_dir"])
    sequence_name = cfg.get("sequence_name", input_frames_dir.name)
    mode = refine.get("mode", "copy_fallback")

    if not input_frames_dir.exists():
        raise FileNotFoundError(f": {input_frames_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"mask: {masks_dir}")

    random.seed(int(runtime.get("seed", 42)))
    np.random.seed(int(runtime.get("seed", 42)))

    frame_files, frame_root = materialize_input_frames(input_frames_dir, output_dir)
    if not frame_files:
        raise RuntimeError(f": {input_frames_dir}")

    keyframes = select_keyframes(frame_files, masks_dir, policy)
    out_refined = output_dir / "refined_keyframes"
    out_debug = output_dir / "debug"

    if mode == "copy_fallback":
        run_copy_fallback(
            keyframes=keyframes,
            input_frames_dir=frame_root,
            masks_dir=masks_dir,
            out_refined=out_refined,
            out_debug=out_debug,
            save_debug=bool(runtime.get("save_debug_overlays", True)),
        )
        actual_backend = "copy_fallback"
    elif mode == "sd_controlnet":
        run_sd_controlnet_refine(
            keyframes=keyframes,
            input_frames_dir=frame_root,
            masks_dir=masks_dir,
            out_refined=out_refined,
            out_debug=out_debug,
            save_debug=bool(runtime.get("save_debug_overlays", True)),
            refine_cfg=refine,
            runtime_cfg=runtime,
        )
        if bool(runtime.get("export_refined_video", True)):
            export_refined_keyframe_video(out_refined, output_dir / "refined_keyframes.mp4")
        actual_backend = "sd_controlnet_diffusers"
    else:
        raise ValueError(f" refine.mode: {mode}")

    manifest = {
        "sequence_name": sequence_name,
        "input_frames_dir": str(input_frames_dir),
        "input_frame_root_actual": str(frame_root),
        "masks_dir": str(masks_dir),
        "output_dir": str(output_dir),
        "mode_requested": mode,
        "mode_actual": actual_backend,
        "num_total_frames": len(frame_files),
        "num_selected_keyframes": len(keyframes),
        "keyframes": [
            {
                "frame_name": kf.frame_name,
                "frame_index": kf.frame_index,
                "mask_area_ratio": round(kf.mask_area_ratio, 6),
            }
            for kf in keyframes
        ],
        "notes": (
            "Part3 skeleton run using copy_fallback."
            if actual_backend == "copy_fallback"
            else "Part3 run with real SD+ControlNet backend."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[ok] sequence={sequence_name}")
    print(f"[ok] selected_keyframes={len(keyframes)}")
    print(f"[save] {manifest_path}")


if __name__ == "__main__":
    main()
