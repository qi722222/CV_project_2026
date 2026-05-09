"""
prepare_diffueraser_inputs.py — DiffuEraser 输入视频和 mask 视频准备

从 DAVIS frames + DAVIS GT masks 合成两个视频：
  - input_video.mp4  ← 原始帧序列
  - input_mask.mp4   ← 二值 mask 序列（白色=待修复区域，黑色=保留区域）

同时生成：
  - masked_in.mp4    ← 用于视觉核查 mask 覆盖是否正确

DiffuEraser 要求：
  - input_video 和 input_mask 帧数一致、分辨率一致
  - mask 为白色区域（255）= inpainting 区域，黑色（0）= 保留区域
  - 建议分辨率 ≤ 512x288 或保持 480p，根据显存调整

用法：
  conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py \\
      --seq tennis [--max_frames 70] [--dilate_px 5] [--fps 24]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_MASKS  = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
RESULTS_ROOT = Path("/data3/jli657/project3/part3/results")


def load_sorted_images(d: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def write_video(out_path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        writer.write(f)
    writer.release()
    # Re-encode with ffmpeg for compatibility
    tmp = out_path.with_suffix(".tmp.mp4")
    out_path.rename(tmp)
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", str(out_path)],
        check=True, capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    print(f"  [video] {out_path}  ({len(frames)} frames, {w}x{h} @ {fps:.1f}fps)")


def dilate_mask(mask_bin: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask_bin, kernel, iterations=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="DAVIS sequence name, e.g. tennis")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to use (0 = all frames)")
    parser.add_argument("--dilate_px", type=int, default=5,
                        help="Dilation radius in pixels applied to GT mask (default: 5)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--version", default="v1",
                        help="Version tag used to name output directory, e.g. v1")
    args = parser.parse_args()

    seq = args.seq
    frame_dir  = DAVIS_FRAMES / seq
    mask_dir   = DAVIS_MASKS  / seq
    out_dir    = RESULTS_ROOT / seq / "direction_c" / f"diffueraser_gtmask_{args.version}"

    if not frame_dir.exists():
        print(f"[ERROR] Frame dir not found: {frame_dir}")
        sys.exit(1)
    if not mask_dir.exists():
        print(f"[ERROR] Mask dir not found: {mask_dir}")
        sys.exit(1)

    frame_paths = load_sorted_images(frame_dir, {".jpg", ".jpeg", ".png"})
    mask_paths  = load_sorted_images(mask_dir,  {".png"})

    if len(frame_paths) != len(mask_paths):
        print(f"[WARN] Frame count {len(frame_paths)} ≠ Mask count {len(mask_paths)}")
        n = min(len(frame_paths), len(mask_paths))
        frame_paths, mask_paths = frame_paths[:n], mask_paths[:n]

    if args.max_frames > 0:
        frame_paths = frame_paths[:args.max_frames]
        mask_paths  = mask_paths[:args.max_frames]

    print(f"[prepare] seq={seq}  frames={len(frame_paths)}  dilate={args.dilate_px}px")

    video_frames: list[np.ndarray] = []
    mask_frames:  list[np.ndarray] = []
    masked_frames: list[np.ndarray] = []

    for fp, mp in zip(frame_paths, mask_paths):
        img  = cv2.imread(str(fp))
        anno = cv2.imread(str(mp))  # DAVIS annotations: color-coded

        if img is None:
            print(f"[WARN] cannot read {fp}")
            continue
        if anno is None:
            print(f"[WARN] cannot read {mp}")
            continue

        # Convert DAVIS annotation to binary mask: any non-zero pixel = foreground
        anno_gray = cv2.cvtColor(anno, cv2.COLOR_BGR2GRAY)
        binary = (anno_gray > 0).astype(np.uint8) * 255
        if args.dilate_px > 0:
            binary = dilate_mask(binary, args.dilate_px)

        # DiffuEraser expects: white=inpaint, black=keep  (same as ProPainter convention)
        mask_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # masked_in: show the region to be inpainted overlaid with red tint
        masked = img.copy()
        masked[binary > 127] = (masked[binary > 127] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)

        video_frames.append(img)
        mask_frames.append(mask_rgb)
        masked_frames.append(masked)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save individual mask frames for evaluation reference
    mask_frames_dir = out_dir / "mask_frames"
    mask_frames_dir.mkdir(exist_ok=True)
    for idx, (mp, binary_rgb) in enumerate(zip(mask_paths, mask_frames)):
        cv2.imwrite(str(mask_frames_dir / f"{idx:05d}.png"), binary_rgb)

    print(f"[prepare] writing videos to {out_dir} ...")
    write_video(out_dir / "input_video.mp4", video_frames, args.fps)
    write_video(out_dir / "input_mask.mp4",  mask_frames,  args.fps)
    write_video(out_dir / "masked_in.mp4",   masked_frames, args.fps)

    print(f"\n[OK] Inputs ready:")
    print(f"  input_video.mp4 → {out_dir / 'input_video.mp4'}")
    print(f"  input_mask.mp4  → {out_dir / 'input_mask.mp4'}")
    print(f"  masked_in.mp4   → {out_dir / 'masked_in.mp4'}")
    print(f"  mask_frames/    → {mask_frames_dir}  ({len(frame_paths)} frames)")
    print(f"\nNext: run DiffuEraser inference with these inputs")


if __name__ == "__main__":
    main()
