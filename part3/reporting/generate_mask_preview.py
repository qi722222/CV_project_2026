"""
generate_mask_preview.py —  masked_in.mp4 ()

:
  -  + mask PNG
  -  mask  ( ProPainter inference_propainter.py )
  -  masked_in.mp4

:
  #  sdxl_repair/bmx-trees  masked_in.mp4
  conda run -n propainter_env python3 part3/generate_mask_preview.py \
    --frames_dir /home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees \
    --masks_dir  /data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/bmx-trees \
    --output_mp4 /data3/jli657/project3/part3/outputs/sdxl_repair/bmx-trees/masked_in.mp4

  #  dilate  ProPainter  dilation=5
  python3 part3/generate_mask_preview.py ... --dilate 5
"""

import argparse
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import scipy.ndimage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True, help="Original video frames (jpg/png)")
    p.add_argument("--masks_dir", required=True, help="Binary mask PNGs directory")
    p.add_argument("--output_mp4", required=True, help="Output masked_in.mp4 path")
    p.add_argument("--fps", type=float, default=25.0, help="Output video FPS")
    p.add_argument("--alpha", type=float, default=0.6, help="Green overlay opacity (default 0.6, same as ProPainter)")
    p.add_argument("--dilate", type=int, default=5,
                   help="Mask dilation iterations (ProPainter default=5). Set 0 to skip.")
    p.add_argument("--quality", type=int, default=7, help="imageio quality for mp4 (1-9)")
    return p.parse_args()


def load_frames_sorted(frames_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem,
    )


def load_masks_sorted(masks_dir: Path):
    return sorted(masks_dir.glob("*.png"), key=lambda p: p.stem)


def main():
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    masks_dir = Path(args.masks_dir)
    output_mp4 = Path(args.output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = load_frames_sorted(frames_dir)
    mask_paths = load_masks_sorted(masks_dir)

    n = min(len(frame_paths), len(mask_paths))
    print(f"Frames: {len(frame_paths)}, Masks: {len(mask_paths)}, Processing: {n}")

    result_frames = []
    for i in range(n):
        frame = cv2.imread(str(frame_paths[i]))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        mask_img = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            mask_img = np.zeros((h, w), dtype=np.uint8)
        else:
            mask_img = (mask_img > 127).astype(np.uint8)
            if mask_img.shape != (h, w):
                mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

        # Dilate mask (same as ProPainter read_mask)
        if args.dilate > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=args.dilate).astype(np.uint8)

        # Green overlay (identical to ProPainter)
        mask_3ch = np.expand_dims(mask_img, 2).repeat(3, axis=2).astype(np.float32)
        frame_f = frame.astype(np.float32)
        green = np.zeros([h, w, 3], dtype=np.float32)
        green[:, :, 1] = 255.0  # RGB green

        fused = (1 - args.alpha) * frame_f + args.alpha * green
        composite = mask_3ch * fused + (1 - mask_3ch) * frame_f
        result_frames.append(composite.astype(np.uint8))

    if not result_frames:
        print("[error] No frames processed!")
        return

    imageio.mimwrite(str(output_mp4), result_frames, fps=args.fps, quality=args.quality)
    print(f"[done] Saved {len(result_frames)} frames → {output_mp4}")


if __name__ == "__main__":
    main()
