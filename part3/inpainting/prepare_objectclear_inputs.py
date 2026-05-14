"""
prepare_objectclear_inputs.py — ObjectClear

 DAVIS frames + DAVIS GT masks  JPG  binary mask PNG
 run_objectclear_gtmask.py  ObjectClearPipeline

 DiffuEraser/ProPainter
  - input_video.mp4   ←  inpaint-only
  - input_mask.mp4    ←  mask ==
  - masked_in.mp4     ←  overlay  mask
  - mask_frames/      ←  mask PNG255=0=
  - imgs/             ←  JPG
  - masks/            ←  mask PNG mask_frames


  conda run -n objectclear_env python3 part3/inpainting/prepare_objectclear_inputs.py \\
      --seq tennis [--version v2] [--dilate_px 0] [--fps 24]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_MASKS  = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
RESULTS_ROOT = Path("/data3/jli657/project3/part3/results")


def load_sorted(d: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def write_video(out_path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()
    # Re-encode with ffmpeg for broad player compatibility
    # Search for ffmpeg in known locations
    for ffmpeg_bin in ["/data2/jli657/envs/diffueraser_env/bin/ffmpeg", "ffmpeg"]:
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", str(out_path)],
            capture_output=True,
        )
        if result.returncode == 0:
            tmp.unlink(missing_ok=True)
            break
    else:
        # Fall back: rename without re-encode
        tmp.rename(out_path)
    print(f"  [video] {out_path.name}  ({len(frames)} frames, {w}x{h} @ {fps:.1f}fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="tennis")
    parser.add_argument("--version", default="v2")
    parser.add_argument("--dilate_px", type=int, default=0,
                        help="Binary mask dilation in pixels (0 = no dilation)")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Truncate to first N frames (0 = all)")
    args = parser.parse_args()

    seq        = args.seq
    version    = args.version
    out_dir    = RESULTS_ROOT / seq / "direction_c" / f"objectclear_gtmask_{version}"
    imgs_dir   = out_dir / "imgs"
    masks_dir  = out_dir / "masks"
    mf_dir     = out_dir / "mask_frames"

    for d in [imgs_dir, masks_dir, mf_dir]:
        d.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted(DAVIS_FRAMES / seq)
    mask_paths  = load_sorted(DAVIS_MASKS / seq)

    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames found at {DAVIS_FRAMES / seq}")
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found at {DAVIS_MASKS / seq}")
    if len(frame_paths) != len(mask_paths):
        print(f"[WARN] frame count {len(frame_paths)} != mask count {len(mask_paths)}, using min")

    n = min(len(frame_paths), len(mask_paths))
    if args.max_frames > 0:
        n = min(n, args.max_frames)

    frame_paths = frame_paths[:n]
    mask_paths  = mask_paths[:n]

    # Dilation kernel
    if args.dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * args.dilate_px + 1, 2 * args.dilate_px + 1)
        )
    else:
        kernel = None

    video_frames: list[np.ndarray] = []
    mask_bgr_frames: list[np.ndarray] = []
    masked_in_frames: list[np.ndarray] = []

    for fp, mp in zip(frame_paths, mask_paths):
        stem = fp.stem

        frame_bgr = cv2.imread(str(fp))
        if frame_bgr is None:
            raise IOError(f"Cannot read frame: {fp}")

        # DAVIS annotations: palette PNG, foreground = any non-zero value
        mask_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise IOError(f"Cannot read mask: {mp}")
        if mask_raw.ndim == 3:
            # Use max across channels to handle palette PNGs where annotation
            # value may appear in any single channel (e.g. DAVIS car-shadow uses
            # only the blue channel with value 128, not the red channel)
            mask_raw = mask_raw.max(axis=2)

        binary = (mask_raw > 0).astype(np.uint8) * 255

        if kernel is not None:
            binary = cv2.dilate(binary, kernel)

        # Standard input_mask: white=inpaint, black=keep (BGR)
        mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Red overlay for masked_in.mp4 visual check
        overlay = frame_bgr.copy()
        overlay[binary > 0] = (
            overlay[binary > 0] * 0.4 + np.array([0, 0, 200]) * 0.6
        ).astype(np.uint8)

        video_frames.append(frame_bgr)
        mask_bgr_frames.append(mask_bgr)
        masked_in_frames.append(overlay)

        # Per-frame inference inputs
        cv2.imwrite(str(imgs_dir / f"{stem}.jpg"), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(masks_dir / f"{stem}.png"), binary)
        cv2.imwrite(str(mf_dir / f"{stem}.png"), binary)

    print(f"[prepare] seq={seq} version={version} frames={n} dilate={args.dilate_px}px")
    write_video(out_dir / "input_video.mp4", video_frames, fps=args.fps)
    write_video(out_dir / "input_mask.mp4",  mask_bgr_frames, fps=args.fps)
    write_video(out_dir / "masked_in.mp4",   masked_in_frames, fps=args.fps)
    print(f"  imgs/        → {imgs_dir}")
    print(f"  masks/       → {masks_dir}")
    print(f"  mask_frames/ → {mf_dir}")
    print(f"\nNext step:")
    print(f"  conda run -n objectclear_env python3 part3/inpainting/run_objectclear_gtmask.py \\")
    print(f"      --seq {seq} --version {version}")


if __name__ == "__main__":
    main()
