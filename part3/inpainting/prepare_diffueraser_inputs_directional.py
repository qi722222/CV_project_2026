"""
prepare_diffueraser_inputs_directional.py — DiffuEraser  mask car-shadow

 mask ****
  - /
  -

/
  - base dilate:  dilate_px
  - directional:  vertical_extra_px


  conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs_directional.py \\
      --seq car-shadow --version v13 \\
      --dilate_px 5 \\
      --bottom_extra_px 15 \\
      --side_extra_px 5
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
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()
    for ffmpeg_bin in ["/data2/jli657/envs/diffueraser_env/bin/ffmpeg", "ffmpeg"]:
        r = subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", str(out_path)],
            capture_output=True,
        )
        if r.returncode == 0:
            tmp.unlink(missing_ok=True)
            break
    else:
        tmp.rename(out_path)
    print(f"  [video] {out_path.name}  ({len(frames)} frames, {w}x{h} @ {fps:.1f}fps)")


def directional_dilate(binary: np.ndarray, dilate_px: int, bottom_extra_px: int, side_extra_px: int) -> np.ndarray:
    """
    1. Base isotropic dilation (ellipse kernel of dilate_px)
    2. Additional bottom-only dilation (tall vertical strip kernel)
    3. Additional side (left-right) dilation (horizontal kernel)
    Combined result = max of all dilated versions
    """
    result = binary.copy()

    if dilate_px > 0:
        k_base = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        result = cv2.dilate(result, k_base)

    # Bottom-only: use a vertical rectangle that extends downward
    # Build custom kernel: 1 pixel wide column, height = bottom_extra_px, all below center
    if bottom_extra_px > 0:
        # asymmetric: tall kernel where center is at top (extends only down)
        h_k = bottom_extra_px * 2 + 1
        k_bottom = np.zeros((h_k, 1), dtype=np.uint8)
        k_bottom[bottom_extra_px:, :] = 1  # only lower half active
        bottom_dil = cv2.dilate(result, k_bottom)
        result = cv2.bitwise_or(result, bottom_dil)

    # Side expansion (horizontal)
    if side_extra_px > 0:
        k_side = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * side_extra_px + 1, 1)
        )
        side_dil = cv2.dilate(result, k_side)
        result = cv2.bitwise_or(result, side_dil)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="car-shadow")
    parser.add_argument("--version", default="v13")
    parser.add_argument("--dilate_px", type=int, default=5,
                        help="Base isotropic dilation (pixels)")
    parser.add_argument("--bottom_extra_px", type=int, default=15,
                        help="Extra downward dilation to cover shadow gradient")
    parser.add_argument("--side_extra_px", type=int, default=5,
                        help="Extra left-right dilation for shadow edges")
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()

    seq    = args.seq
    out_dir = RESULTS_ROOT / seq / "direction_c" / f"diffueraser_gtmask_{args.version}"
    mf_dir  = out_dir / "mask_frames"
    mf_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted(DAVIS_FRAMES / seq)
    mask_paths  = load_sorted(DAVIS_MASKS / seq)
    n = min(len(frame_paths), len(mask_paths))

    video_frames: list[np.ndarray] = []
    mask_bgr_frames: list[np.ndarray] = []
    masked_in_frames: list[np.ndarray] = []

    for fp, mp in zip(frame_paths[:n], mask_paths[:n]):
        stem = fp.stem

        frame_bgr = cv2.imread(str(fp))
        mask_raw  = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if frame_bgr is None or mask_raw is None:
            raise IOError(f"Cannot read {fp} or {mp}")
        if mask_raw.ndim == 3:
            # Use max across channels; cvtColor(GRAY) also works but max is safer
            # for palette PNGs where only one channel carries the annotation value
            mask_raw = mask_raw.max(axis=2)

        binary = (mask_raw > 0).astype(np.uint8) * 255

        binary = directional_dilate(
            binary,
            dilate_px=args.dilate_px,
            bottom_extra_px=args.bottom_extra_px,
            side_extra_px=args.side_extra_px,
        )

        mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        overlay = frame_bgr.copy()
        overlay[binary > 0] = (
            overlay[binary > 0] * 0.4 + np.array([0, 0, 200]) * 0.6
        ).astype(np.uint8)

        video_frames.append(frame_bgr)
        mask_bgr_frames.append(mask_bgr)
        masked_in_frames.append(overlay)
        cv2.imwrite(str(mf_dir / f"{stem}.png"), binary)

    print(f"[prepare] seq={seq} version={args.version} frames={n} "
          f"dilate={args.dilate_px}px bottom_extra={args.bottom_extra_px}px side_extra={args.side_extra_px}px")
    write_video(out_dir / "input_video.mp4", video_frames, fps=args.fps)
    write_video(out_dir / "input_mask.mp4",  mask_bgr_frames, fps=args.fps)
    write_video(out_dir / "masked_in.mp4",   masked_in_frames, fps=args.fps)

    print(f"\n[OK] Inputs ready at {out_dir}")
    print(f"  mask_frames/ : {n} frames")
    print(f"\nNext:")
    print(f"  conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py \\")
    print(f"      --seq {seq} --version {args.version}")


if __name__ == "__main__":
    main()
