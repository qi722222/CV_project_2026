#!/usr/bin/env python3
"""
apply_hard_blend.py
Post-process a DiffuEraser inpainted video by hard-blending:
  - Inside mask  → keep inpainted pixels
  - Outside mask → replace with DAVIS original JPEG frames (not re-encoded input_video.mp4)
This eliminates soft-blending leakage (white phantom images in non-masked areas)
and avoids double-encode compression on non-masked pixels.

Usage:
  python apply_hard_blend.py \\
    --base_version v1 \\
    --out_version  v4 \\
    --results_root /data3/jli657/project3/part3/results/tennis/direction_c \\
    --sequence tennis \\
    [--feather 0]  # optional: Gaussian feather width at mask boundary (pixels)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


# ─── helpers ──────────────────────────────────────────────────────────────────

def read_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames, fps = [], cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video_frames(frames, out_path: Path, fps: float):
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def load_mask_png(path: Path, feather: int = 0) -> np.ndarray:
    """Load a binary mask PNG; returns float32 array in [0,1] (1=mask/inpaint area)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    binary = (img > 127).astype(np.float32)
    if feather > 0 and feather % 2 == 0:
        feather += 1  # kernel size must be odd
    if feather > 0:
        binary = cv2.GaussianBlur(binary, (feather, feather), 0)
    return binary


def hard_blend(orig: np.ndarray, inpaint: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend: mask*inpaint + (1-mask)*orig  (mask is H×W float32 in [0,1])."""
    m = mask[:, :, np.newaxis]
    return (m * inpaint.astype(np.float32) + (1.0 - m) * orig.astype(np.float32)).clip(0, 255).astype(np.uint8)


# ─── main ─────────────────────────────────────────────────────────────────────

DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")


def load_image_dir(dir_path: Path):
    """Load all JPEG/PNG images from a directory, sorted by filename."""
    frames = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files = sorted(dir_path.glob(ext))
        if files:
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    frames.append(img)
            break
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_version", default="v1")
    ap.add_argument("--out_version",  default="v4")
    ap.add_argument("--results_root", default="/data3/jli657/project3/part3/results/tennis/direction_c")
    ap.add_argument("--sequence", default="tennis",
                    help="DAVIS sequence name (used to locate original JPEG frames)")
    ap.add_argument("--feather", type=int, default=0,
                    help="Gaussian feather width at mask boundary (0 = hard cut)")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    base_dir = results_root / f"diffueraser_gtmask_{args.base_version}"
    out_dir  = results_root / f"diffueraser_gtmask_{args.out_version}"

    # validate inputs
    for p in [base_dir / "inpaint_out.mp4",
              base_dir / "input_video.mp4",
              base_dir / "mask_frames"]:
        if not Path(p).exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load videos ──────────────────────────────────────────────────────────
    print(f"Loading base inpainted video from {base_dir/'inpaint_out.mp4'} ...")
    inpaint_frames, fps = read_video_frames(base_dir / "inpaint_out.mp4")

    # Use DAVIS original JPEG frames directly (not re-encoded input_video.mp4)
    # to avoid double-encode compression artifacts in non-masked areas
    davis_jpg_dir = DAVIS_FRAMES / args.sequence
    if davis_jpg_dir.exists():
        print(f"Loading DAVIS original JPEG frames from {davis_jpg_dir} ...")
        orig_frames = load_image_dir(davis_jpg_dir)
        print(f"  Loaded {len(orig_frames)} DAVIS frames")
    else:
        print(f"WARN: DAVIS dir not found at {davis_jpg_dir}, falling back to input_video.mp4")
        orig_frames, _ = read_video_frames(base_dir / "input_video.mp4")

    # Also use DAVIS GT masks for hard blend if available
    gt_mask_dir = DAVIS_GT_MASKS / args.sequence
    if gt_mask_dir.exists():
        gt_mask_files = sorted(gt_mask_dir.glob("*.png"))
        print(f"Using DAVIS GT mask dir: {gt_mask_dir} ({len(gt_mask_files)} masks)")
        use_gt_masks = True
    else:
        print(f"WARN: DAVIS GT mask dir not found, using mask_frames from base version")
        use_gt_masks = False

    # ── load mask frames ─────────────────────────────────────────────────────
    if use_gt_masks:
        mask_files = gt_mask_files
    else:
        mask_dir = base_dir / "mask_frames"
        mask_files = sorted(mask_dir.glob("*.png"))
    if len(mask_files) == 0:
        print("ERROR: no PNG files in mask_frames", file=sys.stderr)
        sys.exit(1)

    n = len(inpaint_frames)
    if len(orig_frames) != n:
        print(f"WARN: inpaint has {n} frames, orig has {len(orig_frames)}")
    if len(mask_files) != n:
        print(f"WARN: inpaint has {n} frames, masks has {len(mask_files)}")

    # ── hard blend ────────────────────────────────────────────────────────────
    print(f"Hard-blending {n} frames (feather={args.feather}) ...")
    blended = []
    for i in range(n):
        orig_f   = orig_frames[min(i, len(orig_frames) - 1)]
        inp_f    = inpaint_frames[i]
        mask_f   = load_mask_png(mask_files[min(i, len(mask_files) - 1)], feather=args.feather)

        # align all frames to original video resolution
        target_h, target_w = orig_f.shape[:2]
        if inp_f.shape[:2] != (target_h, target_w):
            inp_f = cv2.resize(inp_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        if mask_f.shape[:2] != (target_h, target_w):
            mask_f = cv2.resize(mask_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        blended.append(hard_blend(orig_f, inp_f, mask_f))

    # ── write outputs ─────────────────────────────────────────────────────────
    out_inpaint = out_dir / "inpaint_out.mp4"
    print(f"Writing blended video to {out_inpaint} ...")
    write_video_frames(blended, out_inpaint, fps)
    print(f"Output resolution: {blended[0].shape[1]}x{blended[0].shape[0]}")

    # copy supporting files from base version
    for fname in ["input_video.mp4", "input_mask.mp4", "masked_in.mp4"]:
        src = base_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fname}")

    # symlink mask_frames from base version
    dst_masks = out_dir / "mask_frames"
    if not dst_masks.exists():
        src_mask_dir = base_dir / "mask_frames"
        if src_mask_dir.exists():
            dst_masks.symlink_to(src_mask_dir.resolve())
            print("Symlinked mask_frames from base version")

    # ── write run_manifest.json ───────────────────────────────────────────────
    manifest = {
        "exp_id":          f"diffueraser_gtmask_{args.out_version}",
        "readable_name":   f"DiffuEraser GT-mask {args.out_version} (hard blend from {args.base_version})",
        "sequence":        "tennis",
        "family":          "diffueraser",
        "comparison_type": "inpaint_only",
        "direction":       "C",
        "version":         args.out_version,
        "mask_protocol":   "davis_gt",
        "baseline":        f"diffueraser_gtmask_{args.base_version}",
        "audit_status":    "exploratory",
        "stage_gate":      "PSNR_proxy>=34.88 AND SSIM>=0.92",
        "motivation":      "Hard-blend to eliminate soft-blending leakage (white phantom images outside mask)",
        "changes_from_previous": (
            f"Post-process {args.base_version} inpaint_out: replace pixels outside GT mask "
            f"with DAVIS original JPEG frames (no re-encode); feather={args.feather}px"
        ),
        "expected_gain":   "PSNR_proxy should reach ~inf (mask-outside fully preserved); "
                           "PSNR_synthetic unchanged (~v1 level); SSIM close to 1.0",
        "next_decision":   "pending_eval",
        "failure_reason":  "",
        "parameters": {
            "base_version":   args.base_version,
            "feather":        args.feather,
            "hard_blend":     True,
        },
        "created_at": datetime.now().isoformat(),
        "script":     "apply_hard_blend.py",
    }

    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
