"""
run_propainter_gtmask.py —  ProPainter (GT mask )

mask :
  - DAVIS :  DAVIS annotation / GT mask ( inpaint-only )
  - wild_video-1person:  shadow/SAM3 mask (demo ,  GT )

:
  - DAVIS: results/<seq>/direction_c/pure_propainter_gtmask/
  - wild:  results/<seq>/direction_c/pure_propainter/  (, )

:
  #  DAVIS
  conda run -n propainter_env python3 part3/run_propainter_gtmask.py \
      --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

  #
  conda run -n propainter_env python3 part3/run_propainter_gtmask.py --seqs tennis
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage

PROPAINTER_PYTHON = "/data2/jli657/envs/propainter_env/bin/python"
PROPAINTER_DIR    = "/data2/jli657/ProPainter"
RESULTS           = Path("/data3/jli657/project3/part3/results")
MASKS_FINAL       = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final")
DAVIS_FRAMES      = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS    = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
WILD_FRAMES       = Path("/data3/jli657/project3/wild_frames")

SEQUENCES_CFG = {
    "tennis":             {"wild": False, "frames_override": None},
    "koala":              {"wild": False, "frames_override": None},
    "wild_video-1person": {"wild": True,  "frames_override": str(WILD_FRAMES / "wild_video-1person")},
    "bmx-trees":          {"wild": False, "frames_override": None},
    "blackswan":          {"wild": False, "frames_override": None},
    "horsejump-low":      {"wild": False, "frames_override": None},
    "car-shadow":         {"wild": False, "frames_override": None},
}


def resolve_mask_dir(seq: str, cfg: dict) -> tuple[Path, str]:
    """Return (mask_dir, mask_protocol) for a sequence.

    DAVIS sequences: use DAVIS GT annotations (fair inpaint-only comparison).
    wild_video-1person: use existing shadow/SAM3 mask (demo group, no GT available).
    """
    if cfg.get("wild"):
        wild_mask = MASKS_FINAL / "wild_video-1person_with_shadow"
        if not wild_mask.exists():
            wild_mask = MASKS_FINAL / "wild_video-1person"
        return wild_mask, "wild_existing_mask"
    gt_dir = DAVIS_GT_MASKS / seq
    if gt_dir.exists():
        return gt_dir, "davis_gt"
    fallback = MASKS_FINAL / seq
    return fallback, "sam3_fallback"


def load_sorted_imgs(d: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def dilate_masks(src_dir: Path, dst_dir: Path, kernel_size: int = 9) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8) * 255
        if kernel_size > 1:
            k = np.ones((kernel_size, kernel_size), np.uint8)
            m = cv2.dilate(m, k, iterations=1)
        cv2.imwrite(str(dst_dir / p.name), m)


def run_propainter(frames_dir: str, masks_dir: str, output_dir: str) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        PROPAINTER_PYTHON, "inference_propainter.py",
        "--video", frames_dir,
        "--mask",  masks_dir,
        "--output", output_dir,
        "--resize_ratio", "1.0",
        "--neighbor_length", "10",
        "--ref_stride", "10",
    ]
    print(f"  [ProPainter] {frames_dir} -> {output_dir}")
    r = subprocess.run(cmd, cwd=PROPAINTER_DIR, env=env, timeout=900)
    return r.returncode == 0


def generate_masked_preview(masks_dir: Path, frames_dir: Path, output_mp4: Path) -> None:
    if output_mp4.exists():
        return
    try:
        import imageio
    except ImportError:
        print("  [WARN] imageio not available, skipping masked_in preview")
        return
    frame_paths = load_sorted_imgs(frames_dir)
    mask_paths  = sorted(masks_dir.glob("*.png"), key=lambda p: p.stem)
    n = min(len(frame_paths), len(mask_paths))
    result_frames = []
    for i in range(n):
        frame = cv2.imread(str(frame_paths[i]))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        msk = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.zeros((h, w), np.uint8)
        else:
            msk = (msk > 0).astype(np.uint8)
            if msk.shape != (h, w):
                msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
        msk = scipy.ndimage.binary_dilation(msk, iterations=5).astype(np.uint8)
        msk3 = np.expand_dims(msk, 2).repeat(3, axis=2).astype(np.float32)
        ff = frame.astype(np.float32)
        g = np.zeros([h, w, 3], np.float32); g[:, :, 1] = 255.0
        composite = msk3 * (0.4 * ff + 0.6 * g) + (1 - msk3) * ff
        result_frames.append(composite.astype(np.uint8))
    if result_frames:
        imageio.mimwrite(str(output_mp4), result_frames, fps=25.0, quality=7)
        print(f"  [preview] {len(result_frames)} frames -> {output_mp4.name}")


def process_sequence(seq: str, cfg: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"[{seq}] Pure ProPainter")
    print('='*60)

    orig_seq_name = "wild_video-1person" if cfg.get("wild") else seq
    frames_dir = Path(cfg["frames_override"]) if cfg.get("frames_override") else DAVIS_FRAMES / orig_seq_name

    masks_dir, mask_protocol = resolve_mask_dir(seq, cfg)
    print(f"  [mask_protocol={mask_protocol}] {masks_dir}")

    # Output directory: GT-mask runs get a distinct subdirectory to preserve old results
    if mask_protocol == "davis_gt":
        out_subdir = "pure_propainter_gtmask"
    else:
        out_subdir = "pure_propainter"

    out_dir = RESULTS / seq / "direction_c" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    inpaint_mp4 = out_dir / "inpaint_out.mp4"
    if inpaint_mp4.exists():
        print(f"  [skip] {out_subdir}/inpaint_out.mp4 already exists")
        return True

    # Dilate GT masks slightly for ProPainter (standard 9px dilation)
    dilated_dir = out_dir / "masks_dilated"
    dilate_masks(masks_dir, dilated_dir, kernel_size=9)

    # Run ProPainter
    pp_out = out_dir / "propainter_output"
    pp_out.mkdir(exist_ok=True)
    success = run_propainter(str(frames_dir), str(dilated_dir), str(pp_out))
    if not success:
        print(f"  [ERROR] ProPainter failed for {seq}")
        return False

    # Copy output video to standardised location
    candidates = [
        pp_out / "inpaint_out.mp4",
        pp_out / orig_seq_name / "inpaint_out.mp4",
    ]
    pp_video = None
    for c in candidates:
        if c.exists():
            pp_video = c
            break
    if pp_video is None:
        mp4s = list(pp_out.rglob("inpaint_out.mp4"))
        if mp4s:
            pp_video = mp4s[0]

    if pp_video and pp_video != inpaint_mp4:
        shutil.copy2(pp_video, inpaint_mp4)
        print(f"  [copy] inpaint_out.mp4 -> {inpaint_mp4}")
    elif not inpaint_mp4.exists():
        print(f"  [ERROR] Could not find ProPainter output video for {seq}")
        return False

    # Generate masked_in.mp4 overlay
    generate_masked_preview(masks_dir, frames_dir, out_dir / "masked_in.mp4")
    print(f"  [OK] {seq} -> {out_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure ProPainter with DAVIS GT masks (fair inpaint-only protocol)")
    parser.add_argument("--seqs", nargs="+", default=list(SEQUENCES_CFG.keys()))
    args = parser.parse_args()

    for seq in args.seqs:
        if seq not in SEQUENCES_CFG:
            print(f"[WARN] unknown sequence: {seq}, skip")
            continue
        try:
            process_sequence(seq, SEQUENCES_CFG[seq])
        except Exception as e:
            print(f"[ERROR] {seq}: {e}")

    print("\n[run_propainter_gtmask complete]")


if __name__ == "__main__":
    main()
