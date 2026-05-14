"""
prepare_minimax_masks.py — MiniMax-Remover mask refinement

Produces improved mask PNGs before passing to run_minimax_remover_gtmask.py.
Three orthogonal refinement stages, all optional and combinable via --variant:

  1. Edge halo dilation (--dilate_px): isotropic ellipse kernel.
     Removes white edge artifacts caused by hard-blend at mask boundary.

  2. Directional shadow extension (--bottom_extra_px / --side_extra_px):
     Extends mask downward to cover ground shadows, sideways for contact shadows.
     car-shadow:     --bottom_extra_px 20 --side_extra_px 6
     horsejump-low:  --bottom_extra_px 10 --side_extra_px 3
     blackswan:      --bottom_extra_px 6  --side_extra_px 2

  3. Feather mask for soft-blend (--feather_px N, N>0):
     Gaussian-blurs the binary mask boundary so run_minimax_remover_gtmask.py
     can alpha-blend MiniMax output vs original frame with smooth transition.
     Result saved as separate feather_mask_frames/ alongside binary mask_frames/.

Output directory:
  /data3/jli657/project3/part3/outputs/minimax_masks/<variant>/<seq>/

Usage:
  # horsejump-low: edge dilation + contact-shadow + feather
  python3 part3/inpainting/prepare_minimax_masks.py \\
      --seq horsejump-low --variant shadow_edge_v1 \\
      --base_mask_dir /home/jli657/shared_data/project3/DAVIS/Annotations/480p/horsejump-low \\
      --dilate_px 12 --bottom_extra_px 10 --side_extra_px 3 --feather_px 8

  # car-shadow: strong downward shadow extension
  python3 part3/inpainting/prepare_minimax_masks.py \\
      --seq car-shadow --variant shadow_edge_v1 \\
      --base_mask_dir /home/jli657/shared_data/project3/DAVIS/Annotations/480p/car-shadow \\
      --dilate_px 8 --bottom_extra_px 20 --side_extra_px 6 --feather_px 10

  # blackswan: mild water-reflection boundary expansion
  python3 part3/inpainting/prepare_minimax_masks.py \\
      --seq blackswan --variant shadow_edge_v1 \\
      --base_mask_dir /home/jli657/shared_data/project3/DAVIS/Annotations/480p/blackswan \\
      --dilate_px 10 --bottom_extra_px 6 --side_extra_px 2 --feather_px 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

OUTPUTS_ROOT  = Path("/data3/jli657/project3/part3/outputs/minimax_masks")
DAVIS_FRAMES  = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_MASKS   = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

# Sequence-specific defaults used when no explicit flags are supplied
SEQUENCE_DEFAULTS: dict[str, dict] = {
    "car-shadow":    {"dilate_px": 8,  "bottom_extra_px": 20, "side_extra_px": 6,  "feather_px": 10},
    "horsejump-low": {"dilate_px": 12, "bottom_extra_px": 10, "side_extra_px": 3,  "feather_px": 8},
    "blackswan":     {"dilate_px": 10, "bottom_extra_px": 6,  "side_extra_px": 2,  "feather_px": 8},
    "koala":         {"dilate_px": 12, "bottom_extra_px": 4,  "side_extra_px": 2,  "feather_px": 8},
    "bmx-trees":     {"dilate_px": 8,  "bottom_extra_px": 6,  "side_extra_px": 4,  "feather_px": 8},
    "tennis":        {"dilate_px": 10, "bottom_extra_px": 4,  "side_extra_px": 2,  "feather_px": 8},
    "bear":          {"dilate_px": 10, "bottom_extra_px": 6,  "side_extra_px": 2,  "feather_px": 8},
    "camel":         {"dilate_px": 10, "bottom_extra_px": 6,  "side_extra_px": 2,  "feather_px": 8},
}


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_sorted(d: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def load_binary_mask(path: Path, h: int, w: int) -> np.ndarray:
    """Load mask PNG as binary uint8 (0 or 255), resized to (h, w)."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return np.zeros((h, w), dtype=np.uint8)
    if m.ndim == 3:
        m = m.max(axis=2)
    m = (m > 0).astype(np.uint8) * 255
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


def write_video(out_path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()
    print(f"  [video] {out_path.name}  ({len(frames)} frames)")


# ── Mask refinement ───────────────────────────────────────────────────────────

def refine_mask(
    binary: np.ndarray,
    dilate_px: int,
    bottom_extra_px: int,
    side_extra_px: int,
) -> np.ndarray:
    """Apply isotropic dilation, then directional shadow/contact extension.

    Stage 1 — isotropic ellipse dilation (covers edge halos and slight boundary
    imprecision from hard-blend artifacts).

    Stage 2 — downward extension (covers ground shadow gradient that the object
    mask does not reach: car-shadow, horsejump-low contact shadow).

    Stage 3 — sideways extension (covers penumbra/horizontal shadow spread and
    motion-blur tails).

    All stages are unioned so no existing mask coverage is lost.
    """
    result = binary.copy()

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        result = cv2.dilate(result, k)

    if bottom_extra_px > 0:
        # Asymmetric vertical kernel: center at top, extends only downward.
        h_k = bottom_extra_px * 2 + 1
        k_down = np.zeros((h_k, 1), dtype=np.uint8)
        k_down[bottom_extra_px:, :] = 1
        bottom_dil = cv2.dilate(result, k_down)
        result = cv2.bitwise_or(result, bottom_dil)

    if side_extra_px > 0:
        k_side = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * side_extra_px + 1, 1)
        )
        side_dil = cv2.dilate(result, k_side)
        result = cv2.bitwise_or(result, side_dil)

    return result


def feather_mask(binary: np.ndarray, feather_px: int) -> np.ndarray:
    """Create a float32 [0, 1] soft mask by Gaussian-blurring the binary mask.

    The soft mask is used in run_minimax_remover_gtmask.py --soft_blend so the
    final video transitions smoothly at the mask boundary instead of hard-cutting.
    Inside the mask the value stays close to 1; outside it falls to 0 over
    feather_px pixels.
    """
    if feather_px <= 0:
        return (binary > 0).astype(np.float32)
    # Kernel must be odd and large enough for the requested feather radius.
    ksize = max(3, feather_px * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    sigma = feather_px / 3.0
    soft = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (ksize, ksize), sigma)
    # Clip so interior stays at 1.0 (binary interior was 255 → 1.0 before blur).
    return np.clip(soft, 0.0, 1.0)


# ── Preview ───────────────────────────────────────────────────────────────────

def overlay_mask_on_frame(frame_bgr: np.ndarray, binary: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Green tint overlay where mask is active."""
    h, w = frame_bgr.shape[:2]
    if binary.shape[:2] != (h, w):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    active = (binary > 0).astype(np.float32)[:, :, None]
    green = np.zeros((h, w, 3), dtype=np.float32)
    green[:, :, 1] = 255.0
    overlay = (
        active * ((1.0 - alpha) * frame_bgr.astype(np.float32) + alpha * green)
        + (1.0 - active) * frame_bgr.astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare refined mask PNGs for MiniMax-Remover"
    )
    parser.add_argument("--seq",           default="horsejump-low",
                        help="DAVIS sequence name")
    parser.add_argument("--variant",       default="shadow_edge_v1",
                        help="Variant tag, used in output directory name")
    parser.add_argument("--base_mask_dir", type=str, default=None,
                        help="Source mask PNGs. Default: DAVIS GT annotation dir.")
    parser.add_argument("--dilate_px",       type=int, default=None,
                        help="Isotropic dilation (pixels). Default: sequence preset.")
    parser.add_argument("--bottom_extra_px", type=int, default=None,
                        help="Extra downward extension for ground shadow coverage.")
    parser.add_argument("--side_extra_px",   type=int, default=None,
                        help="Extra sideways extension for penumbra coverage.")
    parser.add_argument("--feather_px",      type=int, default=None,
                        help="Gaussian feather radius for soft-blend mask output.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output directory if it exists.")
    args = parser.parse_args()

    # Resolve per-sequence defaults for any unspecified params
    seq_defaults = SEQUENCE_DEFAULTS.get(args.seq, {
        "dilate_px": 10, "bottom_extra_px": 6, "side_extra_px": 2, "feather_px": 8
    })
    dilate_px       = args.dilate_px       if args.dilate_px       is not None else seq_defaults["dilate_px"]
    bottom_extra_px = args.bottom_extra_px if args.bottom_extra_px is not None else seq_defaults["bottom_extra_px"]
    side_extra_px   = args.side_extra_px   if args.side_extra_px   is not None else seq_defaults["side_extra_px"]
    feather_px      = args.feather_px      if args.feather_px      is not None else seq_defaults["feather_px"]

    # Source directories
    frames_dir = DAVIS_FRAMES / args.seq
    masks_dir  = Path(args.base_mask_dir) if args.base_mask_dir else (DAVIS_MASKS / args.seq)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask dir not found: {masks_dir}")

    # Output directories
    out_dir    = OUTPUTS_ROOT / args.variant / args.seq
    mf_dir     = out_dir / "mask_frames"
    feat_dir   = out_dir / "feather_mask_frames"

    if out_dir.exists() and not args.overwrite:
        existing = list(mf_dir.glob("*.png")) if mf_dir.exists() else []
        if existing:
            print(f"[skip] {out_dir} already exists with {len(existing)} masks. "
                  "Use --overwrite to regenerate.")
            return

    mf_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted(frames_dir)
    mask_paths  = load_sorted(masks_dir, {".png"})
    n = min(len(frame_paths), len(mask_paths))

    if n == 0:
        raise RuntimeError(f"No frames/masks found in {frames_dir} / {masks_dir}")

    # Align mask count to frame count
    if len(mask_paths) < len(frame_paths):
        mask_paths = mask_paths + [mask_paths[-1]] * (len(frame_paths) - len(mask_paths))
        mask_paths = mask_paths[:n]

    print(f"[prepare_minimax_masks] seq={args.seq}  variant={args.variant}  frames={n}")
    print(f"  base_mask_dir: {masks_dir}")
    print(f"  dilate_px={dilate_px}  bottom_extra_px={bottom_extra_px}  "
          f"side_extra_px={side_extra_px}  feather_px={feather_px}")
    print(f"  output: {out_dir}")

    # Reference frame size
    ref_frame = cv2.imread(str(frame_paths[0]))
    if ref_frame is None:
        raise IOError(f"Cannot read reference frame: {frame_paths[0]}")
    h_ref, w_ref = ref_frame.shape[:2]

    masked_in_frames: list[np.ndarray] = []
    orig_mask_coverage: list[float] = []
    refined_mask_coverage: list[float] = []

    for fp, mp in zip(frame_paths[:n], mask_paths[:n]):
        frame_bgr = cv2.imread(str(fp))
        if frame_bgr is None:
            frame_bgr = np.zeros((h_ref, w_ref, 3), np.uint8)

        # Load + refine binary mask
        binary_orig   = load_binary_mask(mp, h_ref, w_ref)
        binary_refined = refine_mask(binary_orig, dilate_px, bottom_extra_px, side_extra_px)

        # Statistics
        orig_mask_coverage.append(float((binary_orig > 0).sum()) / (h_ref * w_ref))
        refined_mask_coverage.append(float((binary_refined > 0).sum()) / (h_ref * w_ref))

        # Save binary mask
        cv2.imwrite(str(mf_dir / f"{fp.stem}.png"), binary_refined)

        # Save soft feather mask as 8-bit (0..255 float → uint8)
        soft = feather_mask(binary_refined, feather_px)
        cv2.imwrite(str(feat_dir / f"{fp.stem}.png"), (soft * 255).round().astype(np.uint8))

        # Overlay preview
        masked_in_frames.append(overlay_mask_on_frame(frame_bgr, binary_refined))

    write_video(out_dir / "masked_in.mp4", masked_in_frames, fps=args.fps)

    # Stats summary
    avg_orig    = float(np.mean(orig_mask_coverage))    * 100
    avg_refined = float(np.mean(refined_mask_coverage)) * 100
    expansion   = avg_refined - avg_orig

    print(f"\n  mask coverage: {avg_orig:.1f}% (original) → {avg_refined:.1f}% (refined)  "
          f"[+{expansion:.1f}%]")

    # Write manifest
    manifest = {
        "seq": args.seq,
        "variant": args.variant,
        "base_mask_dir": str(masks_dir),
        "mask_frames_dir": str(mf_dir),
        "feather_mask_frames_dir": str(feat_dir),
        "masked_in_mp4": str(out_dir / "masked_in.mp4"),
        "n_frames": n,
        "params": {
            "dilate_px": dilate_px,
            "bottom_extra_px": bottom_extra_px,
            "side_extra_px": side_extra_px,
            "feather_px": feather_px,
        },
        "stats": {
            "avg_mask_coverage_pct_original": round(avg_orig, 2),
            "avg_mask_coverage_pct_refined": round(avg_refined, 2),
            "coverage_expansion_pct": round(expansion, 2),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = out_dir / "mask_manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [manifest] → {manifest_path}")

    print(f"\n[OK] {n} refined masks → {out_dir}")
    print(f"\nNext: run MiniMax with these masks:")
    print(f"  conda run -p /data3/jli657/envs/minimax_env python3 \\")
    print(f"    .../inpainting/run_minimax_remover_gtmask.py \\")
    print(f"    --seq {args.seq} --version {args.variant} \\")
    print(f"    --mask_dir {mf_dir} \\")
    print(f"    --soft_blend --feather_mask_dir {feat_dir} --gpu 1")


if __name__ == "__main__":
    main()
