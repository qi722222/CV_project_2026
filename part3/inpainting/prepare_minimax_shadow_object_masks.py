"""
prepare_minimax_shadow_object_masks.py

Build stronger MiniMax masks for the two sequences where shadow residuals matter:
horsejump-low and car-shadow.

This script is intentionally different from prepare_minimax_masks.py:
- shadow_edge_v1 was mostly morphological edge expansion, so it fixed white halos
  but did not explicitly identify ground/contact shadows.
- shadow_object_v2 starts from the strongest Direction A/B object masks, unions the
  current MiniMax shadow_edge mask, then adds image-based dark contact-shadow
  candidates near the lower part of the object.

Outputs:
  /data3/jli657/project3/part3/outputs/minimax_masks/shadow_object_v2/<seq>/
    mask_frames/
    feather_mask_frames/
    shadow_candidate_frames/
    masked_in.mp4
    shadow_candidate.mp4
    mask_manifest.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


OUTPUTS_ROOT = Path("/data3/jli657/project3/part3/outputs/minimax_masks")
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DELIVERABLES = Path("/home/jli657/my_storage2_1T/project3/part3/part3_deliverables")


SEQ_DEFAULTS = {
    "tennis": {
        "frames_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
        "base_mask_dir": "/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/tennis",
        "extra_mask_dirs": [
            "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/tennis",
            "/data3/jli657/project3/part3/gdino_vlm/masks/stage1/tennis",
            "/data3/jli657/project3/part3/outputs/official_sam3_best/masks/tennis",
        ],
        "edge_dilate_px": 3,
        "shadow_dilate_px": 2,
        "feather_px": 8,
        "dark_percentile": 18.0,
        "bottom_extend_px": 24,
        "side_expand_frac": 0.14,
        "min_component_area": 40,
    },
    "bmx-trees": {
        "frames_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
        "base_mask_dir": "/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/bmx-trees",
        "extra_mask_dirs": [
            "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/bmx-trees",
            "/data3/jli657/project3/part3/gdino_vlm/masks/stage1/bmx-trees",
            "/data3/jli657/project3/part3/gdino_vlm/masks/sam3/stage1/bmx-trees",
        ],
        "edge_dilate_px": 3,
        "shadow_dilate_px": 3,
        "feather_px": 8,
        "dark_percentile": 22.0,
        "bottom_extend_px": 34,
        "side_expand_frac": 0.18,
        "min_component_area": 50,
    },
    "wild_video-1person": {
        "frames_dir": "/data3/jli657/project3/wild_frames/wild_video-1person",
        "base_mask_dir": "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/wild_video-1person_with_shadow",
        "extra_mask_dirs": [
            "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/wild_video-1person",
            "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_shadow_v2/wild_video-1person",
            "/data3/jli657/project3/part3/outputs/sam3_rebuild_v1/masks/wild/wild_video-1person_dilated",
        ],
        "edge_dilate_px": 2,
        "shadow_dilate_px": 2,
        "feather_px": 8,
        "dark_percentile": 18.0,
        "bottom_extend_px": 24,
        "side_expand_frac": 0.12,
        "min_component_area": 45,
    },
    "horsejump-low": {
        "frames_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/horsejump-low",
        "base_mask_dir": "/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/horsejump-low",
        "extra_mask_dirs": [
            "/data3/jli657/project3/part3/outputs/minimax_masks/shadow_edge_v1/horsejump-low/mask_frames",
            "/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/horsejump-low",
            "/data3/jli657/project3/part3/gdino_vlm/masks/sam3/stage1/horsejump-low",
        ],
        "edge_dilate_px": 3,
        "shadow_dilate_px": 3,
        "feather_px": 8,
        "dark_percentile": 22.0,
        "bottom_extend_px": 36,
        "side_expand_frac": 0.18,
        "min_component_area": 55,
    },
    "car-shadow": {
        "frames_dir": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/car-shadow",
        "base_mask_dir": "/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5/vggt4d/car-shadow",
        "extra_mask_dirs": [
            "/data3/jli657/project3/part3/outputs/minimax_masks/shadow_edge_v1/car-shadow/mask_frames",
            "/data3/jli657/project3/part3/outputs/direction_a/shadow_geom/car-shadow/scale_1.20",
            "/data3/jli657/project3/part3/gdino_vlm/masks/stage1/car-shadow",
        ],
        "edge_dilate_px": 3,
        "shadow_dilate_px": 4,
        "feather_px": 10,
        "dark_percentile": 24.0,
        "bottom_extend_px": 46,
        "side_expand_frac": 0.28,
        "min_component_area": 90,
    },
}


def load_sorted(d: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def load_binary_mask(mask_dir: Path, stem: str, shape: tuple[int, int]) -> np.ndarray:
    path = mask_dir / f"{stem}.png"
    h, w = shape
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return np.zeros((h, w), dtype=np.uint8)
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8) * 255)


def dilate(binary: np.ndarray, radius: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    if radius <= 0:
        return binary
    k = cv2.getStructuringElement(shape, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(binary, k)


def feather_mask(binary: np.ndarray, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return (binary > 0).astype(np.float32)
    ksize = max(3, feather_px * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    soft = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (ksize, ksize), feather_px / 3.0)
    return np.clip(soft, 0.0, 1.0)


def clean_components(binary: np.ndarray, support: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    out = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == i
        if np.logical_and(comp, support > 0).sum() == 0:
            continue
        out[comp] = 255
    return out


def contact_shadow_candidate(
    frame_bgr: np.ndarray,
    object_mask: np.ndarray,
    seq: str,
    dark_percentile: float,
    bottom_extend_px: int,
    side_expand_frac: float,
    shadow_dilate_px: int,
    min_component_area: int,
) -> np.ndarray:
    """Detect dark contact-shadow pixels near and below the lower object body.

    The candidate is deliberately local: it only considers a band around the lower
    half of the current object mask, then keeps dark components connected to a
    dilated lower-object support. This avoids selecting unrelated dark background.
    """
    h, w = object_mask.shape
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)

    pad_x = int(bw * side_expand_frac)
    roi_x1 = max(0, x1 - pad_x)
    roi_x2 = min(w, x2 + pad_x + 1)
    roi_y1 = max(0, y1 + int(0.42 * bh))
    roi_y2 = min(h, y2 + bottom_extend_px + 1)
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return np.zeros((h, w), dtype=np.uint8)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_sat = hsv[roi_y1:roi_y2, roi_x1:roi_x2, 1]
    thresh = np.percentile(roi_gray, dark_percentile)

    dark = (roi_gray <= thresh).astype(np.uint8) * 255
    # Keep colored object fragments out; real cast shadows are usually low/mid saturation.
    dark[roi_sat > 150] = 0

    candidate = np.zeros((h, w), dtype=np.uint8)
    candidate[roi_y1:roi_y2, roi_x1:roi_x2] = dark

    lower_object = np.zeros((h, w), dtype=np.uint8)
    if seq == "car-shadow":
        lower_object[max(0, y1 + int(0.45 * bh)):y2 + 1, x1:x2 + 1] = object_mask[
            max(0, y1 + int(0.45 * bh)):y2 + 1, x1:x2 + 1
        ]
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(21, int(0.95 * bw) | 1), max(15, int(0.55 * bh) | 1)),
        )
    else:
        # horsejump-low: focus on lower legs/contact region; avoid whole horse body.
        lower_object[max(0, y1 + int(0.58 * bh)):y2 + 1, x1:x2 + 1] = object_mask[
            max(0, y1 + int(0.58 * bh)):y2 + 1, x1:x2 + 1
        ]
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(19, int(0.55 * bw) | 1), max(11, int(0.28 * bh) | 1)),
        )

    support = cv2.dilate(lower_object, support_kernel)
    candidate = cv2.bitwise_and(candidate, support)

    if shadow_dilate_px > 0:
        candidate = dilate(candidate, shadow_dilate_px)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k_close)
    candidate = clean_components(candidate, support=dilate(lower_object, shadow_dilate_px + 3), min_area=min_component_area)

    return candidate


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    active = (mask > 0).astype(np.float32)[:, :, None]
    tint = np.zeros_like(frame_bgr, dtype=np.float32)
    tint[:, :] = color
    return (
        active * (0.42 * frame_bgr.astype(np.float32) + 0.58 * tint)
        + (1.0 - active) * frame_bgr.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)


def write_video(out_path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"[video] {out_path} ({len(frames)} frames)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare shadow-object MiniMax masks for horsejump-low/car-shadow")
    p.add_argument("--seq", required=True, choices=sorted(SEQ_DEFAULTS))
    p.add_argument("--variant", default="shadow_object_v2")
    p.add_argument("--frames_dir", default=None)
    p.add_argument("--base_mask_dir", default=None)
    p.add_argument("--extra_mask_dirs", nargs="*", default=None)
    p.add_argument("--edge_dilate_px", type=int, default=None)
    p.add_argument("--shadow_dilate_px", type=int, default=None)
    p.add_argument("--feather_px", type=int, default=None)
    p.add_argument("--dark_percentile", type=float, default=None)
    p.add_argument("--bottom_extend_px", type=int, default=None)
    p.add_argument("--side_expand_frac", type=float, default=None)
    p.add_argument("--min_component_area", type=int, default=None)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SEQ_DEFAULTS[args.seq]
    base_mask_dir = Path(args.base_mask_dir or cfg["base_mask_dir"])
    extra_mask_dirs = [Path(p) for p in (args.extra_mask_dirs if args.extra_mask_dirs is not None else cfg["extra_mask_dirs"])]
    edge_dilate_px = args.edge_dilate_px if args.edge_dilate_px is not None else cfg["edge_dilate_px"]
    shadow_dilate_px = args.shadow_dilate_px if args.shadow_dilate_px is not None else cfg["shadow_dilate_px"]
    feather_px = args.feather_px if args.feather_px is not None else cfg["feather_px"]
    dark_percentile = args.dark_percentile if args.dark_percentile is not None else cfg["dark_percentile"]
    bottom_extend_px = args.bottom_extend_px if args.bottom_extend_px is not None else cfg["bottom_extend_px"]
    side_expand_frac = args.side_expand_frac if args.side_expand_frac is not None else cfg["side_expand_frac"]
    min_component_area = args.min_component_area if args.min_component_area is not None else cfg["min_component_area"]

    frames_dir = Path(args.frames_dir or cfg.get("frames_dir", DAVIS_FRAMES / args.seq))
    out_dir = OUTPUTS_ROOT / args.variant / args.seq
    mask_dir = out_dir / "mask_frames"
    feather_dir = out_dir / "feather_mask_frames"
    shadow_dir = out_dir / "shadow_candidate_frames"

    if out_dir.exists() and not args.overwrite and any(mask_dir.glob("*.png")):
        print(f"[skip] {out_dir} already exists. Use --overwrite to regenerate.")
        return

    for d in [base_mask_dir, *extra_mask_dirs, frames_dir]:
        if not d.exists():
            raise FileNotFoundError(d)

    mask_dir.mkdir(parents=True, exist_ok=True)
    feather_dir.mkdir(parents=True, exist_ok=True)
    shadow_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = load_sorted(frames_dir, {".jpg", ".jpeg", ".png"})
    masked_previews: list[np.ndarray] = []
    shadow_previews: list[np.ndarray] = []
    base_cov, extra_cov, shadow_cov, final_cov = [], [], [], []

    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is None:
            raise FileNotFoundError(fp)
        h, w = frame.shape[:2]

        base = load_binary_mask(base_mask_dir, fp.stem, (h, w))
        union = base.copy()
        for d in extra_mask_dirs:
            union = np.maximum(union, load_binary_mask(d, fp.stem, (h, w)))

        shadow = contact_shadow_candidate(
            frame,
            union,
            args.seq,
            dark_percentile=dark_percentile,
            bottom_extend_px=bottom_extend_px,
            side_expand_frac=side_expand_frac,
            shadow_dilate_px=shadow_dilate_px,
            min_component_area=min_component_area,
        )

        refined = np.maximum(union, shadow)
        refined = dilate(refined, edge_dilate_px)
        soft = feather_mask(refined, feather_px)

        cv2.imwrite(str(mask_dir / f"{fp.stem}.png"), refined)
        cv2.imwrite(str(feather_dir / f"{fp.stem}.png"), np.round(soft * 255).astype(np.uint8))
        cv2.imwrite(str(shadow_dir / f"{fp.stem}.png"), shadow)

        base_cov.append((base > 0).mean() * 100)
        extra_cov.append((union > 0).mean() * 100)
        shadow_cov.append((shadow > 0).mean() * 100)
        final_cov.append((refined > 0).mean() * 100)
        masked_previews.append(overlay_mask(frame, refined, (0, 255, 0)))
        shadow_previews.append(overlay_mask(frame, shadow, (0, 0, 255)))

    write_video(out_dir / "masked_in.mp4", masked_previews, args.fps)
    write_video(out_dir / "shadow_candidate.mp4", shadow_previews, args.fps)

    stats = {
        "avg_base_mask_coverage_pct": round(float(np.mean(base_cov)), 2),
        "avg_union_before_shadow_pct": round(float(np.mean(extra_cov)), 2),
        "avg_shadow_candidate_pct": round(float(np.mean(shadow_cov)), 2),
        "avg_final_mask_coverage_pct": round(float(np.mean(final_cov)), 2),
    }
    manifest = {
        "seq": args.seq,
        "variant": args.variant,
        "base_mask_dir": str(base_mask_dir),
        "frames_dir": str(frames_dir),
        "extra_mask_dirs": [str(p) for p in extra_mask_dirs],
        "mask_frames_dir": str(mask_dir),
        "feather_mask_frames_dir": str(feather_dir),
        "shadow_candidate_frames_dir": str(shadow_dir),
        "masked_in_mp4": str(out_dir / "masked_in.mp4"),
        "shadow_candidate_mp4": str(out_dir / "shadow_candidate.mp4"),
        "n_frames": len(frame_paths),
        "params": {
            "edge_dilate_px": edge_dilate_px,
            "shadow_dilate_px": shadow_dilate_px,
            "feather_px": feather_px,
            "dark_percentile": dark_percentile,
            "bottom_extend_px": bottom_extend_px,
            "side_expand_frac": side_expand_frac,
            "min_component_area": min_component_area,
        },
        "stats": stats,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "mask_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {args.seq} {args.variant}: {len(frame_paths)} frames")
    print(f"  base:  {base_mask_dir}")
    print(f"  extra: {[str(p) for p in extra_mask_dirs]}")
    print(f"  stats: {stats}")
    print(f"  output: {out_dir}")
    print("Next:")
    print("  conda run -p /data3/jli657/envs/minimax_env python3 \\")
    print("    /home/jli657/my_storage2_1T/project3/part3/inpainting/run_minimax_remover_gtmask.py \\")
    print(f"    --seq {args.seq} --version {args.variant} \\")
    print(f"    --mask_dir {mask_dir} \\")
    print(f"    --soft_blend --feather_mask_dir {feather_dir} --gpu 1")


if __name__ == "__main__":
    main()
