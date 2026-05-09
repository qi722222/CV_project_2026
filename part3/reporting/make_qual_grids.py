"""
make_qual_grids.py — Qualitative comparison grids

For each sequence, creates a side-by-side comparison grid:
  Row 1: Original frame | Dir-A mask | B-5 VGGT4D+SAM3 mask | A+B Best mask
  Row 2: Original frame | Dir-A inpainted | A+B Best inpainted

Output:
  /home/jli657/my_storage2_1T/project3/eval/qual_grids/<seq>_grid.jpg

Usage:
  conda run -n controlnet_env python3 part3/make_qual_grids.py
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List

# Paths
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT     = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
RESULTS_OLD  = Path("/data3/jli657/project3/part3/results")
RESULTS_V2   = Path("/data3/jli657/project3/part3/results_v2")
VGGT4D_MASKS = Path("/data3/jli657/project3/part3/outputs/direction_b/vggt4d_masks")
SAM3_REFINED = Path("/data3/jli657/project3/part3/outputs/direction_b/sam3_refined_v5")
SAM3_MASKS   = Path("/data3/jli657/project3/part3/outputs/sam3_rebuild_v1/masks/davis5")
GDINO_MASKS  = Path("/data3/jli657/project3/part3/gdino_vlm/masks/stage1")
OUT_DIR      = Path("/home/jli657/my_storage2_1T/project3/eval/qual_grids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCES = ["tennis", "horsejump-low", "car-shadow", "blackswan", "bmx-trees", "koala"]

# Best mask source per sequence (matching run_a3_best_pipeline.py BEST_MASK_SOURCES)
BEST_MASK_SOURCES = {
    "tennis":        (SAM3_MASKS / "tennis",               "A-SAM3"),
    "horsejump-low": (SAM3_REFINED / "vggt4d/horsejump-low", "B5-VGGT4D+SAM3"),
    "car-shadow":    (SAM3_REFINED / "vggt4d/car-shadow",    "B5-VGGT4D+SAM3"),
    "blackswan":     (SAM3_REFINED / "vggt4d/blackswan",     "B5-VGGT4D+SAM3"),
    "bmx-trees":     (GDINO_MASKS / "bmx-trees",             "A-GDINO+SAM2"),
    "koala":         (SAM3_MASKS / "koala",                  "A-SAM3"),
}

THUMB_W, THUMB_H = 320, 180
BORDER = 4
FONT  = cv2.FONT_HERSHEY_SIMPLEX
FSCALE = 0.45
FTHICK = 1


def thumb(img: np.ndarray, w: int = THUMB_W, h: int = THUMB_H) -> np.ndarray:
    return cv2.resize(img, (w, h))


def load_sorted_frames(d: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def load_video_mid_frame(mp4: Path, frac: float = 0.4) -> Optional[np.ndarray]:
    """Load a representative frame from an MP4 (at 40% of video)."""
    if not mp4.exists():
        return None
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = max(0, int(total * frac))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def load_mask_frame(mask_dir: Path, idx: int) -> Optional[np.ndarray]:
    """Load mask PNG at given frame index (0-based)."""
    if not mask_dir.exists():
        return None
    paths = load_sorted_frames(mask_dir)
    if idx >= len(paths):
        idx = len(paths) // 2
    m = cv2.imread(str(paths[idx]), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m


def mask_to_overlay(orig: np.ndarray, mask_gray: np.ndarray,
                    color=(0, 0, 200), alpha=0.45) -> np.ndarray:
    """Overlay mask as semi-transparent color on original frame."""
    vis = orig.copy()
    m = (mask_gray > 127)
    overlay = vis.copy()
    overlay[m] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def labeled(img: np.ndarray, label: str, bg=(30, 30, 30)) -> np.ndarray:
    h, w = img.shape[:2]
    bar  = np.full((22, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, label, (4, 15), FONT, FSCALE, (220, 220, 220), FTHICK, cv2.LINE_AA)
    return np.vstack([bar, img])


def hstack_border(imgs: List[np.ndarray], gap: int = BORDER) -> np.ndarray:
    spacer = np.full((imgs[0].shape[0], gap, 3), 180, dtype=np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.hstack([out, spacer, im])
    return out


def vstack_border(rows: List[np.ndarray], gap: int = BORDER * 2) -> np.ndarray:
    spacer = np.full((gap, rows[0].shape[1], 3), 255, dtype=np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, spacer, r])
    return out


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((30, img.shape[1], 3), (50, 50, 50), dtype=np.uint8)
    cv2.putText(bar, title, (8, 21), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_grid(seq: str):
    print(f"[{seq}] building grid...")

    # --- pick representative frame index ---
    orig_dir   = DAVIS_FRAMES / seq
    orig_paths = load_sorted_frames(orig_dir)
    n_frames   = len(orig_paths)
    pick_idx   = max(0, int(n_frames * 0.4))

    # Load original frame
    orig = cv2.imread(str(orig_paths[pick_idx]))
    if orig is None:
        print(f"  [WARN] cannot load original frame {orig_paths[pick_idx]}")
        return

    # -------- Row 1: masks comparison --------
    # (a) GT mask
    gt_dir    = DAVIS_GT / seq
    gt_paths  = sorted(gt_dir.glob("*.png"), key=lambda p: p.stem) if gt_dir.exists() else []
    gt_mask   = None
    if pick_idx < len(gt_paths):
        gt_mask = cv2.imread(str(gt_paths[pick_idx]), cv2.IMREAD_GRAYSCALE)

    # (b) Dir-A SAM3 mask
    dir_a_mask = load_mask_frame(SAM3_MASKS / seq if (SAM3_MASKS / seq).exists() else GDINO_MASKS / seq, pick_idx)

    # (c) VGGT4D raw mask
    vggt_mask  = load_mask_frame(VGGT4D_MASKS / seq, pick_idx)

    # (d) B-5 VGGT4D+SAM3 refined mask
    b5_mask    = load_mask_frame(SAM3_REFINED / "vggt4d" / seq, pick_idx)

    # (e) A+B Best mask
    best_path, best_label = BEST_MASK_SOURCES.get(seq, (None, "unknown"))
    best_mask  = load_mask_frame(best_path, pick_idx) if best_path else None

    def mk_panel(frame, mask_gray, label):
        t = thumb(frame)
        if mask_gray is not None:
            mask_rs = cv2.resize(mask_gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_NEAREST)
            t = mask_to_overlay(t, mask_rs)
        return labeled(t, label)

    row1_cells = [
        mk_panel(orig, None,     "Original"),
        mk_panel(orig, gt_mask,  "GT Mask"),
        mk_panel(orig, dir_a_mask, "Dir-A SAM3"),
        mk_panel(orig, vggt_mask,  "Dir-B VGGT4D"),
        mk_panel(orig, b5_mask,    "Dir-B VGGT4D+SAM3"),
        mk_panel(orig, best_mask,  f"Best ({best_label})"),
    ]
    row1 = hstack_border(row1_cells)

    # -------- Row 2: inpainting comparison --------
    # Get inpainted frame at same pick_idx
    def get_inpaint_frame(mp4: Path) -> Optional[np.ndarray]:
        if not mp4.exists():
            return None
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pick_idx)
        ret, f = cap.read()
        cap.release()
        return f if ret else None

    dir_a_mp4 = RESULTS_OLD / seq / "direction_a" / "sam3_propainter" / "inpaint_out.mp4"
    ab_mp4    = RESULTS_V2  / seq / "a_plus_b_best" / "propainter" / seq / "inpaint_out.mp4"

    dir_a_inp = get_inpaint_frame(dir_a_mp4)
    ab_inp    = get_inpaint_frame(ab_mp4)

    def inp_panel(frame, label):
        if frame is None:
            placeholder = np.full((THUMB_H, THUMB_W, 3), 80, dtype=np.uint8)
            cv2.putText(placeholder, "N/A", (THUMB_W//2 - 20, THUMB_H//2),
                        FONT, 0.7, (200, 200, 200), 2)
            return labeled(placeholder, label)
        return labeled(thumb(frame), label)

    row2_cells = [
        inp_panel(orig,      "Original"),
        inp_panel(None,      ""),        # spacer alignment
        inp_panel(dir_a_inp, "Dir-A SAM3 Inpainted"),
        inp_panel(None,      ""),
        inp_panel(None,      ""),
        inp_panel(ab_inp,    "A+B Best Inpainted"),
    ]
    row2 = hstack_border(row2_cells)

    # Combine rows
    grid = vstack_border([row1, row2])
    grid = add_title(grid, f"Qualitative Comparison — {seq}")

    out_path = OUT_DIR / f"{seq}_grid.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    for seq in SEQUENCES:
        make_grid(seq)
    print("\nAll grids saved to", OUT_DIR)
