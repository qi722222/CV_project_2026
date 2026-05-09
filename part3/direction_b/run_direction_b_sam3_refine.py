"""
run_direction_b_sam3_refine.py
Direction B-5: VGGT4D/Pi3 rough masks -> SAM3 spatial prompt refinement

Core idea: VGGT4D/Pi3 produces rough but complete dynamic masks (high recall).
Use these masks as spatial prompts for SAM3 to achieve pixel-perfect refinement.

Two prompt modes:
  1. bbox: Extract bounding box from rough mask, use as SAM3 box prompt
  2. centroid: Extract centroid from rough mask, use as SAM3 point prompt

Expected improvement: VGGT4D tennis JM 0.757 -> 0.9+ after SAM3 refinement
(Since VGGT4D correctly finds the tennis player with JR=1.0, SAM3 just needs to
 clean up the boundaries)

Run environment: sam3_official_env (Python 3.10)
  PYTHONPATH=/data3/jli657/sam3:/data3/jli657/VGGT4D \
  conda run -n sam3_official_env python3 part3/run_direction_b_sam3_refine.py \
    --source_method vggt4d \
    --sequences tennis blackswan horsejump-low koala bmx-trees car-shadow
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

SAM3_REPO = "/data3/jli657/sam3"
SAM3_CKPT = "/data3/jli657/project3/weights/sam3/sam3.pt"
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")

SEQUENCE_VIDEO_ROOTS: Dict[str, Path] = {
    "tennis":           DAVIS_FRAMES / "tennis",
    "bmx-trees":        DAVIS_FRAMES / "bmx-trees",
    "blackswan":        DAVIS_FRAMES / "blackswan",
    "car-shadow":       DAVIS_FRAMES / "car-shadow",
    "horsejump-low":    DAVIS_FRAMES / "horsejump-low",
    "koala":            DAVIS_FRAMES / "koala",
}

# Source mask directories from Direction B
MASK_DIRS: Dict[str, Dict[str, Path]] = {
    "vggt4d": {
        seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt/{seq}")
        for seq in SEQUENCE_VIDEO_ROOTS
    },
    "pi3": {
        seq: Path(f"/data3/jli657/project3/part3/outputs/direction_b/pi3_transplant/{seq}")
        for seq in SEQUENCE_VIDEO_ROOTS
    },
}


# ---------------------------------------------------------------------------
# Mask analysis helpers
# ---------------------------------------------------------------------------

def extract_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Extract tight bounding box from binary mask. Returns (x1, y1, x2, y2) or None."""
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        return None
    y1, y2 = int(nz[0].min()), int(nz[0].max())
    x1, x2 = int(nz[1].min()), int(nz[1].max())
    # Add small padding
    h, w = mask.shape
    pad = 5
    return (max(0, x1-pad), max(0, y1-pad), min(w-1, x2+pad), min(h-1, y2+pad))


def _sample_mask_points(mask: np.ndarray, n: int, frame_w: int, frame_h: int) -> list:
    """Sample n points from the mask foreground, normalized to [0,1]."""
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        return []
    # Use evenly spaced indices across non-zero pixels
    indices = np.linspace(0, len(nz[0]) - 1, n, dtype=int)
    points = [[float(nz[1][i]) / frame_w, float(nz[0][i]) / frame_h] for i in indices]
    return points


def extract_centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Extract centroid of the largest connected component."""
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        return None
    # Find largest connected component
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num_labels <= 1:
        return int(np.mean(nz[1])), int(np.mean(nz[0]))
    largest = np.argmax([np.sum(labels == i) for i in range(1, num_labels)]) + 1
    comp_nz = np.nonzero(labels == largest)
    return int(np.mean(comp_nz[1])), int(np.mean(comp_nz[0]))


def load_rough_masks(mask_dir: Path, frame_paths: List[Path]) -> Dict[str, np.ndarray]:
    """Load rough (VGGT4D/Pi3) masks, resizing to match frame size."""
    masks = {}
    if not mask_dir.exists():
        return masks
    for fp in frame_paths:
        mp = mask_dir / f"{fp.stem}.png"
        if not mp.exists():
            continue
        m = np.array(Image.open(mp).convert("L"))
        # Get target size from frame
        frame = np.array(Image.open(fp))
        h, w = frame.shape[:2]
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks[fp.stem] = (m > 127).astype(np.uint8) * 255
    return masks


# ---------------------------------------------------------------------------
# SAM3 spatial prompt prediction
# ---------------------------------------------------------------------------

def run_sam3_with_spatial_prompts(
    predictor,
    seq_name: str,
    video_dir: Path,
    rough_masks: Dict[str, np.ndarray],
    output_dir: Path,
    prompt_mode: str = "bbox",
) -> Dict:
    """
    Run SAM3 with spatial prompts derived from VGGT4D/Pi3 rough masks.
    Uses SAM3's handle_request API (matching the existing SAM3 video pipeline).

    Prompt modes:
      bbox: Extract bounding box, use as SAM3 box prompt (normalized [0,1])
      text: Fall back to text prompt for sequences with poor rough masks
    """
    # Load all frames
    frame_paths = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.stem)

    if not frame_paths:
        return {"status": "error", "error": "no frames"}

    print(f"  [{seq_name}] SAM3 spatial prompt refinement ({len(frame_paths)} frames)")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get frame dimensions
    first_img = np.array(Image.open(frame_paths[0]))
    frame_h, frame_w = first_img.shape[:2]

    # Find first valid rough mask
    first_valid_frame = None
    first_valid_idx = None
    for i, fp in enumerate(frame_paths):
        if fp.stem in rough_masks and rough_masks[fp.stem].sum() > 0:
            first_valid_frame = fp
            first_valid_idx = i
            break

    if first_valid_frame is None:
        print(f"  [{seq_name}] No valid rough masks, skipping")
        return {"status": "error", "error": "no valid rough masks"}

    rough_mask_first = rough_masks[first_valid_frame.stem]

    # Start SAM3 session (init_state creates fresh state - no need for reset_session)
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(video_dir),
    })
    session_id = response["session_id"]

    # Build prompt from rough mask using POINT mode (triggers SAM3 tracker path,
    # avoids hotstart_delay=15 issue in the detection/visual-reference path).
    # SAM3's tracker path (add_tracker_new_points) requires obj_id and only points.
    centroid = extract_centroid_from_mask(rough_mask_first)
    if centroid is None:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return {"status": "error", "error": "no centroid found"}
    cx, cy = centroid
    cx_norm, cy_norm = cx / frame_w, cy / frame_h

    # Always include centroid as the primary click; add interior sample points
    # for robustness. Centroid is the most reliable anchor (mask center-of-mass).
    interior_points = _sample_mask_points(rough_mask_first, n=4, frame_w=frame_w, frame_h=frame_h)
    all_points = [[cx_norm, cy_norm]] + interior_points
    all_labels = [1] * len(all_points)

    print(f"  [{seq_name}] point prompt at frame {first_valid_frame.stem}: {len(all_points)} pts (centroid=({cx_norm:.3f},{cy_norm:.3f}) + {len(interior_points)} interior)")
    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": first_valid_idx,
        "points": all_points,
        "point_labels": all_labels,
        "obj_id": 1,
        "rel_coordinates": True,
    })

    # Propagate through video (forward only from first valid frame).
    # Must pass start_frame_index so _get_processing_order skips the
    # previous_stages_out nil-check (tracker path doesn't set previous_stages_out).
    outputs_per_frame: Dict[int, dict] = {}
    for prop_response in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "forward",
        "start_frame_index": first_valid_idx,
    }):
        fidx = prop_response["frame_index"]
        outputs_per_frame[fidx] = prop_response["outputs"]

    print(f"  [{seq_name}] propagated {len(outputs_per_frame)} frames")

    # Close session
    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    # Save masks
    saved = 0
    for i, fp in enumerate(frame_paths):
        if i in outputs_per_frame:
            mask = _extract_mask(outputs_per_frame[i], frame_h, frame_w)
        else:
            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        Image.fromarray(mask).save(str(output_dir / f"{fp.stem}.png"))
        saved += 1

    print(f"  [{seq_name}] Saved {saved} SAM3-refined masks")
    return {
        "status": "success",
        "sequence": seq_name,
        "n_frames": len(frame_paths),
        "n_saved": saved,
        "prompt_mode": prompt_mode,
        "prompt_frame": first_valid_frame.stem,
    }


def _extract_mask(outputs: dict, frame_h: int, frame_w: int) -> np.ndarray:
    """Extract binary uint8 mask from SAM3 propagation outputs."""
    combined = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if not outputs:
        return combined
    if "out_binary_masks" in outputs:
        binary_masks = outputs["out_binary_masks"]
        if binary_masks is not None and binary_masks.size > 0:
            for i in range(binary_masks.shape[0]):
                m = binary_masks[i]
                if m.shape != (frame_h, frame_w):
                    m = cv2.resize(m.astype(np.float32), (frame_w, frame_h),
                                   interpolation=cv2.INTER_LINEAR) > 0.5
                combined = np.maximum(combined, m.astype(np.uint8) * 255)
    return combined


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_jm_jr_f(pred_dir: Path, gt_dir: Path) -> Dict[str, float]:
    if not gt_dir.exists():
        return {"JM": -1.0, "JR": -1.0, "F": -1.0}
    ious, recalls, f_scores = [], [], []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt_arr = np.array(Image.open(gt_path).convert("L")) > 0
        pred_img = Image.open(pred_path).convert("L")
        if pred_img.size != (gt_arr.shape[1], gt_arr.shape[0]):
            pred_img = pred_img.resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
        pred_arr = np.array(pred_img) > 127
        inter = np.logical_and(gt_arr, pred_arr).sum()
        union = np.logical_or(gt_arr, pred_arr).sum()
        iou = inter / (union + 1e-8)
        ious.append(iou)
        recalls.append(float(iou >= 0.5))
        gt_cont = cv2.Canny(gt_arr.astype(np.uint8) * 255, 50, 150) > 0
        pred_cont = cv2.Canny(pred_arr.astype(np.uint8) * 255, 50, 150) > 0
        tp = np.logical_and(gt_cont, pred_cont).sum()
        prec = tp / (pred_cont.sum() + 1e-8)
        rec = tp / (gt_cont.sum() + 1e-8)
        f_scores.append(2 * prec * rec / (prec + rec + 1e-8))
    if not ious:
        return {"JM": 0.0, "JR": 0.0, "F": 0.0}
    return {"JM": float(np.mean(ious)), "JR": float(np.mean(recalls)), "F": float(np.mean(f_scores))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Direction B-5: VGGT4D/Pi3 -> SAM3 spatial prompt refinement")
    p.add_argument("--sequences", nargs="+",
                   default=["tennis", "blackswan", "horsejump-low", "koala", "bmx-trees", "car-shadow"])
    p.add_argument("--source_method", default="vggt4d",
                   choices=["vggt4d", "pi3"],
                   help="Which rough masks to use as prompts")
    p.add_argument("--prompt_mode", default="bbox",
                   choices=["bbox", "centroid"])
    p.add_argument("--output_root",
                   default="/data3/jli657/project3/part3/outputs/direction_b/sam3_refined")
    p.add_argument("--out_csv",
                   default="/home/jli657/my_storage2_1T/project3/eval/direction_b_sam3_refined_v5_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root) / args.source_method
    mask_dirs = MASK_DIRS[args.source_method]

    # Load SAM3 predictor once
    if SAM3_REPO not in sys.path:
        sys.path.insert(0, SAM3_REPO)
    from sam3.model_builder import build_sam3_video_predictor
    print("Loading SAM3 predictor...")
    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CKPT,
        strict_state_dict_loading=False,
    )
    print("SAM3 loaded.")

    rows = []
    for seq in args.sequences:
        video_dir = SEQUENCE_VIDEO_ROOTS.get(seq)
        if video_dir is None or not video_dir.exists():
            print(f"[skip] {seq}: video dir not found")
            continue

        mask_dir = mask_dirs[seq]
        refined_dir = output_root / seq
        run_meta: dict = {"sequence": seq, "source": args.source_method}

        # Load rough masks
        frame_paths = sorted(
            [p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}],
            key=lambda p: p.stem)
        rough_masks = load_rough_masks(mask_dir, frame_paths)

        if not rough_masks:
            print(f"  [{seq}] No rough masks found in {mask_dir}, skipping")
            run_meta["status"] = "no_rough_masks"
            rows.append(run_meta)
            continue

        print(f"  [{seq}] Loaded {len(rough_masks)} rough masks from {mask_dir}")

        # Run SAM3 spatial prompt refinement
        try:
            result = run_sam3_with_spatial_prompts(
                predictor, seq, video_dir, rough_masks, refined_dir,
                prompt_mode=args.prompt_mode)
            run_meta.update(result)
        except Exception as e:
            import traceback; traceback.print_exc()
            run_meta["status"] = f"error: {e}"

        # Evaluate refined masks
        gt_dir = DAVIS_GT / seq
        if refined_dir.exists() and gt_dir.exists():
            metrics = compute_jm_jr_f(refined_dir, gt_dir)
            run_meta.update(metrics)
            # Also eval raw rough masks
            rough_metrics = compute_jm_jr_f(mask_dir, gt_dir)
            run_meta["rough_JM"] = rough_metrics.get("JM", -1.0)
            print(f"  [{seq}] Refined JM={metrics['JM']:.4f} (rough={run_meta['rough_JM']:.4f})")
        else:
            run_meta.update({"JM": -1.0, "JR": -1.0, "F": -1.0, "rough_JM": -1.0})

        rows.append(run_meta)

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "source", "status", "n_frames", "n_saved", "prompt_mode",
              "prompt_frame", "JM", "JR", "F", "rough_JM"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {out_csv}")

    print("\n=== Direction B-5 SAM3 Refined Results ===")
    print(f"{'Seq':<20} {'Refined JM':>12} {'Rough JM':>10}")
    for r in rows:
        jm = r.get("JM", -1.0)
        rjm = r.get("rough_JM", -1.0)
        print(f"  {r['sequence']:<18} {jm:>12.4f} {rjm:>10.4f}")


if __name__ == "__main__":
    main()
