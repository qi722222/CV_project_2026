"""
run_direction_b_vggt4d.py
Direction B: VGGT4D Zero-shot Dynamic Mask Generation

Runs VGGT4D (Gram Similarity attention mining) on DAVIS sequences to generate
zero-shot dynamic object masks without any text prompts. Three pipeline stages:
  Stage 1: VGGT forward pass -> QK attention -> Gram Similarity -> rough dynamic mask
  Stage 2: Refine camera extrinsics using dynamic mask
  Stage 3: Projection gradient mask refinement

Then evaluates JM/JR/F vs DAVIS GT masks and outputs comparison with SAM3 baseline.

Usage:
  PYTHONPATH=/data3/jli657/VGGT4D \
  python part3/run_direction_b_vggt4d.py \
    --sequences tennis blackswan horsejump-low bmx-trees car-shadow koala \
    --output_root /data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt

Requirements: vggt4d_env (Python 3.12, PyTorch 2.7.1)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VGGT4D_REPO = "/data3/jli657/VGGT4D"
VGGT4D_CKPT = "/data3/jli657/VGGT4D/ckpts/model_tracker_fixed_e20.pt"
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
WILD_FRAMES = Path("/data3/jli657/project3/wild_frames/wild_video-1person")

SEQUENCE_VIDEO_ROOTS: Dict[str, Path] = {
    "tennis": DAVIS_FRAMES / "tennis",
    "bmx-trees": DAVIS_FRAMES / "bmx-trees",
    "blackswan": DAVIS_FRAMES / "blackswan",
    "car-shadow": DAVIS_FRAMES / "car-shadow",
    "horsejump-low": DAVIS_FRAMES / "horsejump-low",
    "koala": DAVIS_FRAMES / "koala",
    "wild_video-1person": WILD_FRAMES,
}

# Max frames to process per sequence (VGGT4D holds all in GPU memory)
MAX_FRAMES_PER_SEQ = 60  # safe limit for A6000 48GB; wild_video has 284 frames

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sorted_frames(video_dir: Path, max_frames: Optional[int] = None) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in exts],
                   key=lambda p: p.stem)
    if max_frames is not None:
        # Uniform subsample for long sequences
        if len(paths) > max_frames:
            idxs = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
            paths = [paths[i] for i in idxs]
    return paths


def compute_jm_jr_f(pred_dir: Path, gt_dir: Path) -> Dict[str, float]:
    """Compute JM (mean IoU), JR (recall @ tau=0.5), and F (contour F-measure)."""
    if not gt_dir.exists():
        return {"JM": -1.0, "JR": -1.0, "F": -1.0}

    ious, recalls, f_scores = [], [], []

    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt_arr = np.array(Image.open(gt_path).convert("L")) > 0
        pred_img = Image.open(pred_path).convert("L")
        # Resize prediction to match GT if needed
        if pred_img.size != (gt_arr.shape[1], gt_arr.shape[0]):
            pred_img = pred_img.resize((gt_arr.shape[1], gt_arr.shape[0]),
                                        Image.NEAREST)
        pred_arr = np.array(pred_img) > 127

        inter = np.logical_and(gt_arr, pred_arr).sum()
        union = np.logical_or(gt_arr, pred_arr).sum()
        iou = inter / (union + 1e-8)
        ious.append(iou)
        recalls.append(float(iou >= 0.5))

        # F-measure via contour boundary
        gt_cont = cv2.Canny(gt_arr.astype(np.uint8) * 255, 50, 150) > 0
        pred_cont = cv2.Canny(pred_arr.astype(np.uint8) * 255, 50, 150) > 0
        tp = np.logical_and(gt_cont, pred_cont).sum()
        prec = tp / (pred_cont.sum() + 1e-8)
        rec = tp / (gt_cont.sum() + 1e-8)
        f = 2 * prec * rec / (prec + rec + 1e-8)
        f_scores.append(f)

    if not ious:
        return {"JM": 0.0, "JR": 0.0, "F": 0.0}
    return {
        "JM": float(np.mean(ious)),
        "JR": float(np.mean(recalls)),
        "F": float(np.mean(f_scores)),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_vggt4d_on_sequence(
    predictor,
    seq_name: str,
    video_dir: Path,
    output_dir: Path,
    max_frames: int = MAX_FRAMES_PER_SEQ,
) -> Dict:
    """Run VGGT4D on a single sequence and save dynamic masks."""
    from vggt4d.masks.dynamic_mask import (
        adaptive_multiotsu_variance,
        cluster_attention_maps,
        extract_dyn_map,
    )
    from vggt4d.masks.refine_dyn_mask import RefineDynMask
    from vggt4d.utils.model_utils import inference, organize_qk_dict
    from vggt.utils.load_fn import load_and_preprocess_images

    device = next(predictor.parameters()).device
    frame_paths = load_sorted_frames(video_dir, max_frames=max_frames)
    if not frame_paths:
        return {"status": "error", "error": "no frames"}

    print(f"  [{seq_name}] {len(frame_paths)} frames -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_and_preprocess_images(
        [str(p) for p in frame_paths]).to(device)
    n_img, _, h_img, w_img = images.shape

    # ---- Stage 1: predict depth + dynamic map ----------------------------
    print(f"  [{seq_name}] Stage 1: attention Gram Similarity mining...")
    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(predictor, images)
    del agg_tokens_list
    qk_dict = organize_qk_dict(qk_dict, images.shape[0])

    dyn_maps = extract_dyn_map(qk_dict, images)
    h_tok, w_tok = h_img // 14, w_img // 14
    feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps, n_clusters=16)  # reduced for speed
    del enc_feat, feat_map
    torch.cuda.empty_cache()

    upsampled_map = F.interpolate(
        rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
        size=(h_img, w_img), mode="bilinear", align_corners=False)
    upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres
    print(f"  [{seq_name}] Stage 1 done. mask coverage avg: "
          f"{dyn_masks.float().mean().item():.3f}")

    # ---- Stage 2: refine extrinsics with dynamic mask --------------------
    print(f"  [{seq_name}] Stage 2: refine camera extrinsics...")
    torch.cuda.empty_cache()
    predictions2, _, _, _ = inference(predictor, images, dyn_masks.to(device))

    pred_intrinsic = predictions1["intrinsic"]
    pred_cam2world = predictions2["cam2world"]
    pred_depths = predictions1["depth"]

    # ---- Stage 3: projection gradient mask refinement -------------------
    print(f"  [{seq_name}] Stage 3: projection gradient refinement...")
    torch.cuda.empty_cache()
    refiner = RefineDynMask(
        images,
        torch.tensor(pred_depths).to(device),
        dyn_masks.to(device),
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).to(device),
        device,
    )
    refined_masks = refiner.refine_masks()
    del refiner
    torch.cuda.empty_cache()
    print(f"  [{seq_name}] Stage 3 done. refined coverage avg: "
          f"{refined_masks.float().mean().item():.3f}")

    # ---- Save masks (using original frame names for GT matching) ---------
    for i, fp in enumerate(frame_paths):
        mask_np = refined_masks[i].cpu().numpy().astype(np.uint8) * 255
        out_path = output_dir / f"{fp.stem}.png"
        Image.fromarray(mask_np).save(str(out_path))

    # Also save rough masks for comparison
    rough_dir = output_dir.parent / (output_dir.name + "_rough")
    rough_dir.mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(frame_paths):
        mask_np = dyn_masks[i].cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_np).save(str(rough_dir / f"{fp.stem}.png"))

    result = {
        "status": "success",
        "sequence": seq_name,
        "n_frames": len(frame_paths),
        "output_dir": str(output_dir),
        "rough_dir": str(rough_dir),
        "mask_coverage_stage1": float(dyn_masks.float().mean().item()),
        "mask_coverage_stage3": float(refined_masks.float().mean().item()),
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direction B: VGGT4D dynamic mask generation")
    p.add_argument("--sequences", nargs="+",
                   default=["tennis", "blackswan", "horsejump-low", "koala",
                            "bmx-trees", "car-shadow"],
                   help="Sequences to process")
    p.add_argument("--output_root",
                   default="/data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt",
                   help="Output root directory")
    p.add_argument("--ckpt", default=VGGT4D_CKPT,
                   help="VGGT4D checkpoint path")
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_SEQ,
                   help="Max frames per sequence (memory limit)")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip inference, only evaluate existing masks")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_csv",
                   default="/home/jli657/my_storage2_1T/project3/eval/direction_b_vggt4d_results.csv",
                   help="Output CSV path for JM/JR/F results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    if VGGT4D_REPO not in sys.path:
        sys.path.insert(0, VGGT4D_REPO)

    output_root = Path(args.output_root)
    gt_root = DAVIS_GT

    # Load model
    if not args.eval_only:
        from vggt4d.models.vggt4d import VGGTFor4D
        print("Loading VGGT4D model...")
        model = VGGTFor4D()
        state = torch.load(args.ckpt, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        model = model.to(device)
        print("VGGT4D loaded.")
    else:
        model = None

    rows = []
    for seq in args.sequences:
        video_dir = SEQUENCE_VIDEO_ROOTS.get(seq)
        if video_dir is None or not video_dir.exists():
            print(f"[skip] {seq}: video dir not found")
            continue

        mask_dir = output_root / seq
        run_meta: Dict = {"sequence": seq}

        if not args.eval_only:
            try:
                result = run_vggt4d_on_sequence(
                    model, seq, video_dir, mask_dir,
                    max_frames=args.max_frames)
                run_meta.update(result)
                # Save run meta
                with open(mask_dir / "run_meta.json", "w") as f:
                    json.dump(run_meta, f, indent=2)
                print(f"  [{seq}] Masks saved to {mask_dir}")
            except Exception as e:
                print(f"  [{seq}] ERROR during inference: {e}")
                import traceback; traceback.print_exc()
                run_meta["status"] = f"error: {e}"

        # Evaluate masks
        gt_dir = gt_root / seq
        if mask_dir.exists() and gt_dir.exists():
            metrics = compute_jm_jr_f(mask_dir, gt_dir)
            run_meta.update(metrics)
            print(f"  [{seq}] JM={metrics['JM']:.4f} JR={metrics['JR']:.4f} F={metrics['F']:.4f}")
        else:
            metrics = {"JM": -1.0, "JR": -1.0, "F": -1.0}
            run_meta.update(metrics)
            if not gt_dir.exists():
                print(f"  [{seq}] No GT available, skipping eval")

        rows.append(run_meta)

    # Save CSV
    import csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sequence", "status", "n_frames", "JM", "JR", "F",
                  "mask_coverage_stage1", "mask_coverage_stage3", "output_dir"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {out_csv}")

    # Print summary
    print("\n=== Direction B (VGGT4D) Results ===")
    print(f"{'Seq':<20} {'JM':>8} {'JR':>8} {'F':>8}")
    for r in rows:
        jm = r.get("JM", -1.0)
        jr = r.get("JR", -1.0)
        f = r.get("F", -1.0)
        jm_s = f"{jm:.4f}" if jm >= 0 else "N/A"
        jr_s = f"{jr:.4f}" if jr >= 0 else "N/A"
        f_s = f"{f:.4f}" if f >= 0 else "N/A"
        print(f"  {r['sequence']:<18} {jm_s:>8} {jr_s:>8} {f_s:>8}")


if __name__ == "__main__":
    main()
