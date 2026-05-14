"""
run_official_sam3_video.py

 facebookresearch/sam3  video text prompt  DAVIS
 eval_davis_masks.py  mask PNG

:
  python part3/run_official_sam3_video.py \
    --sequences tennis blackswan bmx-trees car-shadow horsejump-low \
    --output_root part3/outputs/official_sam3_video/masks \
    --checkpoint /data3/jli657/project3/weights/sam3/sam3.pt

 prompt  --prompt_map_yaml YAML
 --prompts : sequence_name:prompt_text
"""

from __future__ import annotations

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# SAM3 repo
SAM3_REPO = "/data3/jli657/sam3"

# text promptDirection A
DEFAULT_PROMPTS: Dict[str, str] = {
    "tennis": "person",
    "bmx-trees": "person",
    "blackswan": "black swan",
    "car-shadow": "car",
    "horsejump-low": "horse",
    "bear": "bear",
    "koala": "koala",
    "camel": "camel",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official SAM3 video text prompt mask generation")
    p.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_PROMPTS.keys()),
        help="Sequences to process",
    )
    p.add_argument(
        "--video_root",
        default="/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p",
        help="Root dir with DAVIS JPEG frames",
    )
    p.add_argument(
        "--output_root",
        default="part3/outputs/official_sam3_video/masks",
        help="Output root for mask PNGs",
    )
    p.add_argument(
        "--checkpoint",
        default="/data3/jli657/project3/weights/sam3/sam3.pt",
        help="Path to official SAM3 checkpoint",
    )
    p.add_argument(
        "--prompts",
        nargs="*",
        default=[],
        help="Per-sequence prompts: 'sequence_name:prompt_text'",
    )
    p.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to add the text prompt on",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use",
    )
    p.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Score threshold for mask output",
    )
    return p.parse_args()


def build_prompt_map(args: argparse.Namespace) -> Dict[str, str]:
    """Build per-sequence prompt map from args.prompts and defaults."""
    prompt_map = dict(DEFAULT_PROMPTS)
    for item in args.prompts:
        if ":" in item:
            seq, prompt = item.split(":", 1)
            prompt_map[seq.strip()] = prompt.strip()
    return prompt_map


def load_frames(video_dir: Path) -> List[Path]:
    """Load sorted frame paths from a DAVIS sequence directory."""
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem,
    )
    return frames


def extract_mask_from_outputs(
    outputs: dict, frame_h: int, frame_w: int, score_threshold: float = 0.3
) -> np.ndarray:
    """
    Extract binary mask from official SAM3 propagation outputs.

    SAM3 output structure:
        {
            "out_obj_ids": np.ndarray (n_objects,),
            "out_probs": np.ndarray (n_objects,),
            "out_binary_masks": np.ndarray (n_objects, H, W), dtype=bool,
            "out_boxes_xywh": np.ndarray (n_objects, 4),
            "frame_stats": dict,
        }

    Returns uint8 mask (0/255) combining all detected objects above score_threshold.
    """
    combined = np.zeros((frame_h, frame_w), dtype=np.uint8)

    if not outputs:
        return combined

    # Official SAM3 output format
    if "out_binary_masks" in outputs:
        binary_masks = outputs["out_binary_masks"]  # (n_objects, H, W), bool
        probs = outputs.get("out_probs", None)      # (n_objects,), float32

        if binary_masks is None or binary_masks.size == 0:
            return combined

        n_objects = binary_masks.shape[0]
        for i in range(n_objects):
            # Skip low-confidence detections
            if probs is not None and float(probs[i]) < score_threshold:
                continue

            mask = binary_masks[i]  # (H, W), bool

            # Resize if needed
            if mask.shape != (frame_h, frame_w):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_LINEAR,
                ) > 0.5

            binary = mask.astype(np.uint8) * 255
            combined = np.maximum(combined, binary)
        return combined

    # Fallback: legacy format (obj_id -> dict or tensor)
    import torch
    if isinstance(outputs, dict):
        items = outputs.items()
    else:
        items = enumerate(outputs)

    for _obj_id, obj_data in items:
        if obj_data is None:
            continue
        if isinstance(obj_data, dict):
            masks = obj_data.get("masks", None)
            if masks is None:
                continue
        else:
            masks = obj_data

        if isinstance(masks, torch.Tensor):
            mask_np = masks.squeeze().cpu().numpy()
        elif isinstance(masks, np.ndarray):
            mask_np = masks.squeeze()
        else:
            continue

        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.shape != (frame_h, frame_w):
            mask_np = cv2.resize(
                mask_np.astype(np.float32),
                (frame_w, frame_h),
                interpolation=cv2.INTER_LINEAR,
            )
        combined = np.maximum(combined, (mask_np > 0.5).astype(np.uint8) * 255)

    return combined


def process_sequence(
    predictor,
    seq_name: str,
    video_dir: Path,
    output_dir: Path,
    prompt_text: str,
    frame_idx: int = 0,
) -> dict:
    """Run SAM3 video text prompt on one sequence. Returns meta dict."""
    frames = load_frames(video_dir)
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_dir}")

    print(f"[{seq_name}] frames={len(frames)}, prompt='{prompt_text}', anchor_frame={frame_idx}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions
    first_img = cv2.imread(str(frames[0]))
    frame_h, frame_w = first_img.shape[:2]

    # Start session
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(video_dir),
        )
    )
    session_id = response["session_id"]

    # Reset session to ensure clean state
    predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )

    # Add text prompt on the specified frame
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text,
        )
    )

    # Check if any objects were detected on anchor frame
    anchor_outputs = response.get("outputs", {})
    anchor_mask = extract_mask_from_outputs(anchor_outputs, frame_h, frame_w)
    num_detected = len(anchor_outputs) if isinstance(anchor_outputs, dict) else 0

    print(f"[{seq_name}] detected {num_detected} objects on frame {frame_idx}")

    # Propagate through video
    outputs_per_frame: dict = {}
    for prop_response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        fidx = prop_response["frame_index"]
        outputs_per_frame[fidx] = prop_response["outputs"]

    print(f"[{seq_name}] propagated {len(outputs_per_frame)} frames")

    # Save masks as PNG (0/255 binary)
    png_count = 0
    for i, frame_path in enumerate(frames):
        stem = frame_path.stem  # e.g. "00000"
        out_path = output_dir / f"{stem}.png"

        if i in outputs_per_frame:
            mask = extract_mask_from_outputs(
                outputs_per_frame[i], frame_h, frame_w, score_threshold=0.3
            )
        else:
            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

        Image.fromarray(mask).save(str(out_path))
        png_count += 1

    # Close session
    try:
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
    except Exception:
        pass

    meta = {
        "sequence_name": seq_name,
        "prompt_text": prompt_text,
        "num_frames": len(frames),
        "png_count": png_count,
        "num_detected_anchor": num_detected,
        "anchor_frame": frame_idx,
        "masks_dir": str(output_dir),
        "prompt_source": "official_sam3_text",
    }
    return meta


def main() -> None:
    args = parse_args()
    prompt_map = build_prompt_map(args)

    project_root = Path.cwd().resolve()
    output_root = (project_root / args.output_root).resolve()
    video_root = Path(args.video_root)

    # Add SAM3 repo to path and import
    if SAM3_REPO not in sys.path:
        sys.path.insert(0, SAM3_REPO)

    print("Loading official SAM3 video predictor...")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=args.checkpoint,
        gpus_to_use=[args.gpu],
        strict_state_dict_loading=False,
    )
    print("Official SAM3 loaded successfully.")

    all_meta = []
    for seq in args.sequences:
        video_dir = video_root / seq
        if not video_dir.exists():
            print(f"[warn] video dir not found: {video_dir}, skipping")
            continue

        prompt_text = prompt_map.get(seq, "object")
        output_dir = output_root / seq

        try:
            meta = process_sequence(
                predictor=predictor,
                seq_name=seq,
                video_dir=video_dir,
                output_dir=output_dir,
                prompt_text=prompt_text,
                frame_idx=args.frame_idx,
            )
            all_meta.append({"status": "ok", **meta})
            print(f"[ok] {seq}: {meta['png_count']} masks saved to {output_dir}")
        except Exception as exc:
            import traceback
            print(f"[error] {seq}: {exc}")
            traceback.print_exc()
            all_meta.append({"status": "error", "sequence_name": seq, "error": str(exc)})

    # Save manifest
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n[save] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
