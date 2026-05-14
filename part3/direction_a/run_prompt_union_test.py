"""
run_prompt_union_test.py — Task 3: top-3 union

 prompt policy search  union  bmx-trees
 prompt_policy_search JSON
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

SAM3_REPO = "/data3/jli657/sam3"
os.environ.setdefault("TRITON_CACHE_DIR", "/data3/jli657/tmp/triton_cache")

SEQUENCES_CFG = {
    "bmx-trees": {
        "video_root": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
        "gt_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/bmx-trees",
        # Top candidates from policy search (score-sorted)
        "single_results": [
            ("person and bicycle", 0.6361),
            ("person", 0.5193),
            ("bicycle rider", 0.5175),
            ("cyclist", 0.5164),
            ("person riding bicycle", 0.5150),
            ("mountain biker", 0.4419),
            ("bicycle", 0.2688),
        ],
        # Union combinations to test (list of prompt lists)
        "union_tests": [
            ["person and bicycle", "bicycle"],               # +bicycle
            ["person and bicycle", "person"],                # +person
            ["person and bicycle", "bicycle rider"],         # +
            ["person and bicycle", "cyclist"],               # +cyclist
            ["person", "bicycle"],                           #
            ["person and bicycle", "bicycle", "person"],     #
        ],
    },
    "car-shadow": {
        "video_root": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/car-shadow",
        "gt_dir": "/home/jli657/shared_data/project3/DAVIS/Annotations/480p/car-shadow",
        "single_results": [
            ("car on street", 0.9087),
            ("car", 0.9008),
            ("parked car", 0.9003),
            ("automobile", 0.8979),
            ("vehicle", 0.8923),
        ],
        "union_tests": [
            ["car on street", "car"],
            ["car on street", "automobile"],
            ["car on street", "car", "automobile"],
        ],
    },
}

SAM3_CHECKPOINT = "/data3/jli657/project3/weights/sam3/sam3.pt"
OUT_ROOT = Path("/data3/jli657/project3/part3/outputs/sam3_multiobj/prompt_policy_search/union_tests")
OUT_BMX_JSON = "/home/jli657/my_storage2_1T/project3/eval/prompt_policy_search_bmx.json"
OUT_CAR_JSON = "/home/jli657/my_storage2_1T/project3/eval/prompt_policy_search_carshadow.json"


def compute_jm(pred_dir: Path, gt_dir: Path) -> float:
    if not gt_dir.exists():
        return -1.0
    ious = []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = np.array(Image.open(gt_path).convert("L")) > 0
        pred = np.array(Image.open(pred_path).convert("L")) > 127
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def load_frames(video_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in video_dir.iterdir() if p.suffix.lower() in exts],
                  key=lambda p: p.stem)


def extract_mask_from_outputs(outputs: dict, frame_h: int, frame_w: int,
                               score_threshold: float = 0.3) -> np.ndarray:
    combined = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if not outputs:
        return combined
    if "out_binary_masks" in outputs:
        binary_masks = outputs["out_binary_masks"]
        probs = outputs.get("out_probs", None)
        if binary_masks is None or binary_masks.size == 0:
            return combined
        for i in range(binary_masks.shape[0]):
            if probs is not None and float(probs[i]) < score_threshold:
                continue
            mask = binary_masks[i]
            if mask.shape != (frame_h, frame_w):
                mask = cv2.resize(mask.astype(np.float32), (frame_w, frame_h),
                                  interpolation=cv2.INTER_LINEAR) > 0.5
            combined = np.maximum(combined, mask.astype(np.uint8) * 255)
    return combined


def run_sam3_union(predictor, seq_name: str, video_dir: Path,
                   output_dir: Path, prompt_texts: List[str],
                   score_threshold: float = 0.3) -> int:
    frames = load_frames(video_dir)
    if not frames:
        raise FileNotFoundError(f"No frames in {video_dir}")
    first_img = cv2.imread(str(frames[0]))
    frame_h, frame_w = first_img.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_per_frame: Dict[int, np.ndarray] = {}
    for prompt_text in prompt_texts:
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(video_dir)))
        session_id = response["session_id"]
        predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
        predictor.handle_request(request=dict(
            type="add_prompt", session_id=session_id, frame_index=0, text=prompt_text))

        for prop in predictor.handle_stream_request(
                request=dict(type="propagate_in_video", session_id=session_id)):
            fidx = prop["frame_index"]
            mask = extract_mask_from_outputs(prop["outputs"], frame_h, frame_w, score_threshold)
            combined_per_frame.setdefault(fidx, np.zeros((frame_h, frame_w), dtype=np.uint8))
            combined_per_frame[fidx] = np.maximum(combined_per_frame[fidx], mask)
        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        except Exception:
            pass

    for i, fp in enumerate(frames):
        out_path = output_dir / f"{fp.stem}.png"
        mask = combined_per_frame.get(i, np.zeros((frame_h, frame_w), dtype=np.uint8))
        Image.fromarray(mask).save(str(out_path))
    return len(frames)


def main():
    if SAM3_REPO not in sys.path:
        sys.path.insert(0, SAM3_REPO)

    print("Loading SAM3...")
    from sam3.model_builder import build_sam3_video_predictor
    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CHECKPOINT,
        gpus_to_use=[0],
        strict_state_dict_loading=False,
    )
    print("SAM3 loaded.\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for seq_name, cfg in SEQUENCES_CFG.items():
        print(f"\n{'='*60}\n[{seq_name}] Union Tests\n{'='*60}")
        video_dir = Path(cfg["video_root"])
        gt_dir = Path(cfg["gt_dir"])
        seq_out = OUT_ROOT / seq_name

        union_results = []
        for prompts in cfg["union_tests"]:
            key = " | ".join(prompts)
            out_dir = seq_out / ("_".join(p.replace(" ", "_")[:10] for p in prompts))
            print(f"\n  Testing union: {prompts}")
            try:
                run_sam3_union(predictor, seq_name, video_dir, out_dir, prompts)
                jm = compute_jm(out_dir, gt_dir)
                print(f"  JM = {jm:.4f}")
            except Exception as e:
                jm = -1.0
                print(f"  ERROR: {e}")
            union_results.append({"prompts": prompts, "jm": round(jm, 4)})

        # Combine single + union results
        all_jm = [(r["prompts"][0], r["jm"]) for r in [
            {"prompts": [p], "jm": jm} for p, jm in cfg["single_results"]
        ]] + [(str(r["prompts"]), r["jm"]) for r in union_results]

        # Sort by JM
        all_jm_sorted = sorted(all_jm, key=lambda x: x[1], reverse=True)

        all_results[seq_name] = {
            "single_results": cfg["single_results"],
            "union_results": union_results,
            "best_single": max(cfg["single_results"], key=lambda x: x[1]) if cfg["single_results"] else None,
            "best_union": max(union_results, key=lambda x: x["jm"]) if union_results else None,
            "overall_best": all_jm_sorted[0] if all_jm_sorted else None,
        }

        print(f"\n  [Summary for {seq_name}]")
        print(f"  Best single: {all_results[seq_name]['best_single']}")
        print(f"  Best union: {all_results[seq_name]['best_union']}")
        print(f"  Overall best: {all_results[seq_name]['overall_best']}")

    # Save JSON reports
    Path(OUT_BMX_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BMX_JSON, "w") as f:
        json.dump({
            "sequence": "bmx-trees",
            "task": "Prompt Policy Search + Union Test",
            "comparison_baseline": {"part2_yolo": 0.6403, "part3_best_prev_union": 0.6308},
            **all_results.get("bmx-trees", {})
        }, f, indent=2)

    with open(OUT_CAR_JSON, "w") as f:
        json.dump({
            "sequence": "car-shadow",
            "task": "Prompt Policy Search + Union Test",
            "comparison_baseline": {"part2_yolo": 0.9746, "part3_best_prev_single": 0.9008},
            **all_results.get("car-shadow", {})
        }, f, indent=2)

    print(f"\n[save] {OUT_BMX_JSON}")
    print(f"[save] {OUT_CAR_JSON}")


if __name__ == "__main__":
    main()
