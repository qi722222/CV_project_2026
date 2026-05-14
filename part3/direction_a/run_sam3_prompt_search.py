"""
run_sam3_prompt_search.py

 DAVIS  text prompt JM/JR/F
 eval/prompt_search_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

SAM3_REPO = "/data3/jli657/sam3"

# prompts 3-5
CANDIDATE_PROMPTS: Dict[str, List[str]] = {
    "tennis": ["person", "tennis player", "player"],
    "bmx-trees": ["cyclist", "biker", "bicycle rider"],
    "blackswan": ["black swan", "swan", "bird"],
    "car-shadow": ["car", "vehicle", "automobile"],
    "horsejump-low": ["horse", "horse and rider", "equestrian"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+", default=list(CANDIDATE_PROMPTS.keys()))
    p.add_argument("--video_root", default="/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
    p.add_argument("--gt_root", default="/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
    p.add_argument("--output_root", default="part3/outputs/official_sam3_video/prompt_search")
    p.add_argument("--checkpoint", default="/data3/jli657/project3/weights/sam3/sam3.pt")
    p.add_argument("--eval_script", default="eval/eval_davis_masks.py")
    p.add_argument("--eval_python", default="/home/jli657/.conda/envs/gdino_env/bin/python")
    p.add_argument("--sam3_python", default="/data3/jli657/envs/sam3_official_env/bin/python")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def run_sam3_for_prompt(
    seq: str, prompt: str, video_root: str, output_dir: Path, checkpoint: str, gpu: int, sam3_python: str
) -> bool:
    """Run official SAM3 for a specific sequence and prompt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd()

    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = "/data3/jli657/tmp/triton_cache"

    cmd = [
        sam3_python,
        str(project_root / "part3/run_official_sam3_video.py"),
        "--sequences", seq,
        "--video_root", video_root,
        "--output_root", str(output_dir.parent),
        "--checkpoint", checkpoint,
        "--prompts", f"{seq}:{prompt}",
        "--gpu", str(gpu),
    ]
    ret = subprocess.run(cmd, cwd=str(project_root), env=env, capture_output=True)
    return ret.returncode == 0


def eval_mask_quality(
    seq: str, mask_dir: Path, gt_root: str, eval_python: str, eval_script: str, project_root: Path
) -> Tuple[float, float, float]:
    """Run eval_davis_masks.py and return (JM, JR, F)."""
    import tempfile
    import yaml

    gt_root_path = Path(gt_root)
    gt_seq_dir = gt_root_path / seq
    if not gt_seq_dir.exists():
        return 0.0, 0.0, 0.0

    policy = {
        "version": 1,
        "defaults": {
            "gt_root": str(gt_root_path),
            "pred_root": str(mask_dir.parent),
            "iou_threshold_for_jr": 0.5,
            "boundary_tolerance_px": 2,
            "strict_missing_predictions": False,
        },
        "sequences": [
            {"sequence_name": seq, "eval_mode": "union_all_instances", "pred_subdir": seq}
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir="/data3/jli657/tmp") as f:
        import yaml
        yaml.dump(policy, f)
        policy_path = f.name

    out_csv = policy_path.replace(".yaml", ".csv")
    cmd = [eval_python, str(project_root / eval_script), "--policy", policy_path, "--output_csv", out_csv]
    ret = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)

    try:
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sequence_name"] == seq:
                    return float(row["iou_mean"]), float(row["JR_at_tau"]), float(row["F_mean"])
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    output_root = (project_root / args.output_root).resolve()

    results = []
    best_per_seq: Dict[str, dict] = {}

    for seq in args.sequences:
        candidates = CANDIDATE_PROMPTS.get(seq, ["object"])
        print(f"\n{'='*60}")
        print(f"Sequence: {seq}, candidates: {candidates}")

        for prompt in candidates:
            safe_prompt = prompt.replace(" ", "_")
            run_dir = output_root / f"{seq}_{safe_prompt}"

            print(f"  Testing prompt: '{prompt}'")
            ok = run_sam3_for_prompt(
                seq=seq,
                prompt=prompt,
                video_root=args.video_root,
                output_dir=run_dir / seq,
                checkpoint=args.checkpoint,
                gpu=args.gpu,
                sam3_python=args.sam3_python,
            )

            if not ok:
                print(f"    [error] SAM3 failed for prompt '{prompt}'")
                jm, jr, f = 0.0, 0.0, 0.0
            else:
                jm, jr, f = eval_mask_quality(
                    seq=seq,
                    mask_dir=run_dir / seq,
                    gt_root=args.gt_root,
                    eval_python=args.eval_python,
                    eval_script=args.eval_script,
                    project_root=project_root,
                )

            print(f"    JM={jm:.4f}, JR={jr:.4f}, F={f:.4f}")
            row = {"seq": seq, "prompt": prompt, "jm": jm, "jr": jr, "f": f}
            results.append(row)

            if seq not in best_per_seq or jm > best_per_seq[seq]["jm"]:
                best_per_seq[seq] = {**row, "run_dir": str(run_dir)}

    # Save results
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = project_root / "eval/prompt_search_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "prompt", "jm", "jr", "f"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print("BEST PROMPTS PER SEQUENCE:")
    for seq, best in best_per_seq.items():
        print(f"  {seq}: '{best['prompt']}' → JM={best['jm']:.4f}")

    json_path = project_root / "eval/prompt_search_best.json"
    with open(json_path, "w") as f:
        json.dump(best_per_seq, f, indent=2)
    print(f"\n[save] Results: {csv_path}")
    print(f"[save] Best prompts: {json_path}")


if __name__ == "__main__":
    main()
