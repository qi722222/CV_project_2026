"""
make_davis_table.py
-------------------
CSVpolicyMarkdown
mask/video
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAVIS")
    parser.add_argument("--policy", default="eval/davis_eval_targets.yaml")
    parser.add_argument("--metrics_csv", default="eval/results_davis_masks.csv")
    parser.add_argument("--part2_outputs_root", default="part2/outputs")
    parser.add_argument("--output_md", default="eval/davis_results_table.md")
    return parser.parse_args()


def load_policy(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            out[row["sequence_name"]] = row
        return out


def find_video_path(out_dir: Path) -> str:
    if not out_dir.exists():
        return "N/A"
    mp4s = sorted(out_dir.glob("*.mp4"))
    if mp4s:
        return str(mp4s[0])
    return "N/A"


def make_table(
    policy: Dict,
    metrics: Dict[str, Dict[str, str]],
    outputs_root: Path,
) -> str:
    lines: List[str] = []
    lines.append("# DAVIS ")
    lines.append("")
    lines.append("| Sequence | Frames | IoU/J | JR@0.5 | F | Mask Dir | Video |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    for seq in policy["sequences"]:
        name = seq["sequence_name"]
        pred_subdir = seq.get("pred_subdir", name)
        pred_dir = Path(policy["defaults"]["pred_root"]) / pred_subdir
        video_dir = outputs_root / pred_subdir
        video = find_video_path(video_dir)
        row = metrics.get(name)
        if row is None:
            frames = "0"
            iou = "N/A"
            jr = "N/A"
            f = "N/A"
        else:
            frames = row.get("num_frames", "0")
            iou = row.get("iou_mean", "N/A")
            jr = row.get("JR_at_tau", "N/A")
            f = row.get("F_mean", "N/A")
        lines.append(
            f"| {name} | {frames} | {iou} | {jr} | {f} | `{pred_dir}` | `{video}` |"
        )
    lines.append("")
    lines.append(
        "> DAVIS GT  `Video`  `N/A` mask "
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    policy = load_policy(Path(args.policy))
    metrics = load_metrics(Path(args.metrics_csv))
    md = make_table(policy, metrics, Path(args.part2_outputs_root))
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
