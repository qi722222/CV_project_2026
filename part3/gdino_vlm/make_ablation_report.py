"""
make_ablation_report.py
-----------------------
CSVrun_metaMarkdown
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GDINO")
    parser.add_argument("--ablation_csv", default="part3/gdino_vlm/gdino_ablation.csv")
    parser.add_argument("--stage1_meta", default="part3/gdino_vlm/masks/stage1/tennis/run_meta.json")
    parser.add_argument("--stage2_meta", default="part3/gdino_vlm/masks/stage2/tennis/run_meta.json")
    parser.add_argument("--output_md", default="part3/gdino_vlm/ablation_summary.md")
    return parser.parse_args()


def load_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.ablation_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("ablation csv")

    s1_meta = load_meta(Path(args.stage1_meta))
    s2_meta = load_meta(Path(args.stage2_meta))

    lines = ["# GDINO ", ""]
    lines.append("| Sequence | YOLO IoU/J | GDINO-S1 IoU/J | GDINO-S2 IoU/J | YOLO F | S1 F | S2 F |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['sequence_name']} | {r['yolo_iou_j']} | {r['gdino_s1_iou_j']} | {r['gdino_s2_iou_j']} | "
            f"{r['yolo_f']} | {r['gdino_s1_f']} | {r['gdino_s2_f']} |"
        )
    lines.append("")
    lines.append("## ")
    lines.append("")
    lines.append(f"- Stage1 detector_actual: `{s1_meta.get('detector_actual', 'unknown')}`")
    lines.append(f"- Stage1 anchors: `{s1_meta.get('num_anchor_frames', 'unknown')}`")
    lines.append(f"- Stage2 detector_actual: `{s2_meta.get('detector_actual', 'unknown')}`")
    lines.append(f"- Stage2 anchors: `{s2_meta.get('num_anchor_frames', 'unknown')}`")
    lines.append("")
    lines.append("## ")
    lines.append("")
    if s1_meta.get("detector_actual") != "gdino" or s2_meta.get("detector_actual") != "gdino":
        lines.append("-  `yolo_fallback`Stage1/Stage2")
    lines.append("- Stage2  tennis  Stage1 ")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
