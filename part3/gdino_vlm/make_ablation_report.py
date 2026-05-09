"""
make_ablation_report.py
-----------------------
根据三向消融CSV与run_meta生成简短Markdown结论草稿。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成GDINO消融报告草稿")
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
        raise FileNotFoundError(f"找不到CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("ablation csv为空")

    s1_meta = load_meta(Path(args.stage1_meta))
    s2_meta = load_meta(Path(args.stage2_meta))

    lines = ["# GDINO 三向消融小结（自动生成）", ""]
    lines.append("| Sequence | YOLO IoU/J | GDINO-S1 IoU/J | GDINO-S2 IoU/J | YOLO F | S1 F | S2 F |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['sequence_name']} | {r['yolo_iou_j']} | {r['gdino_s1_iou_j']} | {r['gdino_s2_iou_j']} | "
            f"{r['yolo_f']} | {r['gdino_s1_f']} | {r['gdino_s2_f']} |"
        )
    lines.append("")
    lines.append("## 运行元信息")
    lines.append("")
    lines.append(f"- Stage1 detector_actual: `{s1_meta.get('detector_actual', 'unknown')}`")
    lines.append(f"- Stage1 anchors: `{s1_meta.get('num_anchor_frames', 'unknown')}`")
    lines.append(f"- Stage2 detector_actual: `{s2_meta.get('detector_actual', 'unknown')}`")
    lines.append(f"- Stage2 anchors: `{s2_meta.get('num_anchor_frames', 'unknown')}`")
    lines.append("")
    lines.append("## 可直接写入报告的限制说明")
    lines.append("")
    if s1_meta.get("detector_actual") != "gdino" or s2_meta.get("detector_actual") != "gdino":
        lines.append("- 当前实验在检测端使用了 `yolo_fallback`，Stage1/Stage2结果用于验证重锚机制，不宣称开放词汇能力。")
    lines.append("- Stage2 在 tennis 上较 Stage1 数值下降，说明稀疏重锚需要更稳健的关联策略或更高质量检测框。")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
