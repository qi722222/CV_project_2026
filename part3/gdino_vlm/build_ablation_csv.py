"""
build_ablation_csv.py
---------------------
 YOLO / GDINO Stage1 / GDINO Stage2 CSV
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GDINOCSV")
    parser.add_argument("--yolo_csv", required=True)
    parser.add_argument("--gdino_s1_csv", required=True)
    parser.add_argument("--gdino_s2_csv", required=False, default="")
    parser.add_argument("--output_csv", default="part3/gdino_vlm/gdino_ablation.csv")
    return parser.parse_args()


def read_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["sequence_name"]: row for row in reader}


def main() -> None:
    args = parse_args()
    yolo = read_metrics(Path(args.yolo_csv))
    s1 = read_metrics(Path(args.gdino_s1_csv))
    s2 = read_metrics(Path(args.gdino_s2_csv)) if args.gdino_s2_csv else {}

    seqs = sorted(set(yolo.keys()) | set(s1.keys()) | set(s2.keys()))
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sequence_name",
                "yolo_iou_j",
                "gdino_s1_iou_j",
                "gdino_s2_iou_j",
                "yolo_f",
                "gdino_s1_f",
                "gdino_s2_f",
                "note",
            ]
        )
        for s in seqs:
            y = yolo.get(s, {})
            a = s1.get(s, {})
            b = s2.get(s, {})
            note = ""
            if not b:
                note = "stage2_missing_or_not_run"
            w.writerow(
                [
                    s,
                    y.get("iou_mean", ""),
                    a.get("iou_mean", ""),
                    b.get("iou_mean", ""),
                    y.get("F_mean", ""),
                    a.get("F_mean", ""),
                    b.get("F_mean", ""),
                    note,
                ]
            )
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
