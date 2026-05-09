from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified Part1/2/3(SAM2+SAM3) comparison table")
    parser.add_argument("--part1_csv", default="part3/gdino_vlm/eval_part1.csv")
    parser.add_argument("--part2_csv", default="part3/gdino_vlm/eval_yolo.csv")
    parser.add_argument("--sam2_s1_csv", default="part3/gdino_vlm/eval_stage1.csv")
    parser.add_argument("--sam2_s2_csv", default="part3/gdino_vlm/eval_stage2.csv")
    parser.add_argument("--sam3_s1_csv", default="part3/gdino_vlm/eval_sam3_stage1.csv")
    parser.add_argument("--sam3_s2_csv", default="part3/gdino_vlm/eval_sam3_stage2.csv")
    parser.add_argument("--output_csv", default="part3/gdino_vlm/part123_sam2_sam3_compare.csv")
    parser.add_argument("--output_md", default="part3/gdino_vlm/part123_sam2_sam3_compare.md")
    return parser.parse_args()


def read_csv_map(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["sequence_name"]: r for r in rows}


def to_float(row: Dict[str, str], key: str = "iou_mean") -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def mean_safe(vals: List[float]) -> float:
    good = [v for v in vals if v == v]
    return sum(good) / len(good) if good else float("nan")


def main() -> None:
    args = parse_args()
    p1 = read_csv_map(Path(args.part1_csv))
    p2 = read_csv_map(Path(args.part2_csv))
    s2s1 = read_csv_map(Path(args.sam2_s1_csv))
    s2s2 = read_csv_map(Path(args.sam2_s2_csv))
    s3s1 = read_csv_map(Path(args.sam3_s1_csv))
    s3s2 = read_csv_map(Path(args.sam3_s2_csv))

    seqs = sorted(set(p1) | set(p2) | set(s2s1) | set(s2s2) | set(s3s1) | set(s3s2))
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sequence_name",
                "part1_iou_j",
                "part2_iou_j",
                "part3_sam2_s1_iou_j",
                "part3_sam2_s2_iou_j",
                "part3_sam3_s1_iou_j",
                "part3_sam3_s2_iou_j",
                "part1_f",
                "part2_f",
                "part3_sam2_s1_f",
                "part3_sam2_s2_f",
                "part3_sam3_s1_f",
                "part3_sam3_s2_f",
            ]
        )
        for s in seqs:
            w.writerow(
                [
                    s,
                    p1.get(s, {}).get("iou_mean", ""),
                    p2.get(s, {}).get("iou_mean", ""),
                    s2s1.get(s, {}).get("iou_mean", ""),
                    s2s2.get(s, {}).get("iou_mean", ""),
                    s3s1.get(s, {}).get("iou_mean", ""),
                    s3s2.get(s, {}).get("iou_mean", ""),
                    p1.get(s, {}).get("F_mean", ""),
                    p2.get(s, {}).get("F_mean", ""),
                    s2s1.get(s, {}).get("F_mean", ""),
                    s2s2.get(s, {}).get("F_mean", ""),
                    s3s1.get(s, {}).get("F_mean", ""),
                    s3s2.get(s, {}).get("F_mean", ""),
                ]
            )

    macro = {
        "part1_iou_j": mean_safe([to_float(r) for r in p1.values()]),
        "part2_iou_j": mean_safe([to_float(r) for r in p2.values()]),
        "sam2_s1_iou_j": mean_safe([to_float(r) for r in s2s1.values()]),
        "sam2_s2_iou_j": mean_safe([to_float(r) for r in s2s2.values()]),
        "sam3_s1_iou_j": mean_safe([to_float(r) for r in s3s1.values()]),
        "sam3_s2_iou_j": mean_safe([to_float(r) for r in s3s2.values()]),
    }
    lines = [
        "# Unified DAVIS Comparison (Part1/Part2/Part3 SAM2+SAM3)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Part1 Macro IoU/J | {macro['part1_iou_j']:.6f} |",
        f"| Part2 Macro IoU/J | {macro['part2_iou_j']:.6f} |",
        f"| Part3 SAM2 Stage1 Macro IoU/J | {macro['sam2_s1_iou_j']:.6f} |",
        f"| Part3 SAM2 Stage2 Macro IoU/J | {macro['sam2_s2_iou_j']:.6f} |",
        f"| Part3 SAM3 Stage1 Macro IoU/J | {macro['sam3_s1_iou_j']:.6f} |",
        f"| Part3 SAM3 Stage2 Macro IoU/J | {macro['sam3_s2_iou_j']:.6f} |",
        "",
        f"- CSV: `{out_csv}`",
    ]
    out_md = Path(args.output_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out_csv}")
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
