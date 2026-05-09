from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build backend-aware GDINO ablation CSV")
    p.add_argument("--eval_yolo", default="part3/gdino_vlm/eval_yolo.csv")
    p.add_argument("--eval_sam2_s1", default="part3/gdino_vlm/eval_stage1.csv")
    p.add_argument("--eval_sam2_s2", default="part3/gdino_vlm/eval_stage2.csv")
    p.add_argument("--eval_sam3_s1", default="part3/gdino_vlm/eval_sam3_stage1.csv")
    p.add_argument("--eval_sam3_s2", default="part3/gdino_vlm/eval_sam3_stage2.csv")
    p.add_argument("--output_csv", default="part3/gdino_vlm/gdino_ablation.csv")
    p.add_argument("--output_md", default="part3/gdino_vlm/ablation_summary.md")
    return p.parse_args()


def read_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["sequence_name"]: r for r in rows}


def to_f(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def macro(rows: Dict[str, Dict[str, str]], key: str) -> float:
    vals = [to_f(r.get(key, "nan")) for r in rows.values()]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    args = parse_args()
    data_sources: List[Tuple[str, str, str, Dict[str, Dict[str, str]]]] = [
        ("part2_yolo", "na", "stage1", read_rows(Path(args.eval_yolo))),
        ("part3_gdino", "sam2", "stage1", read_rows(Path(args.eval_sam2_s1))),
        ("part3_gdino", "sam2", "stage2", read_rows(Path(args.eval_sam2_s2))),
        ("part3_gdino", "sam3", "stage1", read_rows(Path(args.eval_sam3_s1))),
        ("part3_gdino", "sam3", "stage2", read_rows(Path(args.eval_sam3_s2))),
    ]
    seqs = sorted({s for _, _, _, d in data_sources for s in d.keys()})
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sequence_name",
                "method_tag",
                "segmentor_backend",
                "stage",
                "iou_mean",
                "f_mean",
            ]
        )
        for seq in seqs:
            for method_tag, backend, stage, d in data_sources:
                r = d.get(seq)
                if r is None:
                    continue
                w.writerow([seq, method_tag, backend, stage, r.get("iou_mean", ""), r.get("F_mean", "")])

    # markdown summary
    yolo = read_rows(Path(args.eval_yolo))
    s2s1 = read_rows(Path(args.eval_sam2_s1))
    s2s2 = read_rows(Path(args.eval_sam2_s2))
    s3s1 = read_rows(Path(args.eval_sam3_s1))
    s3s2 = read_rows(Path(args.eval_sam3_s2))
    lines = [
        "# GDINO 消融总结（含 SAM2/SAM3 版本字段）",
        "",
        "| Variant | Macro IoU/J | Macro F |",
        "|---|---:|---:|",
        f"| Part2 YOLO baseline | {macro(yolo, 'iou_mean'):.6f} | {macro(yolo, 'F_mean'):.6f} |",
        f"| Part3 SAM2 Stage1 | {macro(s2s1, 'iou_mean'):.6f} | {macro(s2s1, 'F_mean'):.6f} |",
        f"| Part3 SAM2 Stage2 | {macro(s2s2, 'iou_mean'):.6f} | {macro(s2s2, 'F_mean'):.6f} |",
        f"| Part3 SAM3 Stage1 | {macro(s3s1, 'iou_mean'):.6f} | {macro(s3s1, 'F_mean'):.6f} |",
        f"| Part3 SAM3 Stage2 | {macro(s3s2, 'iou_mean'):.6f} | {macro(s3s2, 'F_mean'):.6f} |",
        "",
        "- 观察：SAM3 Stage1 在 tennis / blackswan / car-shadow 保持较高精度，但 bmx-trees 上明显弱于 SAM2 Stage1。",
        "- 观察：SAM3 Stage2 当前低于 SAM3 Stage1，说明稀疏重锚在现参数下仍有退化风险。",
        "",
        f"- CSV: `{out_csv}`",
    ]
    out_md = Path(args.output_md)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out_csv}")
    print(f"[save] {out_md}")


if __name__ == "__main__":
    main()
