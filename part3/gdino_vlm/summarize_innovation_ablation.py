from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize SAM3 innovation ablation")
    p.add_argument("--baseline_csv", default="part3/gdino_vlm/eval_sam3_stage2_davis2.csv")
    p.add_argument("--quality_gate_csv", default="part3/gdino_vlm/eval_sam3_innov_quality_gate.csv")
    p.add_argument("--o2o_csv", default="part3/gdino_vlm/eval_sam3_innov_o2o.csv")
    p.add_argument("--real_vlm_csv", default="part3/gdino_vlm/eval_sam3_innov_real_vlm.csv")
    p.add_argument("--output_md", default="part3/gdino_vlm/sam3_innovation_ablation.md")
    return p.parse_args()


def read_map(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["sequence_name"]: r for r in rows}


def fval(row: Dict[str, str], key: str = "iou_mean") -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def macro(rows: Dict[str, Dict[str, str]], key: str = "iou_mean") -> float:
    vals = [fval(r, key) for r in rows.values()]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    args = parse_args()
    base = read_map(Path(args.baseline_csv))
    qg = read_map(Path(args.quality_gate_csv))
    o2o = read_map(Path(args.o2o_csv))
    rv = read_map(Path(args.real_vlm_csv))
    seqs = sorted(set(base) | set(qg) | set(o2o) | set(rv))

    lines = ["# SAM3 ", ""]
    lines.append("| Sequence | Baseline-S2 | +QualityGate | +O2O | +RealVLM |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in seqs:
        lines.append(
            f"| {s} | {fval(base.get(s, {})):.6f} | {fval(qg.get(s, {})):.6f} | "
            f"{fval(o2o.get(s, {})):.6f} | {fval(rv.get(s, {})):.6f} |"
        )

    m_base = macro(base)
    m_qg = macro(qg)
    m_o2o = macro(o2o)
    m_rv = macro(rv)
    lines.extend(
        [
            "",
            "## Macro IoU/J",
            "",
            f"- Baseline-S2: `{m_base:.6f}`",
            f"- +QualityGate: `{m_qg:.6f}` (delta `{m_qg - m_base:+.6f}`)",
            f"- +O2O: `{m_o2o:.6f}` (delta `{m_o2o - m_base:+.6f}`)",
            f"- +RealVLM: `{m_rv:.6f}` (delta `{m_rv - m_base:+.6f}`)",
            "",
            "## ",
            "",
            "- BO2O `bmx-trees` /ID",
            "- AQualityGate",
            "- CRealVLM`prompt_source=real_vlm` promptprompt",
            "-  `outputs/sam3/*bmx-trees*`  overlay ",
        ]
    )
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
