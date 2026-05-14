from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


EVAL = Path("/home/jli657/my_storage2_1T/project3/eval")
OUT_DIR = Path("/data3/jli657/project3/part3/part3_deliverables")
DAVIS5 = ["tennis", "blackswan", "horsejump-low", "bmx-trees", "car-shadow"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f4(x) -> str:
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    return f"{x:.4f}"


def avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def build_mask_table() -> List[Dict[str, str]]:
    part1 = {r["sequence_name"]: float(r["J_mean"]) for r in read_csv(EVAL / "results_part1.csv")}
    part2 = {r["sequence_name"]: float(r["J_mean"]) for r in read_csv(EVAL / "results_davis_masks.csv")}
    part3_a = {r["sequence_name"]: float(r["J_mean"]) for r in read_csv(EVAL / "results_sam3_multiobj_final.csv")}
    part3_b1 = {r["sequence"]: float(r["JM"]) for r in read_csv(EVAL / "direction_b_vggt4d_results.csv")}
    part3_b5 = {r["sequence"]: float(r["JM"]) for r in read_csv(EVAL / "direction_b_sam3_refined_v5_results.csv")}

    # A+B from comparison v2
    comparison_rows = read_csv(EVAL / "direction_b_comparison_v2.csv")
    ab_best = {}
    for row in comparison_rows:
        if row["method"] == "A+B Best Fusion":
            for seq in ["tennis", "blackswan", "horsejump-low", "koala", "bmx-trees", "car-shadow"]:
                val = row.get(seq, "N/A")
                if val != "N/A":
                    ab_best[seq] = float(val)

    rows: List[Dict[str, str]] = []
    for seq in DAVIS5:
        rows.append(
            {
                "sequence": seq,
                "Part1_JM": f4(part1.get(seq)),
                "Part2_JM": f4(part2.get(seq)),
                "Part3_A_JM": f4(part3_a.get(seq)),
                "Part3_B1_JM": f4(part3_b1.get(seq)),
                "Part3_B5_JM": f4(part3_b5.get(seq)),
                "Part3_ABBest_JM": f4(ab_best.get(seq)),
            }
        )

    rows.append(
        {
            "sequence": "DAVIS5_Macro",
            "Part1_JM": f4(avg([part1[s] for s in DAVIS5])),
            "Part2_JM": f4(avg([part2[s] for s in DAVIS5])),
            "Part3_A_JM": f4(avg([part3_a[s] for s in DAVIS5])),
            "Part3_B1_JM": f4(avg([part3_b1[s] for s in DAVIS5])),
            "Part3_B5_JM": f4(avg([part3_b5[s] for s in DAVIS5])),
            "Part3_ABBest_JM": f4(avg([ab_best[s] for s in DAVIS5])),
        }
    )
    return rows


def build_video_table() -> List[Dict[str, str]]:
    rows_in = read_csv(EVAL / "unified_eval_v2.csv")
    method_map = {
        "Part2 YOLO+SAM2+PP": "Part2_Baseline",
        "Dir-A SAM3+PP": "Part3_A",
        "Dir-B VGGT4D (VGGT)": "Part3_B1",
        "Dir-B VGGT4D+SAM3": "Part3_B5",
        "A+B Best Fusion+PP": "Part3_ABBest",
    }
    per_seq: Dict[str, Dict[str, Dict[str, str]]] = {}
    for r in rows_in:
        seq = r["sequence"]
        key = method_map.get(r["method"])
        if not key:
            continue
        per_seq.setdefault(seq, {})[key] = {
            "mask_JM": r["mask_JM"],
            "PSNR_proxy": r["psnr_proxy"],
            "SSIM": r["ssim"],
        }

    rows: List[Dict[str, str]] = []
    for seq in DAVIS5 + ["koala"]:
        row = {"sequence": seq}
        for key in ["Part2_Baseline", "Part3_A", "Part3_B1", "Part3_B5", "Part3_ABBest"]:
            payload = per_seq.get(seq, {}).get(key, {})
            row[f"{key}_maskJM"] = payload.get("mask_JM", "N/A")
            row[f"{key}_PSNR"] = payload.get("PSNR_proxy", "N/A")
            row[f"{key}_SSIM"] = payload.get("SSIM", "N/A")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(mask_rows: List[Dict[str, str]], video_rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Part1 / Part2 / Part3 ",
        "",
        "",
        "",
        "## DAVIS5 Macro JM",
        "",
        "| Part |  | DAVIS5 Macro JM |",
        "|---|---|---:|",
        "| **Part 1** | YOLO+Lucas-Kanade+cv2.inpaint | **0.4922** |",
        "| **Part 2** | YOLO+SAM2+ProPainter | **0.8451** |",
        "| **Part 3 A+B Best** | (GDINO/VLM+SAM3) ∪ (VGGT4D+SAM3 refine) +ProPainter | **0.9119** |",
        "",
        "> Part2 → Part3 mask  +7.9pp +9.4%",
        "",
        "## 1. Mask JM",
        "",
        "- `Part1`",
        "- `Part2`YOLO+SAM2 ",
        "- `Part3-A`SAM3 multi-object ",
        "- `Part3-B1`VGGT4D ",
        "- `Part3-B5`VGGT4D + SAM3 refine",
        "- `Part3-A+B Best`",
        "",
        "| Sequence | Part1 | Part2 | Part3-A | Part3-B1 | Part3-B5 | Part3-A+B Best |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in mask_rows:
        lines.append(
            f"| {r['sequence']} | {r['Part1_JM']} | {r['Part2_JM']} | {r['Part3_A_JM']} | {r['Part3_B1_JM']} | {r['Part3_B5_JM']} | {r['Part3_ABBest_JM']} |"
        )

    lines.extend(
        [
            "",
            "## 2. mask_JM / PSNR_proxy / SSIM",
            "",
            "",
            "",
            "- `Part2_Baseline`Part2 YOLO+SAM2+ProPainter",
            "- `Part3_A`Part3 SAM3+ProPainter",
            "- `Part3_B1`Part3 VGGT4D  mask",
            "- `Part3_B5`Part3 VGGT4D+SAM3 refine unified ",
            "- `Part3_ABBest`Part3 A+B ",
            "",
            "| Sequence | P2 maskJM | P2 PSNR | P2 SSIM | P3-A maskJM | P3-A PSNR | P3-A SSIM | P3-B5 maskJM | P3-ABBest maskJM | P3-ABBest PSNR | P3-ABBest SSIM |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in video_rows:
        lines.append(
            f"| {r['sequence']} | {r['Part2_Baseline_maskJM']} | {r['Part2_Baseline_PSNR']} | {r['Part2_Baseline_SSIM']} | "
            f"{r['Part3_A_maskJM']} | {r['Part3_A_PSNR']} | {r['Part3_A_SSIM']} | "
            f"{r['Part3_B5_maskJM']} | {r['Part3_ABBest_maskJM']} | {r['Part3_ABBest_PSNR']} | {r['Part3_ABBest_SSIM']} |"
        )

    lines.extend(
        [
            "",
            "## 3. ",
            "",
            "- **Mask **Part1 0.4922 → Part2 0.8451 → Part3 A+B Best **0.9119**DAVIS5 Macro JM",
            "- `Direction A`GDINO/VLM + SAM3`Direction B`VGGT4D + SAM3 refine `horsejump-low``car-shadow` ",
            "- `Part3-A+B Best`  `Part3-A`SAM3 multi-object",
            "- `inpaint_only`  DAVIS GT mask  `pure_propainter_gtmask` / `sdxl_kf5_gtmask_propainter` / `lama_gtmask_propainter`",
            "",
            "`part3_deliverables/experiment_registry.csv`",
        ]
    )

    (OUT_DIR / "part123_team_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    mask_rows = build_mask_table()
    video_rows = build_video_table()
    write_csv(OUT_DIR / "part123_team_mask_comparison.csv", mask_rows)
    write_csv(OUT_DIR / "part123_team_video_comparison.csv", video_rows)
    write_markdown(mask_rows, video_rows)
    payload = {"mask_rows": mask_rows, "video_rows": video_rows}
    (OUT_DIR / "part123_team_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[done] team comparison files generated")


if __name__ == "__main__":
    main()
