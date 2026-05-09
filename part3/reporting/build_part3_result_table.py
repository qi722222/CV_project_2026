from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DELIVERABLES = Path("/data3/jli657/project3/part3/part3_deliverables")
OUT_CSV = DELIVERABLES / "part3_results_full_table.csv"
OUT_MD = DELIVERABLES / "part3_results_full_table.md"
OUT_JSON = DELIVERABLES / "part3_results_full_table.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_value(metrics: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in metrics:
            return fmt(metrics[key])
    return ""


def row_quality_note(row: Dict[str, str]) -> str:
    ctype = row["comparison_type"]
    status = row["audit_status"]
    if status == "failed":
        return "失败实验：保留用于解释为什么这条路线暂时不主推。"
    if status in {"partial_or_failed", "superseded", "legacy"}:
        return "中间或旧版结果：可追溯，但引用时需要看 experiment_card。"
    if ctype == "mask_only":
        return "只看 mask，不应直接和 inpaint 视频指标混排。"
    if ctype == "inpaint_only":
        return "固定 mask 比修复工具，重点看 masked_in 与 inpaint_out。"
    if ctype == "full_pipeline":
        return "完整流程结果，适合看最终效果，但要能拆回 mask 和 inpaint。"
    return ""


def build_rows() -> List[Dict[str, str]]:
    registry = load_json(DELIVERABLES / "experiment_registry.json")
    rows: List[Dict[str, str]] = []
    for item in registry:
        method_dir = Path(item["deliverable_dir"])
        metrics_path = method_dir / "metrics.json"
        metrics_blob = load_json(metrics_path) if metrics_path.exists() else {}
        metrics = metrics_blob.get("metrics", {})
        evidence = metrics_blob.get("evidence_paths", {})
        row: Dict[str, str] = {
            "sequence": item["sequence"],
            "method_id": item["method_id"],
            "readable_name": item["readable_name"],
            "family": item["family"],
            "comparison_type": item["comparison_type"],
            "audit_status": item["audit_status"],
            # schema v2 fields
            "version": item.get("version", "legacy"),
            "mask_protocol": item.get("mask_protocol", ""),
            "baseline": item.get("baseline", ""),
            "next_decision": item.get("next_decision", ""),
            "failure_reason": item.get("failure_reason", ""),
            "JM_or_mask_JM": metric_value(metrics, "JM", "mask_JM", "best_JM"),
            "JR": metric_value(metrics, "JR"),
            "F": metric_value(metrics, "F"),
            "PSNR_proxy": metric_value(metrics, "PSNR_proxy"),
            "PSNR_synthetic": metric_value(metrics, "PSNR_synthetic"),
            "SSIM": metric_value(metrics, "SSIM", "SSIM_proxy"),
            "n_frames": metric_value(metrics, "n_frames"),
            "routing_or_prompt": metric_value(metrics, "routing_strategy", "prompt_mode", "best_scale"),
            "metrics_source": metrics.get("source_file", item.get("metrics_source", "")),
            "mask_frames": evidence.get("mask_frames", evidence.get("masks_dilated", "")),
            "masked_in": evidence.get("masked_in", evidence.get("generated_masked_in", "")),
            "inpaint_out": evidence.get("inpaint_out", ""),
            "script_path": item.get("script_path", ""),
            "config_path": item.get("config_path", ""),
            "deliverable_dir": item["deliverable_dir"],
            "plain_explanation": item["plain_explanation"],
            "current_takeaway": item["current_takeaway"],
        }
        row["quality_note"] = row_quality_note(row)
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, str]]) -> None:
    fields = [
        "sequence",
        "method_id",
        "readable_name",
        "family",
        "comparison_type",
        "audit_status",
        "version",
        "mask_protocol",
        "baseline",
        "next_decision",
        "failure_reason",
        "JM_or_mask_JM",
        "JR",
        "F",
        "PSNR_proxy",
        "PSNR_synthetic",
        "SSIM",
        "n_frames",
        "routing_or_prompt",
        "quality_note",
        "metrics_source",
        "mask_frames",
        "masked_in",
        "inpaint_out",
        "script_path",
        "config_path",
        "deliverable_dir",
        "plain_explanation",
        "current_takeaway",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]]) -> None:
    by_type: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_type.setdefault(row["comparison_type"], []).append(row)

    lines = [
        "# Part3 完整结果整理表",
        "",
        "这张表来自 `part3_deliverables/experiment_registry.json` 和每个方法目录下的 `metrics.json`。",
        "",
        "## 怎么读",
        "",
        "- `mask_only`：只看 mask 指标，不能直接拿 PSNR/SSIM 比。",
        "- `inpaint_only`：固定 mask 后比较修复工具。",
        "- `full_pipeline`：mask + 修复工具一起看，是最终视频效果。",
        "- `failed / partial_or_failed / superseded / legacy`：保留用于追溯，不建议直接作为主结果引用。",
        "",
        "## 汇总",
        "",
        f"- 总实验行数：`{len(rows)}`",
    ]
    for ctype in ["mask_only", "inpaint_only", "full_pipeline"]:
        lines.append(f"- `{ctype}`：`{len(by_type.get(ctype, []))}`")
    lines.extend(
        [
            "",
            "## 完整表",
            "",
            "| sequence | method_id | ver | mask_protocol | type | status | JM/mask_JM | PSNR_proxy | SSIM | baseline | next_decision | 结论 |",
            "|---|---|---|---|---|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["sequence"],
                    f"`{row['method_id']}`",
                    row.get("version", "legacy"),
                    row.get("mask_protocol", ""),
                    row["comparison_type"],
                    row["audit_status"],
                    row["JM_or_mask_JM"],
                    row["PSNR_proxy"],
                    row["SSIM"],
                    row.get("baseline", ""),
                    row.get("next_decision", "").replace("|", "/"),
                    row["current_takeaway"].replace("|", "/"),
                ]
            )
            + " |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(rows: List[Dict[str, str]]) -> None:
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_markdown(rows)
    write_json(rows)
    print(f"[done] {OUT_CSV}")
    print(f"[done] {OUT_MD}")
    print(f"[done] {OUT_JSON}")


if __name__ == "__main__":
    main()
