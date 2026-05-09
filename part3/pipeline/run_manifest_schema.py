"""
run_manifest_schema.py — Part3 统一实验 manifest 协议 (schema v1)

每次新实验跑完后，wrapper 脚本必须在输出目录下写入 run_manifest.json，
然后调用 register_manifest() 或直接运行本脚本来自动进入 part3_deliverables。

Manifest 字段说明
-----------------
必填字段 (required):
  exp_id          : str  — 实验唯一 ID，例如 diffueraser_gtmask_v1
  readable_name   : str  — 中英文混合可读名，例如 "DiffuEraser GT mask v1"
  sequence        : str  — DAVIS 序列或 wild 视频名，例如 "tennis"
  family          : str  — 方法族，例如 "DiffuEraser", "ProPainter", "LaMa+PP"
  comparison_type : str  — 必须是 "mask_only" | "inpaint_only" | "full_pipeline"
  audit_status    : str  — 初始填 "exploratory"，通过验收后改 "stable"/"reference"
  version         : str  — 例如 "v1", "v2", "legacy"
  mask_protocol   : str  — "davis_gt" | "sam3_mask" | "wild_existing_mask"
  baseline        : str  — 对比基线 exp_id，首版写 "pure_propainter_gtmask"

可选字段 (optional, 有则填):
  stage_gate      : str  — 进入下一阶段门槛描述
  next_decision   : str  — 当前结果后下一步怎么做
  failure_reason  : str  — 失败/superseded 时填写

  script_path     : str  — 运行该实验的脚本路径（相对 part3/ 根）
  config_path     : str  — 配置文件路径（可选）
  command         : str  — 完整可复现命令，包含序列参数

  output_dir      : str  — 输出目录绝对路径（用于 builder 查找产物）
  inpaint_out     : str  — inpaint_out.mp4 绝对路径
  masked_in       : str  — masked_in.mp4 绝对路径
  mask_frames_dir : str  — mask 帧目录绝对路径
  log_path        : str  — 运行日志路径

  plain_explanation : str  — 一段话描述这个实验在比什么
  what_to_check     : str  — 看结果时重点看什么
  current_takeaway  : str  — 跑完后填写当前结论

  metrics         : dict  — 若已计算则直接嵌入（否则 builder 会调用 evaluator）
    PSNR_proxy    : float
    PSNR_synthetic: float
    SSIM          : float
    JM            : float  (仅 mask_only / full_pipeline)
    JR            : float
    F             : float

注意
----
- 实验目录若已有 run_manifest.json，builder 会优先读 manifest，不再扫 results/ 目录。
- 每次调参产生新版本，必须写新目录（不覆盖旧版），manifest 的 exp_id 中体现版本号。
- 未写 run_manifest.json 的实验，builder 仍然可以扫 ExperimentDef 列表，但后续新实验
  必须走 manifest 流程，避免人工补登记。

用法
----
  # 在实验脚本末尾调用
  from part3.pipeline.run_manifest_schema import write_manifest, register_manifest
  write_manifest(output_dir, manifest_dict)
  register_manifest(output_dir / "run_manifest.json")

  # 或单独跑注册
  python3 part3/pipeline/run_manifest_schema.py --manifest path/to/run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REQUIRED_FIELDS = [
    "exp_id", "readable_name", "sequence", "family",
    "comparison_type", "audit_status", "version",
    "mask_protocol", "baseline",
]

VALID_COMPARISON_TYPES = {"mask_only", "inpaint_only", "full_pipeline"}
VALID_AUDIT_STATUSES = {
    "reference", "stable", "promising", "exploratory",
    "legacy", "superseded", "partial_or_failed", "failed", "needs_review",
}

BUILD_SCRIPT = Path(__file__).parent.parent / "reporting" / "build_part3_deliverables.py"


def validate_manifest(manifest: Dict[str, Any]) -> list[str]:
    """Returns a list of validation errors (empty = valid)."""
    errors = []
    for f in REQUIRED_FIELDS:
        if f not in manifest or not str(manifest[f]).strip():
            errors.append(f"Missing required field: {f}")
    ct = manifest.get("comparison_type", "")
    if ct and ct not in VALID_COMPARISON_TYPES:
        errors.append(f"comparison_type must be one of {VALID_COMPARISON_TYPES}, got '{ct}'")
    st = manifest.get("audit_status", "")
    if st and st not in VALID_AUDIT_STATUSES:
        errors.append(f"audit_status must be one of {VALID_AUDIT_STATUSES}, got '{st}'")
    return errors


def write_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    """Write run_manifest.json to output_dir after validation."""
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] written → {manifest_path}")
    return manifest_path


def register_manifest(manifest_path: Path) -> None:
    """
    Call the deliverables builder to ingest a newly written manifest.

    The builder reads run_manifest.json files in addition to ExperimentDef entries;
    this function triggers a targeted rebuild for the affected sequence.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seq = manifest.get("sequence", "")
    if not seq:
        raise ValueError("Manifest missing 'sequence' field, cannot register.")

    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"[register] sequence={seq}  exp_id={manifest.get('exp_id')}")
    if BUILD_SCRIPT.exists():
        result = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--manifest", str(manifest_path)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"[register] builder exited with code {result.returncode}")
        else:
            print("[register] builder completed successfully")
    else:
        print(f"[register] builder script not found at {BUILD_SCRIPT}; "
              "run build_part3_deliverables.py manually to ingest this manifest.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and register a run_manifest.json")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.json")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate; do not trigger builder")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest)
    if errors:
        print("[FAIL] Manifest validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("[OK] Manifest is valid.")

    if not args.validate_only:
        register_manifest(manifest_path)


if __name__ == "__main__":
    main()
