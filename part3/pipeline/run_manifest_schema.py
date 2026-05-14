"""
run_manifest_schema.py — Part3  manifest  (schema v1)

wrapper  run_manifest.json
 register_manifest()  part3_deliverables

Manifest
-----------------
 (required):
  exp_id          : str  —  ID diffueraser_gtmask_v1
  readable_name   : str  —  "DiffuEraser GT mask v1"
  sequence        : str  — DAVIS  wild  "tennis"
  family          : str  —  "DiffuEraser", "ProPainter", "LaMa+PP"
  comparison_type : str  —  "mask_only" | "inpaint_only" | "full_pipeline"
  audit_status    : str  —  "exploratory" "stable"/"reference"
  version         : str  —  "v1", "v2", "legacy"
  mask_protocol   : str  — "davis_gt" | "sam3_mask" | "wild_existing_mask"
  baseline        : str  —  exp_id "pure_propainter_gtmask"

 (optional, ):
  stage_gate      : str  —
  next_decision   : str  —
  failure_reason  : str  — /superseded

  script_path     : str  —  part3/
  config_path     : str  —
  command         : str  —

  output_dir      : str  —  builder
  inpaint_out     : str  — inpaint_out.mp4
  masked_in       : str  — masked_in.mp4
  mask_frames_dir : str  — mask
  log_path        : str  —

  plain_explanation : str  —
  what_to_check     : str  —
  current_takeaway  : str  —

  metrics         : dict  —  builder  evaluator
    PSNR_proxy    : float
    PSNR_synthetic: float
    SSIM          : float
    JM            : float  ( mask_only / full_pipeline)
    JR            : float
    F             : float


----
-  run_manifest.jsonbuilder  manifest results/
- manifest  exp_id
-  run_manifest.json builder  ExperimentDef
   manifest


----
  #
  from part3.pipeline.run_manifest_schema import write_manifest, register_manifest
  write_manifest(output_dir, manifest_dict)
  register_manifest(output_dir / "run_manifest.json")

  #
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
