"""
validate_davis_policy.py
------------------------
 eval/davis_eval_targets.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAVIS policy")
    parser.add_argument(
        "--policy",
        default="eval/davis_eval_targets.yaml",
        help="policy YAML",
    )
    parser.add_argument(
        "--check_paths",
        action="store_true",
        help="GT/predpred",
    )
    return parser.parse_args()


def _require_keys(obj: Dict, keys: List[str], where: str) -> None:
    for key in keys:
        if key not in obj:
            raise ValueError(f"{where} : {key}")


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"policy: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("policy ")
    return data


def validate_schema(policy: Dict) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    try:
        _require_keys(policy, ["defaults", "sequences"], "policy")
    except ValueError as exc:
        errors.append(str(exc))
        return warnings, errors

    defaults = policy["defaults"]
    if not isinstance(defaults, dict):
        errors.append("defaults ")
        return warnings, errors
    for key in ["gt_root", "pred_root", "iou_threshold_for_jr", "boundary_tolerance_px"]:
        if key not in defaults:
            errors.append(f"defaults : {key}")

    seqs = policy["sequences"]
    if not isinstance(seqs, list) or not seqs:
        errors.append("sequences ")
        return warnings, errors

    seen = set()
    for i, seq in enumerate(seqs):
        where = f"sequences[{i}]"
        if not isinstance(seq, dict):
            errors.append(f"{where} ")
            continue
        try:
            _require_keys(seq, ["sequence_name", "eval_mode", "pred_subdir", "prompt_text_for_gdino"], where)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        name = str(seq["sequence_name"])
        if name in seen:
            errors.append(f"sequence_name : {name}")
        seen.add(name)

        mode = seq["eval_mode"]
        if mode not in {"union_all_instances", "instance_ids"}:
            errors.append(f"{where}.eval_mode : {mode}")
        if mode == "instance_ids":
            ids = seq.get("instance_ids")
            if not isinstance(ids, list) or not ids:
                errors.append(f"{where}.instance_ids ")

        k = seq.get("gdino_reanchor_stride", None)
        if k is not None:
            try:
                k_val = int(k)
                if k_val <= 0:
                    errors.append(f"{where}.gdino_reanchor_stride  > 0")
            except Exception:
                errors.append(f"{where}.gdino_reanchor_stride null")

    if len(seqs) < 5:
        warnings.append("sequences 5 run-davis-5seq ")

    return warnings, errors


def validate_paths(policy: Dict) -> List[str]:
    msgs: List[str] = []
    defaults = policy["defaults"]
    gt_root = Path(defaults["gt_root"])
    pred_root = Path(defaults["pred_root"])
    if not gt_root.exists():
        msgs.append(f"[error] gt_root : {gt_root}")
    if not pred_root.exists():
        msgs.append(f"[warn] pred_root : {pred_root}")

    for seq in policy["sequences"]:
        name = seq["sequence_name"]
        pred_subdir = seq["pred_subdir"]
        gt_dir = gt_root / name
        pred_dir = pred_root / pred_subdir
        if not gt_dir.exists():
            msgs.append(f"[error] GT: {gt_dir}")
        if not pred_dir.exists():
            msgs.append(f"[warn] : {pred_dir}")
    return msgs


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy)
    policy = load_yaml(policy_path)
    warnings, errors = validate_schema(policy)

    if warnings:
        for w in warnings:
            print(f"[warn] {w}")
    if errors:
        for e in errors:
            print(f"[error] {e}")
        raise SystemExit(1)

    print(f"[ok] schema: {policy_path}")
    if args.check_paths:
        for msg in validate_paths(policy):
            print(msg)


if __name__ == "__main__":
    main()
