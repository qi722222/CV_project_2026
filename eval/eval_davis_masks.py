"""
DAVIS maskpolicy

:
1) eval/davis_eval_targets.yamlGTmask
2) : IoU_mean, J_mean(IoU_mean), JR@tau, F_mean
3) CSV

:
- GTDAVIS Annotations
- eval_mode:
  - union_all_instances: GT
  - instance_ids: [1, 2, ...] ID
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import yaml
except ImportError as exc:
    raise RuntimeError(
        "PyYAML: pip install pyyaml"
    ) from exc


@dataclass
class SequenceResult:
    sequence_name: str
    num_frames: int
    iou_mean: float
    jr_at_tau: float
    f_mean: float
    eval_mode: str
    pred_subdir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAVIS mask")
    parser.add_argument(
        "--policy",
        default="eval/davis_eval_targets.yaml",
        help="policy YAML",
    )
    parser.add_argument(
        "--output_csv",
        default="eval/results_davis_masks.csv",
        help="CSV",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--output_json",
        default="",
        help="JSON",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"policy: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("policy: dict")
    return data


def sorted_pngs(folder: Path) -> List[Path]:
    return sorted(p for p in folder.glob("*.png"))


def read_palette_mask(path: Path) -> np.ndarray:
    # DAVIS annotationPNGPILID
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # : RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def read_pred_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"mask: {path}")
    if arr.shape != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr


def gt_to_binary(gt_index_mask: np.ndarray, seq_cfg: Dict) -> np.ndarray:
    mode = seq_cfg.get("eval_mode", "union_all_instances")
    if mode == "union_all_instances":
        return (gt_index_mask > 0).astype(np.uint8)

    instance_ids = seq_cfg.get("instance_ids")
    if not isinstance(instance_ids, list) or not instance_ids:
        raise ValueError(
            f"{seq_cfg.get('sequence_name')} : eval_mode=instance_idsinstance_ids"
        )
    out = np.zeros_like(gt_index_mask, dtype=np.uint8)
    for iid in instance_ids:
        out = np.logical_or(out, gt_index_mask == int(iid))
    return out.astype(np.uint8)


def to_binary_pred(pred_mask: np.ndarray) -> np.ndarray:
    return (pred_mask > 0).astype(np.uint8)


def calc_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = mask_u8 - eroded
    return boundary


def calc_boundary_f(pred: np.ndarray, gt: np.ndarray, tolerance_px: int) -> float:
    b_pred = mask_boundary(pred)
    b_gt = mask_boundary(gt)

    if b_pred.sum() == 0 and b_gt.sum() == 0:
        return 1.0
    if b_pred.sum() == 0 or b_gt.sum() == 0:
        return 0.0

    k = max(1, int(tolerance_px))
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    b_pred_d = cv2.dilate(b_pred, kernel, iterations=1)
    b_gt_d = cv2.dilate(b_gt, kernel, iterations=1)

    pred_match = np.logical_and(b_pred > 0, b_gt_d > 0).sum()
    gt_match = np.logical_and(b_gt > 0, b_pred_d > 0).sum()

    precision = pred_match / max(1, int((b_pred > 0).sum()))
    recall = gt_match / max(1, int((b_gt > 0).sum()))

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_sequence(
    seq_cfg: Dict,
    gt_root: Path,
    pred_root: Path,
    tau: float,
    tolerance_px: int,
    strict: bool,
) -> Optional[SequenceResult]:
    seq_name = seq_cfg["sequence_name"]
    pred_subdir = seq_cfg.get("pred_subdir", seq_name)
    gt_dir = gt_root / seq_name
    pred_dir = pred_root / pred_subdir

    if not gt_dir.exists():
        msg = f"[skip] GT: {gt_dir}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    if not pred_dir.exists():
        msg = f"[skip] : {pred_dir}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
        return None

    gt_files = sorted_pngs(gt_dir)
    pred_files = sorted_pngs(pred_dir)
    if not gt_files:
        msg = f"[skip] GTpng: {gt_dir}"
        if strict:
            raise ValueError(msg)
        print(msg)
        return None
    if not pred_files:
        msg = f"[skip] png: {pred_dir}"
        if strict:
            raise ValueError(msg)
        print(msg)
        return None

    gt_map = {p.stem: p for p in gt_files}
    pred_map = {p.stem: p for p in pred_files}
    common_stems = sorted(set(gt_map.keys()) & set(pred_map.keys()))

    if not common_stems:
        msg = f"[skip] : seq={seq_name}"
        if strict:
            raise ValueError(msg)
        print(msg)
        return None

    if strict and len(common_stems) != len(gt_files):
        raise ValueError(
            f"{seq_name} : GT={len(gt_files)} ={len(common_stems)}"
        )

    ious: List[float] = []
    fs: List[float] = []
    for stem in common_stems:
        gt_raw = read_palette_mask(gt_map[stem])
        gt_bin = gt_to_binary(gt_raw, seq_cfg)
        pred_raw = read_pred_mask(pred_map[stem], gt_bin.shape)
        pred_bin = to_binary_pred(pred_raw)

        iou = calc_iou(pred_bin, gt_bin)
        f = calc_boundary_f(pred_bin, gt_bin, tolerance_px=tolerance_px)
        ious.append(iou)
        fs.append(f)

    ious_np = np.array(ious, dtype=np.float64)
    fs_np = np.array(fs, dtype=np.float64)

    return SequenceResult(
        sequence_name=seq_name,
        num_frames=len(common_stems),
        iou_mean=float(ious_np.mean()),
        jr_at_tau=float((ious_np >= tau).mean()),
        f_mean=float(fs_np.mean()),
        eval_mode=str(seq_cfg.get("eval_mode", "union_all_instances")),
        pred_subdir=str(pred_subdir),
    )


def write_csv(path: Path, rows: Iterable[SequenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence_name",
                "num_frames",
                "iou_mean",
                "J_mean",
                "JR_at_tau",
                "F_mean",
                "eval_mode",
                "pred_subdir",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.sequence_name,
                    r.num_frames,
                    f"{r.iou_mean:.6f}",
                    f"{r.iou_mean:.6f}",
                    f"{r.jr_at_tau:.6f}",
                    f"{r.f_mean:.6f}",
                    r.eval_mode,
                    r.pred_subdir,
                ]
            )


def print_summary(results: List[SequenceResult]) -> None:
    if not results:
        print("[summary] ")
        return
    iou = np.array([r.iou_mean for r in results], dtype=np.float64)
    jr = np.array([r.jr_at_tau for r in results], dtype=np.float64)
    f = np.array([r.f_mean for r in results], dtype=np.float64)
    frames = np.array([r.num_frames for r in results], dtype=np.float64)

    print("\n=== DAVIS ===")
    print(f": {len(results)}")
    print(f": {int(frames.sum())}")
    print(f"Macro IoU/J: {iou.mean():.4f}")
    print(f"Macro JR:    {jr.mean():.4f}")
    print(f"Macro F:     {f.mean():.4f}")


def build_summary_dict(results: List[SequenceResult]) -> Dict:
    if not results:
        return {
            "num_sequences": 0,
            "num_frames": 0,
            "macro_iou_j": None,
            "macro_jr": None,
            "macro_f": None,
            "sequences": [],
        }
    iou = np.array([r.iou_mean for r in results], dtype=np.float64)
    jr = np.array([r.jr_at_tau for r in results], dtype=np.float64)
    f = np.array([r.f_mean for r in results], dtype=np.float64)
    frames = np.array([r.num_frames for r in results], dtype=np.float64)
    return {
        "num_sequences": int(len(results)),
        "num_frames": int(frames.sum()),
        "macro_iou_j": float(iou.mean()),
        "macro_jr": float(jr.mean()),
        "macro_f": float(f.mean()),
        "sequences": [
            {
                "sequence_name": r.sequence_name,
                "num_frames": r.num_frames,
                "iou_j_mean": r.iou_mean,
                "jr_at_tau": r.jr_at_tau,
                "f_mean": r.f_mean,
                "eval_mode": r.eval_mode,
                "pred_subdir": r.pred_subdir,
            }
            for r in results
        ],
    }


def main() -> None:
    args = parse_args()
    policy = load_yaml(Path(args.policy))
    defaults = policy.get("defaults", {})
    seqs = policy.get("sequences", [])
    if not isinstance(seqs, list) or not seqs:
        raise ValueError("policysequences")

    gt_root = Path(defaults.get("gt_root", ""))
    pred_root = Path(defaults.get("pred_root", ""))
    tau = float(defaults.get("iou_threshold_for_jr", 0.5))
    tolerance_px = int(defaults.get("boundary_tolerance_px", 2))
    strict = bool(defaults.get("strict_missing_predictions", False)) or args.strict

    if not gt_root.exists():
        raise FileNotFoundError(f"GT: {gt_root}")
    if not pred_root.exists():
        print(f"[warn] : {pred_root}skip")

    results: List[SequenceResult] = []
    for seq_cfg in seqs:
        if "sequence_name" not in seq_cfg:
            raise ValueError("sequencesequence_name")
        ret = evaluate_sequence(
            seq_cfg=seq_cfg,
            gt_root=gt_root,
            pred_root=pred_root,
            tau=tau,
            tolerance_px=tolerance_px,
            strict=strict,
        )
        if ret is not None:
            results.append(ret)
            print(
                f"[ok] {ret.sequence_name:16s} "
                f"frames={ret.num_frames:4d} "
                f"IoU/J={ret.iou_mean:.4f} JR={ret.jr_at_tau:.4f} F={ret.f_mean:.4f}"
            )

    out_csv = Path(args.output_csv)
    write_csv(out_csv, results)
    print(f"\n[save] CSV: {out_csv}")
    print_summary(results)
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_summary_dict(results)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[save] JSON: {out_json}")


if __name__ == "__main__":
    main()
