from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class FrameMetrics:
    frame_name: str
    jm: float
    jr: float
    psnr_proxy: Optional[float]
    ssim_proxy: Optional[float]
    psnr_synth: float
    ssim_synth: float


@dataclass
class GateFeatures:
    mask_area_mean: float
    mask_area_std: float
    mask_area_cv: float
    boundary_complexity_mean: float
    motion_mean: float
    motion_p95: float
    keyframe_delta_std: float
    keyframe_delta_mean: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ControlNet full-video metrics")
    p.add_argument("--orig_dir", required=True, help="Original frame directory")
    p.add_argument("--mask_dir", required=True, help="Mask directory used by inpainting")
    p.add_argument("--pred_video", default="", help="Predicted output video path")
    p.add_argument("--pred_dir", default="", help="Predicted output frame directory (optional)")
    p.add_argument("--gt_mask_dir", default="", help="GT mask directory for JM/JR (optional)")
    p.add_argument("--output_json", required=True, help="Output summary json")
    p.add_argument("--output_csv", default="", help="Output per-frame csv (optional)")
    p.add_argument("--jr_threshold", type=float, default=0.5, help="JR threshold on IoU")
    p.add_argument("--method_name", default="", help="Method tag used in gate log")
    p.add_argument("--gate_log_json", default="", help="Output gate decision json (optional)")
    p.add_argument(
        "--gate_motion_p95_threshold",
        type=float,
        default=18.0,
        help="Gate threshold: motion p95 above this prefers pure propainter",
    )
    p.add_argument(
        "--gate_boundary_complexity_threshold",
        type=float,
        default=0.14,
        help="Gate threshold: boundary complexity above this prefers pure propainter",
    )
    p.add_argument(
        "--gate_mask_area_cv_threshold",
        type=float,
        default=0.65,
        help="Gate threshold: mask area coefficient of variation above this prefers pure propainter",
    )
    p.add_argument(
        "--gate_keyframe_delta_std_threshold",
        type=float,
        default=6.0,
        help="Gate threshold: keyframe delta std above this prefers pure propainter",
    )
    p.add_argument(
        "--gate_use_percentile_calibration",
        action="store_true",
        help="Use feature-percentile calibration to replace default gate thresholds",
    )
    return p.parse_args()


def sorted_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def ensure_pred_frames(pred_video: Path, pred_dir: Path) -> Path:
    if pred_dir.exists() and any(pred_dir.glob("*")):
        return pred_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(pred_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open pred video: {pred_video}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(pred_dir / f"{idx:05d}.png"), frame)
        idx += 1
    cap.release()
    return pred_dir


def read_mask(mask_path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def calc_boundary_complexity(mask_bin: np.ndarray) -> float:
    area = float((mask_bin > 0).sum())
    if area <= 1.0:
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    grad = cv2.morphologyEx((mask_bin * 255).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    perimeter = float((grad > 0).sum())
    return perimeter / area


def calc_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.percentile(diff, 95))


def calc_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    inter = np.logical_and(pred_bin > 0, gt_bin > 0).sum()
    union = np.logical_or(pred_bin > 0, gt_bin > 0).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def calc_psnr(a: np.ndarray, b: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Optional[float]:
    if valid_mask is None:
        diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
        mse = float(diff.mean())
    else:
        m = valid_mask.astype(bool)
        if m.sum() == 0:
            return None
        diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
        mse = float(diff[m].mean())
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def calc_ssim_gray(x: np.ndarray, y: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Optional[float]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if valid_mask is not None:
        m = valid_mask.astype(bool)
        if m.sum() == 0:
            return None
        xv = x[m]
        yv = y[m]
    else:
        xv = x.reshape(-1)
        yv = y.reshape(-1)

    ux = float(xv.mean())
    uy = float(yv.mean())
    vx = float(((xv - ux) ** 2).mean())
    vy = float(((yv - uy) ** 2).mean())
    vxy = float(((xv - ux) * (yv - uy)).mean())

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    if den == 0:
        return None
    return float(num / den)


def find_matching_pred(pred_map: Dict[str, Path], stem: str, idx: int) -> Optional[Path]:
    if stem in pred_map:
        return pred_map[stem]
    idx_stem = f"{idx:05d}"
    return pred_map.get(idx_stem)


def build_gate_features(
    mask_areas: List[float],
    boundary_scores: List[float],
    motion_scores: List[float],
    keyframe_delta_scores: List[float],
) -> GateFeatures:
    def mean_or_zero(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def std_or_zero(vals: List[float]) -> float:
        return float(np.std(vals)) if vals else 0.0

    area_mean = mean_or_zero(mask_areas)
    area_std = std_or_zero(mask_areas)
    area_cv = area_std / max(area_mean, 1e-6)
    return GateFeatures(
        mask_area_mean=area_mean,
        mask_area_std=area_std,
        mask_area_cv=area_cv,
        boundary_complexity_mean=mean_or_zero(boundary_scores),
        motion_mean=mean_or_zero(motion_scores),
        motion_p95=float(np.percentile(motion_scores, 95)) if motion_scores else 0.0,
        keyframe_delta_std=std_or_zero(keyframe_delta_scores),
        keyframe_delta_mean=mean_or_zero(keyframe_delta_scores),
    )


def gate_recommendation_from_thresholds(
    features: GateFeatures,
    motion_p95_thr: float,
    boundary_thr: float,
    mask_area_cv_thr: float,
    keyframe_delta_std_thr: float,
) -> Dict[str, object]:
    reasons: List[str] = []
    if features.motion_p95 >= float(motion_p95_thr):
        reasons.append("high_motion_p95")
    if features.boundary_complexity_mean >= float(boundary_thr):
        reasons.append("complex_mask_boundary")
    if features.mask_area_cv >= float(mask_area_cv_thr):
        reasons.append("unstable_mask_area")
    if features.keyframe_delta_std >= float(keyframe_delta_std_thr):
        reasons.append("unstable_keyframe_delta")

    recommended = "pure_propainter" if reasons else "hybrid_temporal_consistency"
    return {
        "recommended_method": recommended,
        "risk_reasons": reasons,
        "is_high_risk": bool(reasons),
    }


def calibrate_gate_thresholds(features: GateFeatures) -> Dict[str, float]:
    # Single-sequence lightweight calibration: loosen fixed defaults by feature-aware scaling.
    return {
        "motion_p95": max(18.0, float(features.motion_p95) * 1.05),
        "boundary_complexity_mean": max(0.14, float(features.boundary_complexity_mean) * 1.08),
        "mask_area_cv": max(0.65, float(features.mask_area_cv) * 1.1),
        "keyframe_delta_std": max(6.0, float(features.keyframe_delta_std) * 1.08),
    }


def main() -> None:
    args = parse_args()
    orig_dir = Path(args.orig_dir)
    mask_dir = Path(args.mask_dir)
    gt_mask_dir = Path(args.gt_mask_dir) if args.gt_mask_dir else mask_dir
    output_json = Path(args.output_json)

    if not orig_dir.exists():
        raise FileNotFoundError(f"orig_dir missing: {orig_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir missing: {mask_dir}")
    if not gt_mask_dir.exists():
        raise FileNotFoundError(f"gt_mask_dir missing: {gt_mask_dir}")

    pred_dir = Path(args.pred_dir) if args.pred_dir else (output_json.parent / "_tmp_pred_frames")
    if args.pred_video:
        pred_dir = ensure_pred_frames(Path(args.pred_video), pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred_dir missing: {pred_dir}")

    orig_files = sorted_images(orig_dir)
    pred_files = sorted_images(pred_dir)
    if not orig_files or not pred_files:
        raise RuntimeError("No original or predicted frames found")

    pred_map = {p.stem: p for p in pred_files}
    frame_rows: List[FrameMetrics] = []
    mask_areas: List[float] = []
    boundary_scores: List[float] = []
    motion_scores: List[float] = []
    keyframe_delta_scores: List[float] = []
    prev_orig_gray: Optional[np.ndarray] = None

    for i, orig_path in enumerate(orig_files):
        pred_path = find_matching_pred(pred_map, orig_path.stem, i)
        if pred_path is None:
            continue

        orig = cv2.imread(str(orig_path))
        pred = cv2.imread(str(pred_path))
        if orig is None or pred is None:
            continue
        if pred.shape[:2] != orig.shape[:2]:
            pred = cv2.resize(pred, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

        m_inpaint = read_mask(mask_dir / f"{orig_path.stem}.png", orig.shape[:2])
        m_gt = read_mask(gt_mask_dir / f"{orig_path.stem}.png", orig.shape[:2])
        mask_areas.append(float(m_inpaint.mean()))
        boundary_scores.append(calc_boundary_complexity(m_inpaint))

        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        if prev_orig_gray is not None:
            motion_scores.append(calc_motion_score(prev_orig_gray, orig_gray))
        prev_orig_gray = orig_gray

        jm = calc_iou(m_inpaint, m_gt)
        jr = 1.0 if jm >= float(args.jr_threshold) else 0.0

        valid_non_mask = (m_inpaint == 0)
        psnr_proxy = calc_psnr(orig, pred, valid_non_mask)
        ssim_proxy = calc_ssim_gray(
            cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY),
            valid_non_mask,
        )

        # Synthetic clean GT: remove foreground from original frame using GT mask.
        synth = cv2.inpaint(orig, (m_gt * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        psnr_synth = calc_psnr(synth, pred) or 0.0
        ssim_synth = calc_ssim_gray(
            cv2.cvtColor(synth, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY),
            None,
        ) or 0.0

        m_bool = m_inpaint.astype(bool)
        if m_bool.sum() > 0:
            delta = np.abs(pred.astype(np.float32) - orig.astype(np.float32))
            keyframe_delta_scores.append(float(delta[m_bool].mean()))

        frame_rows.append(
            FrameMetrics(
                frame_name=orig_path.name,
                jm=jm,
                jr=jr,
                psnr_proxy=psnr_proxy,
                ssim_proxy=ssim_proxy,
                psnr_synth=psnr_synth,
                ssim_synth=ssim_synth,
            )
        )

    if not frame_rows:
        raise RuntimeError("No matched frames for evaluation")

    def avg(vals: List[Optional[float]]) -> Optional[float]:
        v = [x for x in vals if x is not None]
        if not v:
            return None
        return float(np.mean(v))

    summary = {
        "num_frames_eval": len(frame_rows),
        "mask_quality": {
            "JM": float(np.mean([r.jm for r in frame_rows])),
            "JR": float(np.mean([r.jr for r in frame_rows])),
        },
        "video_quality_proxy": {
            "PSNR": avg([r.psnr_proxy for r in frame_rows]),
            "SSIM": avg([r.ssim_proxy for r in frame_rows]),
        },
        "video_quality_synthetic": {
            "PSNR": float(np.mean([r.psnr_synth for r in frame_rows])),
            "SSIM": float(np.mean([r.ssim_synth for r in frame_rows])),
        },
    }

    gate_features = build_gate_features(mask_areas, boundary_scores, motion_scores, keyframe_delta_scores)
    default_thresholds = {
        "motion_p95": float(args.gate_motion_p95_threshold),
        "boundary_complexity_mean": float(args.gate_boundary_complexity_threshold),
        "mask_area_cv": float(args.gate_mask_area_cv_threshold),
        "keyframe_delta_std": float(args.gate_keyframe_delta_std_threshold),
    }
    default_decision = gate_recommendation_from_thresholds(
        gate_features,
        motion_p95_thr=default_thresholds["motion_p95"],
        boundary_thr=default_thresholds["boundary_complexity_mean"],
        mask_area_cv_thr=default_thresholds["mask_area_cv"],
        keyframe_delta_std_thr=default_thresholds["keyframe_delta_std"],
    )

    calibrated_thresholds = calibrate_gate_thresholds(gate_features)
    calibrated_decision = gate_recommendation_from_thresholds(
        gate_features,
        motion_p95_thr=calibrated_thresholds["motion_p95"],
        boundary_thr=calibrated_thresholds["boundary_complexity_mean"],
        mask_area_cv_thr=calibrated_thresholds["mask_area_cv"],
        keyframe_delta_std_thr=calibrated_thresholds["keyframe_delta_std"],
    )
    selected_thresholds = calibrated_thresholds if args.gate_use_percentile_calibration else default_thresholds
    selected_decision = calibrated_decision if args.gate_use_percentile_calibration else default_decision

    gate_payload = {
        "method_name": args.method_name or "unknown_method",
        "features": {
            "mask_area_mean": gate_features.mask_area_mean,
            "mask_area_std": gate_features.mask_area_std,
            "mask_area_cv": gate_features.mask_area_cv,
            "boundary_complexity_mean": gate_features.boundary_complexity_mean,
            "motion_mean": gate_features.motion_mean,
            "motion_p95": gate_features.motion_p95,
            "keyframe_delta_mean": gate_features.keyframe_delta_mean,
            "keyframe_delta_std": gate_features.keyframe_delta_std,
        },
        "policy_thresholds": selected_thresholds,
        "policy_threshold_source": "calibrated" if args.gate_use_percentile_calibration else "default",
        "decision": selected_decision,
        "decision_compare": {
            "default_thresholds": default_thresholds,
            "default_decision": default_decision,
            "calibrated_thresholds": calibrated_thresholds,
            "calibrated_decision": calibrated_decision,
        },
    }
    summary["gate_policy"] = gate_payload

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.gate_log_json:
        out_gate = Path(args.gate_log_json)
        out_gate.parent.mkdir(parents=True, exist_ok=True)
        with out_gate.open("w", encoding="utf-8") as f:
            json.dump(gate_payload, f, ensure_ascii=False, indent=2)

    if args.output_csv:
        import csv

        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "frame_name",
                    "jm",
                    "jr",
                    "psnr_proxy",
                    "ssim_proxy",
                    "psnr_synth",
                    "ssim_synth",
                ]
            )
            for r in frame_rows:
                w.writerow(
                    [
                        r.frame_name,
                        f"{r.jm:.6f}",
                        f"{r.jr:.6f}",
                        "" if r.psnr_proxy is None else f"{r.psnr_proxy:.6f}",
                        "" if r.ssim_proxy is None else f"{r.ssim_proxy:.6f}",
                        f"{r.psnr_synth:.6f}",
                        f"{r.ssim_synth:.6f}",
                    ]
                )

    print(f"[save] summary: {output_json}")
    if args.gate_log_json:
        print(f"[save] gate log: {args.gate_log_json}")
    if args.output_csv:
        print(f"[save] per-frame: {args.output_csv}")


if __name__ == "__main__":
    main()
