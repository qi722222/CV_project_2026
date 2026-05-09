from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml


SEQ_CONFIGS = {
    "tennis": {
        "base": "part3/configs/controlnet_davis_tennis.yaml",
        "tc": "part3/configs/controlnet_davis_tennis_tc.yaml",
    },
    "bmx-trees": {
        "base": "part3/configs/controlnet_davis_bmx_trees.yaml",
        "tc": "part3/configs/controlnet_davis_bmx_trees_tc.yaml",
    },
    "koala": {
        "base": "part3/configs/controlnet_davis_koala.yaml",
        "tc": "part3/configs/controlnet_davis_koala_tc.yaml",
    },
    "bear": {
        "base": "part3/configs/controlnet_davis_bear.yaml",
        "tc": "part3/configs/controlnet_davis_bear_tc.yaml",
    },
    "camel": {
        "base": "part3/configs/controlnet_davis_camel.yaml",
        "tc": "part3/configs/controlnet_davis_camel_tc.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 5-seq ControlNet ablation with ProPainter")
    p.add_argument(
        "--workspace_root",
        default="/home/jli657/my_storage2_1T/project3",
        help="Project root path",
    )
    p.add_argument(
        "--propainter_dir",
        default="/home/jli657/my_storage2_1T/ProPainter",
        help="ProPainter repository root",
    )
    p.add_argument(
        "--sequences",
        default="tennis,bmx-trees,koala,bear,camel",
        help="Comma-separated sequence list",
    )
    p.add_argument(
        "--refine_python",
        default=sys.executable,
        help="Python executable for refine/eval stage",
    )
    p.add_argument(
        "--propainter_python",
        default=sys.executable,
        help="Python executable for ProPainter stage",
    )
    p.add_argument("--skip_refine", action="store_true", help="Skip run_part3_refine")
    p.add_argument("--skip_propainter", action="store_true", help="Skip ProPainter stage")
    p.add_argument("--skip_eval", action="store_true", help="Skip evaluation stage")
    p.add_argument(
        "--output_tag",
        default="ablation_5seq_tuned",
        help="Output folder name under part3/outputs/controlnet",
    )
    p.add_argument(
        "--summary_prefix",
        default="summary_5seq_tuned",
        help="Prefix for summary csv/json file names",
    )
    p.add_argument("--baseline_neighbor_length", type=int, default=10, help="ProPainter baseline neighbor_length")
    p.add_argument("--baseline_ref_stride", type=int, default=10, help="ProPainter baseline ref_stride")
    p.add_argument("--hybrid_tc_neighbor_length", type=int, default=14, help="ProPainter tuned neighbor_length")
    p.add_argument("--hybrid_tc_ref_stride", type=int, default=8, help="ProPainter tuned ref_stride")
    p.add_argument("--hybrid_tc_mask_open", type=int, default=3, help="Morph open kernel for hybrid_tc mask")
    p.add_argument("--hybrid_tc_mask_close", type=int, default=5, help="Morph close kernel for hybrid_tc mask")
    p.add_argument("--hybrid_tc_mask_dilate", type=int, default=7, help="Extra dilation kernel for hybrid_tc mask")
    p.add_argument(
        "--use_calibrated_gate_thresholds",
        action="store_true",
        help="Enable percentile-calibrated gate thresholds in eval script",
    )
    return p.parse_args()


def read_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: List[str], cwd: Path, extra_env: Dict[str, str] | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    ret = subprocess.run(cmd, cwd=str(cwd), env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code={ret.returncode}: {' '.join(cmd)}")


def build_hybrid_frames(orig_dir: Path, refined_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_frames = sorted([p for p in orig_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for p in orig_frames:
        shutil.copy2(p, out_dir / f"{p.stem}.png")
    refined_frames = sorted([p for p in refined_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for p in refined_frames:
        shutil.copy2(p, out_dir / f"{p.stem}.png")


def materialize_masks_local(mask_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    for p in mask_files:
        shutil.copy2(p, out_dir / p.name)
    return out_dir


def preprocess_masks_for_hybrid_tc(
    src_mask_dir: Path,
    out_dir: Path,
    open_kernel: int,
    close_kernel: int,
    dilate_kernel: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted([p for p in src_mask_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    for p in mask_files:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = np.where(m > 0, 255, 0).astype(np.uint8)
        if open_kernel > 1:
            k = np.ones((open_kernel, open_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if close_kernel > 1:
            k = np.ones((close_kernel, close_kernel), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if dilate_kernel > 1:
            k = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            m = cv2.dilate(m, k, iterations=1)
        cv2.imwrite(str(out_dir / p.name), m)
    return out_dir


def parse_summary(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "JM": float(data["mask_quality"]["JM"]),
        "JR": float(data["mask_quality"]["JR"]),
        "PSNR_proxy": float(data["video_quality_proxy"]["PSNR"] or 0.0),
        "SSIM_proxy": float(data["video_quality_proxy"]["SSIM"] or 0.0),
        "PSNR_synth": float(data["video_quality_synthetic"]["PSNR"]),
        "SSIM_synth": float(data["video_quality_synthetic"]["SSIM"]),
        "recommended_method": str(data.get("gate_policy", {}).get("decision", {}).get("recommended_method", "")),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.workspace_root)
    propainter_dir = Path(args.propainter_dir)
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    if not sequences:
        raise ValueError("No sequences provided")

    all_rows: List[Dict[str, object]] = []
    common_env = {
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HOME": "/data3/jli657/hf_cache",
        "HUGGINGFACE_HUB_CACHE": "/data3/jli657/hf_cache",
    }
    for seq in sequences:
        if seq not in SEQ_CONFIGS:
            raise KeyError(f"Unsupported sequence: {seq}")
        cfg_base = read_yaml(root / SEQ_CONFIGS[seq]["base"])
        cfg_tc = read_yaml(root / SEQ_CONFIGS[seq]["tc"])
        orig_dir = Path(cfg_base["paths"]["input_frames_dir"])
        mask_dir = Path(cfg_base["paths"]["masks_dir"])
        out_base = Path(cfg_base["paths"]["output_dir"])
        out_tc = Path(cfg_tc["paths"]["output_dir"])
        out_ablation = root / "part3" / "outputs" / "controlnet" / args.output_tag / seq
        hybrid_dir = out_ablation / "hybrid_frames"
        hybrid_tc_dir = out_ablation / "hybrid_tc_frames"
        pure_out = out_ablation / "propainter_pure" / seq
        hybrid_out = out_ablation / "propainter_hybrid" / "hybrid_frames"
        hybrid_tc_out = out_ablation / "propainter_hybrid_tc" / "hybrid_tc_frames"
        eval_dir = out_ablation / "eval"
        local_mask_dir = materialize_masks_local(mask_dir=mask_dir, out_dir=out_ablation / "local_masks")
        hybrid_tc_mask_dir = preprocess_masks_for_hybrid_tc(
            src_mask_dir=local_mask_dir,
            out_dir=out_ablation / "local_masks_hybrid_tc",
            open_kernel=max(1, int(args.hybrid_tc_mask_open)),
            close_kernel=max(1, int(args.hybrid_tc_mask_close)),
            dilate_kernel=max(1, int(args.hybrid_tc_mask_dilate)),
        )

        if not args.skip_refine:
            run_cmd(
                [args.refine_python, "part3/run_part3_refine.py", "--config", SEQ_CONFIGS[seq]["base"]],
                cwd=root,
                extra_env=common_env,
            )
            run_cmd(
                [args.refine_python, "part3/run_part3_refine.py", "--config", SEQ_CONFIGS[seq]["tc"]],
                cwd=root,
                extra_env=common_env,
            )

        if not args.skip_propainter:
            build_hybrid_frames(orig_dir=orig_dir, refined_dir=out_base / "refined_keyframes", out_dir=hybrid_dir)
            build_hybrid_frames(orig_dir=orig_dir, refined_dir=out_tc / "refined_keyframes", out_dir=hybrid_tc_dir)

            run_cmd(
                [
                    args.propainter_python,
                    # ProPainter and its dependencies usually live in another env.
                    # Use --propainter_python to point to that interpreter.
                    "part2/run_propainter.py",
                    "--video",
                    str(orig_dir),
                    "--masks",
                    str(local_mask_dir),
                    "--output",
                    str(pure_out.parent),
                    "--propainter_dir",
                    str(propainter_dir),
                    "--dilate_kernel",
                    "9",
                    "--resize_ratio",
                    "1.0",
                    "--neighbor_length",
                    str(args.baseline_neighbor_length),
                    "--ref_stride",
                    str(args.baseline_ref_stride),
                ],
                cwd=root,
                extra_env=common_env,
            )
            run_cmd(
                [
                    args.propainter_python,
                    "part2/run_propainter.py",
                    "--video",
                    str(hybrid_dir),
                    "--masks",
                    str(local_mask_dir),
                    "--output",
                    str(hybrid_out.parent),
                    "--propainter_dir",
                    str(propainter_dir),
                    "--dilate_kernel",
                    "9",
                    "--resize_ratio",
                    "1.0",
                    "--neighbor_length",
                    str(args.baseline_neighbor_length),
                    "--ref_stride",
                    str(args.baseline_ref_stride),
                ],
                cwd=root,
                extra_env=common_env,
            )
            run_cmd(
                [
                    args.propainter_python,
                    "part2/run_propainter.py",
                    "--video",
                    str(hybrid_tc_dir),
                    "--masks",
                    str(hybrid_tc_mask_dir),
                    "--output",
                    str(hybrid_tc_out.parent),
                    "--propainter_dir",
                    str(propainter_dir),
                    "--dilate_kernel",
                    "9",
                    "--resize_ratio",
                    "1.0",
                    "--neighbor_length",
                    str(args.hybrid_tc_neighbor_length),
                    "--ref_stride",
                    str(args.hybrid_tc_ref_stride),
                ],
                cwd=root,
                extra_env=common_env,
            )

        if not args.skip_eval:
            eval_dir.mkdir(parents=True, exist_ok=True)
            methods = {
                "pure_propainter": pure_out / "inpaint_out.mp4",
                "hybrid": hybrid_out / "inpaint_out.mp4",
                "hybrid_temporal_consistency": hybrid_tc_out / "inpaint_out.mp4",
            }
            for method, pred_mp4 in methods.items():
                method_dir = eval_dir / method
                method_dir.mkdir(parents=True, exist_ok=True)
                summary_json = method_dir / "metrics_summary.json"
                per_frame_csv = method_dir / "metrics_per_frame.csv"
                gate_json = method_dir / "gate_log.json"
                run_cmd(
                    [  # keep default thresholds for baseline comparability
                        args.refine_python,
                        "part3/eval_controlnet_video_metrics.py",
                        "--orig_dir",
                        str(orig_dir),
                        "--mask_dir",
                        str(local_mask_dir),
                        "--gt_mask_dir",
                        str(local_mask_dir),
                        "--pred_video",
                        str(pred_mp4),
                        "--pred_dir",
                        str(method_dir / "_pred_frames"),
                        "--method_name",
                        method,
                        "--gate_log_json",
                        str(gate_json),
                        "--output_json",
                        str(summary_json),
                        "--output_csv",
                        str(per_frame_csv),
                    ]
                    + (["--gate_use_percentile_calibration"] if args.use_calibrated_gate_thresholds else []),
                    cwd=root,
                    extra_env=common_env,
                )
                row = {"sequence": seq, "method": method}
                row.update(parse_summary(summary_json))
                all_rows.append(row)

    out_root = root / "part3" / "outputs" / "controlnet" / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / f"{args.summary_prefix}.json"
    out_csv = out_root / f"{args.summary_prefix}.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "method",
                "JM",
                "JR",
                "PSNR_proxy",
                "SSIM_proxy",
                "PSNR_synth",
                "SSIM_synth",
                "recommended_method",
            ],
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"[save] {out_json}")
    print(f"[save] {out_csv}")


if __name__ == "__main__":
    main()
