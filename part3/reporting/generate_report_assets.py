"""
generate_report_assets.py — Phase 5: 报告素材自动生成

产出:
  1. qualitative_comparison/ — 每序列一张图: [masked_in | part2 | dir_a | dir_c_best]
  2. quantitative_table.csv / .tex — LaTeX 表格
  3. video_package/ — 按报告顺序命名的 mp4 文件

用法:
  conda run -n controlnet_env python3 part3/generate_report_assets.py
  conda run -n controlnet_env python3 part3/generate_report_assets.py --seqs tennis koala
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

RESULTS         = Path("/data3/jli657/project3/part3/results")
REPORT_ASSETS   = Path("/home/jli657/my_storage2_1T/project3/report_assets")
DAVIS_FRAMES    = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
WILD_FRAMES     = Path("/data3/jli657/project3/wild_frames")

ALL_SEQUENCES = [
    "tennis", "koala", "wild_video-1person",
    "bmx-trees", "blackswan", "horsejump-low", "car-shadow"
]

WILD_MAP = {"wild_video-1person": WILD_FRAMES / "wild_video-1person"}

# Method display order and labels
METHODS = [
    ("masked_in",        "Masked Input"),
    ("part2_baseline",   "Part2 (YOLO+SAM2)"),
    ("part3_dir_a",      "Part3 Dir-A (SAM3)"),
    ("part3_sdxl_kf5",   "Dir-C SDXL kf5"),
    ("part3_lama",       "Dir-C LaMa kf5"),
]

METHOD_MP4_MAP = {
    "masked_in":      None,  # special: read from direction_a
    "part2_baseline": "part2_baseline/inpaint_out.mp4",
    "part3_dir_a":    "direction_a/sam3_propainter/inpaint_out.mp4",
    "part3_sdxl_kf5": "direction_c/sdxl_kf5_propainter/inpaint_out.mp4",
    "part3_lama":     "direction_c/lama_propainter/inpaint_out.mp4",
}

METHOD_MASKED_MAP = {
    "masked_in":      "direction_a/sam3_propainter/masked_in.mp4",
    "part2_baseline": "part2_baseline/masked_in.mp4",
    "part3_dir_a":    "direction_a/sam3_propainter/masked_in.mp4",
    "part3_sdxl_kf5": "direction_c/sdxl_kf5_propainter/masked_in.mp4",
    "part3_lama":     "direction_c/lama_propainter/masked_in.mp4",
}


def load_video_frame(mp4_path: Path, frame_idx: int = 10) -> Optional[np.ndarray]:
    """Load a single frame from mp4 for qualitative comparison."""
    if mp4_path is None or not (mp4_path.exists() or mp4_path.is_symlink()):
        return None
    if mp4_path.is_symlink() and not mp4_path.resolve().exists():
        return None
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_idx, total - 1) if total > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def make_qual_comparison(seq: str, out_dir: Path, frame_indices: List[int] = [10, 30]):
    """Generate qualitative comparison grid: selected frames x methods."""
    seq_dir = RESULTS / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in frame_indices:
        cols = []
        labels = []
        for method_key, label in METHODS:
            mp4_rel = METHOD_MASKED_MAP[method_key] if method_key == "masked_in" else METHOD_MP4_MAP[method_key]
            if mp4_rel is None:
                continue
            mp4_path = seq_dir / mp4_rel
            frame = load_video_frame(mp4_path, frame_idx)
            if frame is not None:
                cols.append(frame)
                labels.append(label)

        if not cols:
            print(f"  [SKIP] {seq} frame {frame_idx}: no frames available")
            continue

        # Resize to same height
        H = min(f.shape[0] for f in cols)
        cols_resized = [cv2.resize(f, (int(f.shape[1] * H / f.shape[0]), H)) for f in cols]

        # Add label banners
        labeled = []
        for img, lbl in zip(cols_resized, labels):
            h, w = img.shape[:2]
            banner = np.zeros((35, w, 3), np.uint8)
            cv2.putText(banner, lbl, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            labeled.append(np.vstack([banner, img]))

        grid = np.hstack(labeled)
        out_path = out_dir / f"{seq}_frame{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  [qual] {out_path.name} ({grid.shape[1]}x{grid.shape[0]})")


def generate_latex_table(eval_csv: Path, out_path: Path):
    """Generate LaTeX-ready table from evaluation CSV."""
    if not eval_csv.exists():
        print(f"[SKIP] eval CSV not found: {eval_csv}")
        return

    rows = []
    with open(eval_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Part3 Quantitative Evaluation (PSNR/SSIM)}",
        r"\label{tab:part3_quant}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Sequence & Method & PSNR$_\text{proxy}$ & PSNR$_\text{synth}$ & SSIM \\",
        r"\midrule",
    ]

    last_seq = None
    for row in rows:
        seq = row["sequence"]
        if seq != last_seq and last_seq is not None:
            lines.append(r"\midrule")
        last_seq = seq
        method_display = {
            "part2_baseline": r"Part2 (YOLO+SAM2+PP)",
            "part3_dir_a":    r"Part3 Dir-A (SAM3+PP)",
            "part3_pure_pp":  r"Part3 Pure PP",
            "part3_sdxl_kf5": r"Part3 SDXL kf5+PP",
            "part3_lama":     r"Part3 LaMa kf5+PP",
        }.get(row["method"], row["method"])

        try:
            psnr_p = float(row["PSNR_proxy"])
            psnr_s = float(row["PSNR_synthetic"])
            ssim   = float(row["SSIM"])
            lines.append(
                f"{seq} & {method_display} & {psnr_p:.2f} & {psnr_s:.2f} & {ssim:.4f} \\\\"
            )
        except (ValueError, KeyError):
            continue

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[LaTeX table] saved to {out_path}")


def package_videos(seqs: List[str], out_dir: Path):
    """Copy/symlink key videos to a unified video package directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        ("part2_baseline", "part2_baseline/inpaint_out.mp4"),
        ("dir_a",          "direction_a/sam3_propainter/inpaint_out.mp4"),
        ("dir_c_sdxl",     "direction_c/sdxl_kf5_propainter/inpaint_out.mp4"),
        ("dir_c_lama",     "direction_c/lama_propainter/inpaint_out.mp4"),
        ("masked_in",      "direction_a/sam3_propainter/masked_in.mp4"),
    ]
    for seq in seqs:
        seq_dir = RESULTS / seq
        for method_key, rel_path in methods:
            src = seq_dir / rel_path
            if src.exists() or (src.is_symlink() and src.resolve().exists()):
                dst = out_dir / f"{seq}_{method_key}.mp4"
                if not dst.exists():
                    if src.is_symlink():
                        shutil.copy2(str(src.resolve()), str(dst))
                    else:
                        shutil.copy2(str(src), str(dst))
                    print(f"  [pkg] {dst.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", default=ALL_SEQUENCES)
    parser.add_argument("--frame_indices", nargs="+", type=int, default=[10, 30, 50])
    args = parser.parse_args()

    qual_dir = REPORT_ASSETS / "part3_qualitative"
    pkg_dir  = REPORT_ASSETS / "part3_video_package"
    tex_path = REPORT_ASSETS / "part3_quantitative_table.tex"
    eval_csv = RESULTS / "evaluation_summary.csv"

    print("=== Phase 5: Report Assets Generation ===\n")

    print("1. Qualitative comparison frames...")
    for seq in args.seqs:
        print(f"  {seq}:")
        make_qual_comparison(seq, qual_dir / seq, args.frame_indices[:2])

    print("\n2. LaTeX table...")
    generate_latex_table(eval_csv, tex_path)

    print("\n3. Video package...")
    package_videos(args.seqs, pkg_dir)

    print("\n[Phase 5 complete]")
    print(f"  Qual images: {qual_dir}")
    print(f"  LaTeX table: {tex_path}")
    print(f"  Video pkg:   {pkg_dir}")


if __name__ == "__main__":
    main()
