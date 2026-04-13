"""
compare.py
----------
生成三列并排对比图（原帧 / Part1 / Part2）以及 Part1 vs Part2 的 mask IoU 分析。

用法：
    # 对比图（从视频文件读帧）
    python compare.py compare \
        --orig_dir    /data2/shared/project3/bmx-trees \
        --part1_video /data2/jli657/project3/part1/outputs/bmx-trees.mp4 \
        --part2_video /data2/jli657/project3/part2/outputs/bmx-trees.mp4 \
        --output_dir  /data2/jli657/project3/part1/compare_figs \
        --dataset     bmx-trees \
        --frame_ids   10 20 30

    # IoU 分析（mask 文件夹）
    python compare.py iou \
        --part1_masks /data2/jli657/project3/part1/masks_cache/bmx-trees \
        --part2_masks /data2/jli657/project3/part2/masks_cache/bmx-trees \
        --output_dir  /data2/jli657/project3/part1/compare_figs \
        --dataset     bmx-trees
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt


# ──────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────

def read_frame_from_video(video_path, frame_idx):
    """从 mp4 文件读指定帧（0-indexed）"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def read_frame_from_dir(frames_dir, stem):
    """从帧文件夹读一帧"""
    d = Path(frames_dir)
    for ext in [".jpg", ".jpeg", ".png"]:
        p = d / (stem + ext)
        if p.exists():
            return cv2.imread(str(p))
    raise FileNotFoundError(f"Frame {stem} not found in {frames_dir}")


def get_sorted_stems(frames_dir):
    d = Path(frames_dir)
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p.stem for p in d.iterdir() if p.suffix.lower() in exts])


# ──────────────────────────────────────────
# 对比图
# ──────────────────────────────────────────

def make_comparison_figure(orig_dir, part1_video, part2_video,
                            output_dir, dataset="dataset", frame_ids=None):
    """
    为每个 frame_id 生成一张三列并排对比图。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = get_sorted_stems(orig_dir)
    T = len(stems)

    if frame_ids is None:
        # 自动挑帧：头尾和中间各一帧
        frame_ids = [T // 4, T // 2, 3 * T // 4]
    frame_ids = [f for f in frame_ids if f < T]

    for fid in frame_ids:
        stem = stems[fid]
        orig   = read_frame_from_dir(orig_dir, stem)
        part1  = read_frame_from_video(part1_video, fid)
        part2  = read_frame_from_video(part2_video, fid)

        # BGR → RGB
        orig_rgb  = cv2.cvtColor(orig,  cv2.COLOR_BGR2RGB)
        part1_rgb = cv2.cvtColor(part1, cv2.COLOR_BGR2RGB)
        part2_rgb = cv2.cvtColor(part2, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, img, title in zip(
            axes,
            [orig_rgb, part1_rgb, part2_rgb],
            ["Original", "Part 1 (YOLOv8+LK+Inpaint)", "Part 2 (SAM2+ProPainter)"]
        ):
            ax.imshow(img)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.axis("off")

        fig.suptitle(f"{dataset} — Frame {fid:05d}", fontsize=15)
        plt.tight_layout()
        save_path = output_dir / f"comparison_{dataset}_frame{fid:05d}.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[compare] Saved: {save_path}")


# ──────────────────────────────────────────
# IoU 分析
# ──────────────────────────────────────────

def compute_iou(mask_a, mask_b):
    """计算两个二值 mask 的 IoU"""
    a = mask_a > 0
    b = mask_b > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:
        return 1.0   # 两帧都没有 mask，视为完美一致
    return float(inter) / float(union)


def analyze_mask_iou(part1_masks_dir, part2_masks_dir, output_dir, dataset="dataset"):
    """
    逐帧计算 Part1 mask vs Part2 mask 的 IoU，画折线图并打印统计。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d1 = Path(part1_masks_dir)
    d2 = Path(part2_masks_dir)

    stems1 = sorted([p.stem for p in d1.iterdir() if p.suffix == ".png"])
    stems2 = set(p.stem for p in d2.iterdir() if p.suffix == ".png")
    common = [s for s in stems1 if s in stems2]

    if not common:
        print("[compare] No common mask stems found. Check naming consistency.")
        return

    ious = []
    for stem in common:
        m1 = cv2.imread(str(d1 / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        m2 = cv2.imread(str(d2 / (stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if m1 is None or m2 is None:
            continue
        # Resize to same shape if needed
        if m1.shape != m2.shape:
            m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)
        ious.append(compute_iou(m1, m2))

    ious = np.array(ious)
    print(f"\n[compare] === Mask IoU Analysis: {dataset} ===")
    print(f"  Frames compared : {len(ious)}")
    print(f"  Mean IoU (JM)   : {ious.mean():.4f}")
    print(f"  Recall IoU (JR) : {(ious >= 0.5).mean():.4f}  (fraction with IoU ≥ 0.5)")
    print(f"  Min IoU         : {ious.min():.4f}")
    print(f"  Max IoU         : {ious.max():.4f}")
    print(f"  Std             : {ious.std():.4f}")

    # 折线图
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(ious)), ious, linewidth=1.2, color="steelblue", label="Per-frame IoU")
    ax.axhline(ious.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {ious.mean():.3f}")
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=1.2, label="IoU = 0.5 threshold")
    ax.set_xlabel("Frame index", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title(f"Part1 mask vs Part2 (SAM2) mask IoU — {dataset}", fontsize=13)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_path = output_dir / f"iou_analysis_{dataset}.png"
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[compare] IoU plot saved: {save_path}")

    # 也保存 csv
    csv_path = output_dir / f"iou_{dataset}.csv"
    with open(csv_path, "w") as f:
        f.write("frame_idx,stem,iou\n")
        for i, (s, v) in enumerate(zip(common, ious)):
            f.write(f"{i},{s},{v:.6f}\n")
    print(f"[compare] IoU CSV saved:  {csv_path}")

    return ious


# ──────────────────────────────────────────
# Mask 可视化（叠在原帧上，红色半透明）
# ──────────────────────────────────────────

def visualize_masks(orig_dir, masks_dir, output_dir, dataset="dataset", max_frames=10):
    """
    将 mask 以红色半透明叠加在原帧上，保存可视化图。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = get_sorted_stems(orig_dir)
    T = len(stems)
    step = max(1, T // max_frames)

    for i in range(0, T, step):
        stem = stems[i]
        frame = read_frame_from_dir(orig_dir, stem)
        mask_path = Path(masks_dir) / (stem + ".png")
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[mask > 0] = [0, 0, 255]   # BGR red
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        save_path = output_dir / f"mask_vis_{dataset}_{stem}.jpg"
        cv2.imwrite(str(save_path), vis)

    print(f"[compare] Mask visualizations saved to {output_dir}")


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison figures and IoU analysis")
    sub = parser.add_subparsers(dest="cmd")

    # ── compare subcommand ──
    p_cmp = sub.add_parser("compare", help="Generate 3-column comparison figures")
    p_cmp.add_argument("--orig_dir",    required=True)
    p_cmp.add_argument("--part1_video", required=True)
    p_cmp.add_argument("--part2_video", required=True)
    p_cmp.add_argument("--output_dir",  required=True)
    p_cmp.add_argument("--dataset",     default="dataset")
    p_cmp.add_argument("--frame_ids",   nargs="*", type=int, default=None,
                       help="Frame indices to visualize (0-indexed). Default: auto.")

    # ── iou subcommand ──
    p_iou = sub.add_parser("iou", help="Compute per-frame mask IoU between Part1 and Part2")
    p_iou.add_argument("--part1_masks", required=True)
    p_iou.add_argument("--part2_masks", required=True)
    p_iou.add_argument("--output_dir",  required=True)
    p_iou.add_argument("--dataset",     default="dataset")

    # ── mask_vis subcommand ──
    p_vis = sub.add_parser("mask_vis", help="Visualize masks overlaid on original frames")
    p_vis.add_argument("--orig_dir",   required=True)
    p_vis.add_argument("--masks_dir",  required=True)
    p_vis.add_argument("--output_dir", required=True)
    p_vis.add_argument("--dataset",    default="dataset")
    p_vis.add_argument("--max_frames", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "compare":
        make_comparison_figure(
            orig_dir=args.orig_dir,
            part1_video=args.part1_video,
            part2_video=args.part2_video,
            output_dir=args.output_dir,
            dataset=args.dataset,
            frame_ids=args.frame_ids,
        )
    elif args.cmd == "iou":
        analyze_mask_iou(
            part1_masks_dir=args.part1_masks,
            part2_masks_dir=args.part2_masks,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )
    elif args.cmd == "mask_vis":
        visualize_masks(
            orig_dir=args.orig_dir,
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            max_frames=args.max_frames,
        )
    else:
        parser.print_help()
