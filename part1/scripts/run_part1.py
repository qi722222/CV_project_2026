"""
run_part1.py
------------
Part 1 主串联脚本：帧序列 → mask → 修复视频

用法：
    python run_part1.py --video_dir /data2/shared/project3/bmx-trees \
                        --dataset   bmx-trees \
                        --output    /data2/jli657/project3/part1/outputs/bmx-trees.mp4

    # 或用 --classes 直接指定（不依赖 prompts.json）
    python run_part1.py --video_dir /data2/shared/project3/tennis \
                        --classes person \
                        --output /data2/jli657/project3/part1/outputs/tennis.mp4

流程：
    帧序列
       │
       ▼
    YOLOv8-Seg（按类别过滤）
       │
       ▼
    Lucas-Kanade 判动（过滤静态物体）
       │
       ▼
    cv2.dilate 膨胀
       │
       ▼
    时域平滑补漏检
       │
       ├── 保存 masks_cache/ (可选)
       │
       ▼
    时序背景传播 inpaint
       │  （窗口内能借 → 中位数合并）
       │  （借不到 → cv2.inpaint Telea fallback）
       │
       ▼
    输出 mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 把 scripts/ 加入路径，方便直接 import
sys.path.insert(0, str(Path(__file__).parent))

from gen_masks_yolo import generate_masks
from inpaint_temporal import (
    load_frame_sequence,
    load_mask_sequence,
    temporal_propagation_inpaint,
    temporal_propagation_inpaint_chunked,
    save_video,
)


# ──────────────────────────────────────────
# 默认配置
# ──────────────────────────────────────────

DEFAULT_PROMPTS = {
    "bmx-trees": ["person", "bicycle"],
    "tennis":    ["person"],
    "wild":      ["person"],
}


def load_prompts(prompts_file, dataset_name, cli_classes):
    """
    优先用 CLI 的 --classes，其次查 prompts.json，最后 fallback 到默认配置。
    """
    if cli_classes:
        return cli_classes
    if prompts_file and Path(prompts_file).exists():
        data = json.loads(Path(prompts_file).read_text())
        if dataset_name and dataset_name in data:
            return data[dataset_name]
    if dataset_name and dataset_name in DEFAULT_PROMPTS:
        print(f"[run_part1] Using default classes for '{dataset_name}': "
              f"{DEFAULT_PROMPTS[dataset_name]}")
        return DEFAULT_PROMPTS[dataset_name]
    print("[run_part1] WARNING: No class info found, defaulting to ['person']")
    return ["person"]


# ──────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────

def run_part1(
    video_dir,
    output_path,
    dataset_name=None,
    classes=None,
    prompts_file="prompts.json",
    masks_dir=None,
    model_path="yolov8x-seg.pt",
    dilate_kernel=9,
    motion_threshold=2.0,
    conf_threshold=0.3,
    temporal_window=15,
    inpaint_radius=3,
    fps=30,
    save_masks=True,
    chunked=False,
    chunk_size=60,
    skip_mask_gen=False,
):
    video_dir = Path(video_dir)
    output_path = Path(output_path)

    # ── 1. 决定 mask 存放位置 ──
    if masks_dir is None:
        masks_dir = output_path.parent.parent / "masks_cache" / (dataset_name or video_dir.name)
    masks_dir = Path(masks_dir)

    # ── 2. 读帧 ──
    print(f"\n{'='*60}")
    print(f"[run_part1] Video dir  : {video_dir}")
    print(f"[run_part1] Output     : {output_path}")
    print(f"[run_part1] Masks dir  : {masks_dir}")
    print(f"{'='*60}\n")

    frames_list, stems = load_frame_sequence(str(video_dir))
    T = len(frames_list)
    print(f"[run_part1] Loaded {T} frames.")

    # ── 3. 生成 mask（可跳过，直接读已有 mask）──
    if skip_mask_gen and masks_dir.exists():
        print(f"[run_part1] Skipping mask generation, loading from {masks_dir}")
        masks_arr = load_mask_sequence(str(masks_dir), stems)
    else:
        target_classes = load_prompts(prompts_file, dataset_name, classes)
        print(f"[run_part1] Target classes: {target_classes}")
        masks_arr, _ = generate_masks(
            video_dir=str(video_dir),
            output_dir=str(masks_dir),
            class_names=target_classes,
            dilate_kernel=dilate_kernel,
            motion_threshold=motion_threshold,
            conf_threshold=conf_threshold,
            model_path=model_path,
            temporal_smooth=True,
        )

    masked_frames = (masks_arr > 0).sum(axis=(1, 2))
    print(f"[run_part1] Frames with mask: {(masked_frames > 0).sum()}/{T}")

    # ── 4. 修复 ──
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if chunked:
        # 直接走分块 pipeline（从磁盘读帧，内存友好）
        from inpaint_temporal import temporal_propagation_inpaint_chunked
        temporal_propagation_inpaint_chunked(
            frames_dir=str(video_dir),
            masks_dir=str(masks_dir),
            output_path=str(output_path),
            window=temporal_window,
            inpaint_radius=inpaint_radius,
            chunk_size=chunk_size,
            fps=fps,
        )
    else:
        frames_arr = np.array(frames_list, dtype=np.uint8)
        result = temporal_propagation_inpaint(
            frames_arr, masks_arr,
            window=temporal_window,
            inpaint_radius=inpaint_radius,
        )
        save_video(result, str(output_path), fps=fps)

    print(f"\n[run_part1] ✅ Done! Output: {output_path}\n")


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part 1: YOLOv8-Seg + LK Flow + Temporal Inpainting pipeline"
    )

    # 输入输出
    parser.add_argument("--video_dir", required=True,
                        help="Input frame folder (images sorted by name)")
    parser.add_argument("--output", required=True,
                        help="Output mp4 path")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name (for prompts lookup, e.g. bmx-trees)")
    parser.add_argument("--masks_dir", default=None,
                        help="Where to save/load masks. Default: outputs/../masks_cache/<dataset>")

    # 类别
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Override class names, e.g. --classes person bicycle")
    parser.add_argument("--prompts", default="prompts.json",
                        help="Path to prompts.json")

    # YOLO
    parser.add_argument("--model", default="yolov8x-seg.pt",
                        help="YOLOv8-seg model path/name")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO confidence threshold")

    # Mask 后处理
    parser.add_argument("--dilate_kernel", type=int, default=9,
                        help="Dilation kernel size (increase to cover motion blur)")
    parser.add_argument("--motion_threshold", type=float, default=2.0,
                        help="LK motion threshold in pixels")

    # Inpainting
    parser.add_argument("--window", type=int, default=15,
                        help="Temporal window for background borrowing (each side)")
    parser.add_argument("--inpaint_radius", type=int, default=3,
                        help="cv2.inpaint radius for fallback")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS")

    # 模式
    parser.add_argument("--skip_mask_gen", action="store_true",
                        help="Skip YOLO mask generation (use existing masks in masks_dir)")
    parser.add_argument("--chunked", action="store_true",
                        help="Use memory-friendly chunked inpainting (for long videos)")
    parser.add_argument("--chunk_size", type=int, default=60,
                        help="Chunk size for chunked mode")

    args = parser.parse_args()

    run_part1(
        video_dir=args.video_dir,
        output_path=args.output,
        dataset_name=args.dataset,
        classes=args.classes,
        prompts_file=args.prompts,
        masks_dir=args.masks_dir,
        model_path=args.model,
        dilate_kernel=args.dilate_kernel,
        motion_threshold=args.motion_threshold,
        conf_threshold=args.conf,
        temporal_window=args.window,
        inpaint_radius=args.inpaint_radius,
        fps=args.fps,
        skip_mask_gen=args.skip_mask_gen,
        chunked=args.chunked,
        chunk_size=args.chunk_size,
    )
