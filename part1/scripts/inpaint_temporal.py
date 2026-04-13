"""
inpaint_temporal.py
-------------------
时序背景传播 + cv2.inpaint fallback。

核心思路：
  对每帧中 mask 标记的像素，优先从时间窗口内的"干净帧"借背景像素（取中位数，抗噪）；
  若整个窗口都借不到，则 fallback 到 cv2.inpaint (Telea)。

用法（单独调用）：
    python inpaint_temporal.py \
        --frames_dir /data2/shared/project3/bmx-trees \
        --masks_dir  /data2/jli657/project3/part1/masks_cache/bmx-trees \
        --output     /data2/jli657/project3/part1/outputs/bmx-trees.mp4 \
        --window 15 \
        --inpaint_radius 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────
# I/O 工具
# ──────────────────────────────────────────

def load_frame_sequence(frames_dir):
    """读帧序列，返回 sorted list of BGR ndarray"""
    d = Path(frames_dir)
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in d.iterdir() if p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"No images in {frames_dir}")
    return [cv2.imread(str(p)) for p in paths], [p.stem for p in paths]


def load_mask_sequence(masks_dir, stems):
    """按 stems 顺序读 mask，返回 (T, H, W) uint8 数组"""
    d = Path(masks_dir)
    masks = []
    for stem in stems:
        # 尝试 .png 和 .jpg
        for ext in [".png", ".jpg"]:
            p = d / (stem + ext)
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                masks.append(m if m is not None else np.zeros_like(masks[-1] if masks else (480, 640)))
                break
        else:
            # mask 不存在 → 全零（不修复）
            h, w = masks[0].shape if masks else (480, 640)
            masks.append(np.zeros((h, w), dtype=np.uint8))
    return np.array(masks, dtype=np.uint8)


def save_video(frames_array, output_path, fps=30):
    """
    frames_array: list or (T, H, W, 3) ndarray, BGR uint8
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frames_array, np.ndarray):
        frames_list = [frames_array[i] for i in range(len(frames_array))]
    else:
        frames_list = frames_array

    h, w = frames_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames_list:
        writer.write(f)
    writer.release()
    print(f"[inpaint] Video saved: {output_path}")


# ──────────────────────────────────────────
# 核心算法
# ──────────────────────────────────────────

def temporal_propagation_inpaint(frames, masks, window=15, inpaint_radius=3):
    """
    时序背景传播 + cv2.inpaint fallback。

    参数：
        frames  : (T, H, W, 3) uint8  BGR
        masks   : (T, H, W)    uint8  255=需要修复, 0=保留
        window  : 前后各借多少帧
        inpaint_radius: cv2.inpaint 的修复半径（小纹理用3，大区域用5~7）

    返回：
        result  : (T, H, W, 3) uint8
    """
    if isinstance(frames, list):
        frames = np.array(frames, dtype=np.uint8)

    T, H, W, C = frames.shape
    result = frames.copy()

    # 预先把 mask 归一化成 bool (255 → True)
    masks_bool = masks > 0   # (T, H, W) bool

    print(f"[inpaint] Running temporal propagation: {T} frames, window={window} ...")
    for t in tqdm(range(T)):
        need_fix = masks_bool[t]          # (H, W) bool
        if not need_fix.any():
            continue

        lo = max(0, t - window)
        hi = min(T, t + window + 1)

        window_frames = frames[lo:hi]          # (Wt, H, W, 3)
        window_bg = ~masks_bool[lo:hi]         # (Wt, H, W) True=背景可借

        # 哪些像素在窗口里至少一帧是干净背景
        any_bg = window_bg.any(axis=0)         # (H, W)

        # 只对需要修复且有可借像素的位置取中位数
        borrow_where = need_fix & any_bg       # (H, W)

        fixed = result[t].copy()

        if borrow_where.any():
            # 构建 masked array：屏蔽掉"不是背景"的帧位置
            # window_frames: (Wt, H, W, 3)
            # window_bg:     (Wt, H, W)  → expand to (Wt, H, W, 3)
            bg_expanded = np.broadcast_to(
                window_bg[..., None], window_frames.shape
            )
            mf = np.ma.masked_array(window_frames, mask=~bg_expanded)
            # 沿时间轴取中位数
            borrowed = np.ma.median(mf, axis=0).filled(0).astype(np.uint8)  # (H, W, 3)
            fixed[borrow_where] = borrowed[borrow_where]

        # fallback：窗口内借不到的区域 → cv2.inpaint
        fallback_pixels = need_fix & ~any_bg   # (H, W)
        if fallback_pixels.any():
            fallback_mask = fallback_pixels.astype(np.uint8) * 255
            fixed = cv2.inpaint(fixed, fallback_mask, inpaint_radius, cv2.INPAINT_TELEA)

        result[t] = fixed

    print("[inpaint] Done.")
    return result


# ──────────────────────────────────────────
# 分块处理版本（内存友好，适合长视频）
# ──────────────────────────────────────────

def temporal_propagation_inpaint_chunked(frames_dir, masks_dir, output_path,
                                          window=15, inpaint_radius=3,
                                          chunk_size=60, fps=30):
    """
    分块处理版本，避免长视频一次性加载 OOM。
    每块 = chunk_size 帧，块间重叠 window 帧保证边界处理正确。
    """
    frames_list, stems = load_frame_sequence(frames_dir)
    masks_arr = load_mask_sequence(masks_dir, stems)
    T = len(frames_list)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frames_arr = np.array(frames_list, dtype=np.uint8)

    start = 0
    while start < T:
        # 块的实际范围（含 overlap）
        lo_ctx = max(0, start - window)
        hi_ctx = min(T, start + chunk_size + window)

        chunk_frames = frames_arr[lo_ctx:hi_ctx]
        chunk_masks  = masks_arr[lo_ctx:hi_ctx]

        result_chunk = temporal_propagation_inpaint(
            chunk_frames, chunk_masks, window=window, inpaint_radius=inpaint_radius
        )

        # 只写出属于 [start, start+chunk_size) 的帧
        write_lo = start - lo_ctx
        write_hi = min(start + chunk_size, T) - lo_ctx
        for i in range(write_lo, write_hi):
            writer.write(result_chunk[i])

        start += chunk_size

    writer.release()
    print(f"[inpaint] Chunked video saved: {output_path}")


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal background propagation inpainting")
    parser.add_argument("--frames_dir", required=True, help="Input frame folder (BGR images)")
    parser.add_argument("--masks_dir",  required=True, help="Mask folder (PNG, 255=inpaint)")
    parser.add_argument("--output",     required=True, help="Output mp4 path")
    parser.add_argument("--window",      type=int,   default=15,  help="Temporal window (frames each side)")
    parser.add_argument("--inpaint_radius", type=int, default=3,  help="cv2.inpaint radius for fallback")
    parser.add_argument("--fps",         type=int,   default=30,  help="Output video FPS")
    parser.add_argument("--chunked",     action="store_true",     help="Use memory-friendly chunked mode")
    parser.add_argument("--chunk_size",  type=int,   default=60,  help="Chunk size for chunked mode")
    args = parser.parse_args()

    if args.chunked:
        temporal_propagation_inpaint_chunked(
            frames_dir=args.frames_dir,
            masks_dir=args.masks_dir,
            output_path=args.output,
            window=args.window,
            inpaint_radius=args.inpaint_radius,
            chunk_size=args.chunk_size,
            fps=args.fps,
        )
    else:
        frames_list, stems = load_frame_sequence(args.frames_dir)
        masks_arr = load_mask_sequence(args.masks_dir, stems)
        frames_arr = np.array(frames_list, dtype=np.uint8)

        result = temporal_propagation_inpaint(
            frames_arr, masks_arr,
            window=args.window,
            inpaint_radius=args.inpaint_radius,
        )
        save_video(result, args.output, fps=args.fps)
