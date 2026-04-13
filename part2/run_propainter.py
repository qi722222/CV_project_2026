"""
run_propainter.py
-----------------
接收 SAM2 生成的 mask 文件夹，膨胀处理后调用 ProPainter 进行视频修复。

用法:
    python run_propainter.py \
        --video   /path/to/frames_folder \
        --masks   /path/to/sam2_masks_folder \
        --output  /path/to/output_folder \
        --propainter_dir /data2/jli657/ProPainter \
        [--dilate_kernel 9] \
        [--resize_ratio 1.0]
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import cv2
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="ProPainter 集成脚本（含 mask 膨胀）")
    parser.add_argument("--video",           required=True,  help="输入帧序列文件夹")
    parser.add_argument("--masks",           required=True,  help="SAM2 输出的 mask 文件夹")
    parser.add_argument("--output",          required=True,  help="最终结果输出文件夹")
    parser.add_argument("--propainter_dir",  required=True,  help="ProPainter 仓库根目录")
    parser.add_argument("--dilate_kernel",   type=int, default=9,
                        help="膨胀 kernel 大小（默认 9，残影严重时调到 13 或 15）")
    parser.add_argument("--resize_ratio",    type=float, default=1.0,
                        help="分辨率缩放比例（显存不足时设 0.5）")
    return parser.parse_args()


def dilate_masks(src_mask_dir: str, dst_mask_dir: str, kernel_size: int):
    os.makedirs(dst_mask_dir, exist_ok=True)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask_files = sorted(Path(src_mask_dir).glob("*.png"))
    if not mask_files:
        mask_files = sorted(Path(src_mask_dir).glob("*.jpg"))
    if not mask_files:
        raise FileNotFoundError(f"在 {src_mask_dir} 中找不到任何 mask 文件！")

    print(f"[膨胀] 共找到 {len(mask_files)} 帧 mask，kernel={kernel_size}x{kernel_size}")
    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ⚠️  读取失败，跳过: {mask_path.name}")
            continue
        dilated = cv2.dilate(mask, kernel, iterations=1)
        cv2.imwrite(os.path.join(dst_mask_dir, mask_path.name), dilated)

    print(f"[膨胀] 完成，膨胀后的 mask 已存到: {dst_mask_dir}")


def run_propainter(propainter_dir: str, video_dir: str,
                   mask_dir: str, output_dir: str, resize_ratio: float):
    script = os.path.join(propainter_dir, "inference_propainter.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"找不到 ProPainter 脚本: {script}")

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, script,
        "--video",        video_dir,
        "--mask",         mask_dir,
        "--output",   output_dir,
        "--resize_ratio", str(resize_ratio),
    ]

    print(f"[ProPainter] 运行命令:\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=propainter_dir)

    if result.returncode != 0:
        print("❌ ProPainter 运行失败，请检查以上报错信息。")
        sys.exit(1)
    else:
        print(f"✅ ProPainter 完成！结果已保存到: {output_dir}")


def main():
    args = parse_args()

    if not os.path.isdir(args.video):
        raise NotADirectoryError(f"帧序列文件夹不存在: {args.video}")
    if not os.path.isdir(args.masks):
        raise NotADirectoryError(f"mask 文件夹不存在: {args.masks}")

    dilated_mask_dir = args.masks.rstrip("/") + "_dilated"

    print("=" * 50)
    print(f"  视频帧目录 : {args.video}")
    print(f"  原始 mask  : {args.masks}")
    print(f"  膨胀 mask  : {dilated_mask_dir}")
    print(f"  输出目录   : {args.output}")
    print(f"  膨胀 kernel: {args.dilate_kernel}x{args.dilate_kernel}")
    print(f"  缩放比例   : {args.resize_ratio}")
    print("=" * 50)

    dilate_masks(args.masks, dilated_mask_dir, args.dilate_kernel)
    run_propainter(
        propainter_dir=args.propainter_dir,
        video_dir=args.video,
        mask_dir=dilated_mask_dir,
        output_dir=args.output,
        resize_ratio=args.resize_ratio,
    )


if __name__ == "__main__":
    main()
