"""
gen_masks_sam2.py
-----------------
用 SAM2 对视频帧序列生成逐帧 mask。

用法:
    python gen_masks_sam2.py \
        --video   /data2/shared/project3/bmx-trees \
        --output  /data2/jli657/project3/part2/masks_cache/bmx-trees \
        --sam2_dir /data2/jli657/sam2 \
        --point_x 320 --point_y 240
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 视频逐帧 mask 生成脚本")
    parser.add_argument("--video",     required=True, help="输入帧序列文件夹")
    parser.add_argument("--output",    required=True, help="mask 输出文件夹")
    parser.add_argument("--sam2_dir",  required=True, help="SAM2 仓库根目录")
    parser.add_argument("--point_x",   type=int, required=True, help="第一帧点击的 x 坐标")
    parser.add_argument("--point_y",   type=int, required=True, help="第一帧点击的 y 坐标")
    parser.add_argument("--config",    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(args.sam2_dir)
    sys.path.insert(0, args.sam2_dir)

    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM2] 使用设备: {device}")

    predictor = build_sam2_video_predictor(
        args.config,
        args.checkpoint,
        device=device,
    )
    print("[SAM2] 模型加载成功！")

    video_dir = args.video
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    print(f"[SAM2] 共找到 {len(frame_files)} 帧")

    inference_state = predictor.init_state(video_path=video_dir)

    point = np.array([[args.point_x, args.point_y]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,
    )
    print(f"[SAM2] 第一帧提示点设置完成: ({args.point_x}, {args.point_y})")

    os.makedirs(args.output, exist_ok=True)

    print("[SAM2] 开始传播 mask...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        mask = (mask_logits[0, 0] > 0.0).cpu().numpy()
        mask_img = (mask * 255).astype(np.uint8)
        frame_name = frame_files[frame_idx]
        out_name = Path(frame_name).stem + ".png"
        cv2.imwrite(os.path.join(args.output, out_name), mask_img)

    print(f"[SAM2] 完成！共保存 {len(frame_files)} 帧 mask 到: {args.output}")


if __name__ == "__main__":
    main()
