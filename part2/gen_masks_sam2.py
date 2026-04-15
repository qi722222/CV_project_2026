"""
gen_masks_sam2.py
-----------------
用 SAM2 对视频帧序列生成逐帧 mask。

用法:
    python gen_masks_sam2.py \
        --video   /data2/shared/project3/bmx-trees \
        --output  /data2/jli657/project3/part2/masks_cache/bmx-trees \
        --sam2_dir /data2/jli657/sam2 \
        --yolo_weight /data2/jli657/project3/part1/yolov8x-seg.pt
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
    parser.add_argument("--yolo_weight", default="/data2/jli657/project3/part1/yolov8x-seg.pt", help="YOLO 权重路径")
    parser.add_argument("--config",    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    # 在 parse_args 里加一个参数
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1], help="YOLO 类别 ID，默认 [0,1] 即 person+bicycle；tennis 用 [0]")
    parser.add_argument("--conf", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()

    os.chdir(args.sam2_dir)
    sys.path.insert(0, args.sam2_dir)

    from sam2.build_sam import build_sam2_video_predictor
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM2] 使用设备: {device}")

    predictor = build_sam2_video_predictor(
        args.config,
        args.checkpoint,
        device=device,
    )
    print("[SAM2] 模型加载成功！")

    # 加载 YOLO 模型
    yolo_model = YOLO(args.yolo_weight)
    print("[YOLO] 模型加载成功！")

    video_dir = args.video
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    print(f"[SAM2] 共找到 {len(frame_files)} 帧")

    inference_state = predictor.init_state(video_path=video_dir)

    # 对第一帧运行 YOLO 获取 bbox
    first_frame_path = os.path.join(video_dir, frame_files[0])
    results = yolo_model(first_frame_path, classes=args.classes, conf=args.conf)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"[YOLO] 第一帧检测到 {len(boxes)} 个 bbox")

    # 方案 B：每个 bbox 一个 obj_id，分别追踪
    for i, box in enumerate(boxes):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=i + 1,  # 1, 2, 3...
            box=box.astype(np.float32),
        )
    print(f"[SAM2] 第一帧 bbox 设置完成，共 {len(boxes)} 个对象")

    os.makedirs(args.output, exist_ok=True)

    print("[SAM2] 开始传播 mask...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # mask_logits 形状 (num_objs, 1, H, W)
        combined = (mask_logits > 0).any(dim=0).squeeze().cpu().numpy()  # 并集
        mask_img = (combined * 255).astype(np.uint8)
        out_name = f"{frame_idx:05d}.png"
        cv2.imwrite(os.path.join(args.output, out_name), mask_img)

    print(f"[SAM2] 完成！共保存 {len(frame_files)} 帧 mask 到: {args.output}")


if __name__ == "__main__":
    main()
