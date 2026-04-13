"""
gen_masks_yolo.py
-----------------
用 YOLOv8-Seg 提取 mask，Lucas-Kanade 光流判动，cv2.dilate 膨胀。
输出：masks_cache/<dataset>/<00000.png> 格式的二值 mask。

用法:
    python gen_masks_yolo.py \
        --video_dir /data2/shared/project3/bmx-trees \
        --output_dir /data2/jli657/project3/part1/masks_cache/bmx-trees \
        --classes person bicycle \
        --dilate_kernel 9 \
        --motion_threshold 2.0
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# COCO 类别名称 → ID 映射（常用）
COCO_NAME2ID = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "bus": 5, "truck": 7, "sports ball": 32, "tennis racket": 38,
}


# ──────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────

def load_frames(video_dir: str):
    """
    从帧序列文件夹读取所有图片，按文件名排序。
    返回 list of (filename_stem, BGR ndarray)
    """
    video_dir = Path(video_dir)
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"No image files found in {video_dir}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f"Failed to read {p}")
        frames.append((p.stem, img))
    return frames


def get_class_ids(class_names):
    """把类别名列表转成 COCO ID 列表"""
    ids = []
    for name in class_names:
        name = name.lower()
        if name not in COCO_NAME2ID:
            raise ValueError(f"Unknown class '{name}'. Known: {list(COCO_NAME2ID.keys())}")
        ids.append(COCO_NAME2ID[name])
    return ids


# ──────────────────────────────────────────
# Lucas-Kanade 判动
# ──────────────────────────────────────────

def estimate_global_homography(prev_gray, curr_gray):
    """
    估计两帧间的全局单应性（用于补偿相机运动）。
    返回 3x3 矩阵 H，若估计失败返回 None。
    """
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def compensate_motion(displacement, H, pt):
    """
    用单应性矩阵补偿相机运动后的净位移。
    pt: (x, y) 原点坐标
    """
    if H is None:
        return displacement
    pt_h = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pt_h, H)[0][0]
    camera_disp = projected - pt
    return displacement - camera_disp


def is_moving(prev_gray, curr_gray, binary_mask,
              threshold=2.0, compensate_camera=True):
    """
    判断 binary_mask 区域内的物体是否在运动。
    binary_mask: uint8，非零区域为前景
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=50, qualityLevel=0.01,
        minDistance=5, mask=mask_u8
    )
    if pts is None or len(pts) < 3:
        # 特征点太少，保守地认为在动
        return True

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    valid = status.flatten() == 1
    if valid.sum() < 3:
        return True

    displacements = next_pts[valid] - pts[valid]  # (N, 1, 2)
    disp_norms = np.linalg.norm(displacements.reshape(-1, 2), axis=1)

    if compensate_camera:
        # 估全局单应性来补偿相机运动
        H = estimate_global_homography(prev_gray, curr_gray)
        if H is not None:
            compensated = []
            for i, pt in enumerate(pts[valid].reshape(-1, 2)):
                net = compensate_motion(displacements[i].reshape(2), H, pt)
                compensated.append(np.linalg.norm(net))
            disp_norms = np.array(compensated)

    return float(np.median(disp_norms)) > threshold


# ──────────────────────────────────────────
# Mask 生成主函数
# ──────────────────────────────────────────

def generate_masks(video_dir, output_dir, class_names,
                   dilate_kernel=9, motion_threshold=2.0,
                   conf_threshold=0.3, model_path="yolov8x-seg.pt",
                   temporal_smooth=True):
    """
    主函数：为每帧生成二值 mask，保存为 PNG。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen_masks] Loading frames from {video_dir}")
    frames = load_frames(video_dir)
    T = len(frames)
    print(f"[gen_masks] {T} frames found")

    class_ids = get_class_ids(class_names)
    print(f"[gen_masks] Target classes: {class_names} → COCO IDs: {class_ids}")

    print(f"[gen_masks] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    H, W = frames[0][1].shape[:2]
    raw_masks = np.zeros((T, H, W), dtype=np.uint8)   # 0/255
    stems = [stem for stem, _ in frames]
    bgr_frames = [img for _, img in frames]
    gray_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in bgr_frames]

    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)

    print("[gen_masks] Running YOLO + LK optical flow ...")
    for t in tqdm(range(T)):
        stem, frame = frames[t]
        results = model(frame, classes=class_ids, conf=conf_threshold, verbose=False)

        if results[0].masks is None:
            continue  # 无检测，mask 保持全零

        seg_masks = results[0].masks.data.cpu().numpy()   # (N, H', W')
        seg_classes = results[0].boxes.cls.cpu().numpy().astype(int)  # (N,)

        # 合并所有目标类别的 mask
        combined = np.zeros((H, W), dtype=np.uint8)
        for i, cls_id in enumerate(seg_classes):
            if cls_id not in class_ids:
                continue
            m = seg_masks[i]
            # YOLO mask 可能尺寸和原图不一致，resize 回来
            m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            binary = (m_resized > 0.5).astype(np.uint8)

            # ── Lucas-Kanade 判动 ──
            if t == 0:
                # 第一帧没有前帧，默认认为在动
                moving = True
            else:
                moving = is_moving(
                    gray_frames[t - 1], gray_frames[t],
                    binary, threshold=motion_threshold
                )

            if moving:
                combined = np.maximum(combined, binary)

        # 膨胀
        if combined.any():
            combined = cv2.dilate(combined, kernel, iterations=1)

        raw_masks[t] = (combined > 0).astype(np.uint8) * 255

    # ── 时域平滑补漏检 ──
    if temporal_smooth:
        print("[gen_masks] Applying temporal smoothing ...")
        smoothed = raw_masks.copy()
        for t in range(1, T - 1):
            prev_has = raw_masks[t - 1] > 0
            next_has = raw_masks[t + 1] > 0
            # 前后都有检测但当前帧没有 → 补上（取并集）
            fill = prev_has & next_has & (raw_masks[t] == 0)
            if fill.any():
                union = (raw_masks[t - 1] | raw_masks[t + 1]).astype(np.uint8)
                union = cv2.dilate(union, kernel, iterations=1)
                smoothed[t] = np.maximum(smoothed[t], union & (fill * 255))
        raw_masks = smoothed

    # ── 保存 ──
    print(f"[gen_masks] Saving masks to {output_dir}")
    for t, stem in enumerate(tqdm(stems)):
        out_path = output_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), raw_masks[t])

    print(f"[gen_masks] Done. {T} masks saved.")
    return raw_masks, stems


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks with YOLOv8-Seg + LK optical flow")
    parser.add_argument("--video_dir", required=True, help="Input frame folder")
    parser.add_argument("--output_dir", required=True, help="Output mask folder")
    parser.add_argument("--classes", nargs="+", default=["person"],
                        help="Target class names (COCO), e.g. person bicycle")
    parser.add_argument("--model", default="yolov8x-seg.pt", help="YOLOv8-seg model path")
    parser.add_argument("--dilate_kernel", type=int, default=9, help="Dilation kernel size")
    parser.add_argument("--motion_threshold", type=float, default=2.0,
                        help="LK motion threshold in pixels (higher = only detect fast motion)")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--no_smooth", action="store_true", help="Disable temporal smoothing")
    args = parser.parse_args()

    generate_masks(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        class_names=args.classes,
        dilate_kernel=args.dilate_kernel,
        motion_threshold=args.motion_threshold,
        conf_threshold=args.conf,
        model_path=args.model,
        temporal_smooth=not args.no_smooth,
    )
