#!/bin/bash
# Part 1 残影优化 · 6 路并行扫参
# bmx-trees: 3 组 (GPU 1/2/3)
# tennis:    3 组 (GPU 4/5/6)
# 完全避开 GPU0 (别人在用)

set -u
cd /data2/jli657/project3/part1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate part1_env

mkdir -p logs outputs/sweep masks_cache/sweep

echo "=== Sweep start: $(date) ==="

# ============ bmx-trees 三组 ============
# A: 基线加大 kernel
CUDA_VISIBLE_DEVICES=1 python scripts/run_part1.py \
    --video_dir /data2/shared/project3/bmx-trees \
    --dataset bmx-trees \
    --output outputs/sweep/bmx_A_k13.mp4 \
    --masks_dir masks_cache/sweep/bmx_A \
    --dilate_kernel 13 \
    > logs/bmx_A.log 2>&1 &
echo "[GPU1] bmx_A  (k=13)                       PID=$!"

# B: kernel + 放宽光流(应对相机运动)
CUDA_VISIBLE_DEVICES=2 python scripts/run_part1.py \
    --video_dir /data2/shared/project3/bmx-trees \
    --dataset bmx-trees \
    --output outputs/sweep/bmx_B_k13_m25.mp4 \
    --masks_dir masks_cache/sweep/bmx_B \
    --dilate_kernel 13 --motion_threshold 2.5 \
    > logs/bmx_B.log 2>&1 &
echo "[GPU2] bmx_B  (k=13, motion=2.5)           PID=$!"

# C: 全开(大 kernel + 宽光流 + 大窗口)
CUDA_VISIBLE_DEVICES=3 python scripts/run_part1.py \
    --video_dir /data2/shared/project3/bmx-trees \
    --dataset bmx-trees \
    --output outputs/sweep/bmx_C_k15_m25_w25.mp4 \
    --masks_dir masks_cache/sweep/bmx_C \
    --dilate_kernel 15 --motion_threshold 2.5 --window 25 \
    > logs/bmx_C.log 2>&1 &
echo "[GPU3] bmx_C  (k=15, motion=2.5, w=25)     PID=$!"

# ============ tennis 三组 ============
# A: kernel + 降置信度(解决漏检)
CUDA_VISIBLE_DEVICES=4 python scripts/run_part1.py \
    --video_dir /data2/shared/project3/tennis \
    --dataset tennis \
    --output outputs/sweep/tennis_A_k13_c02.mp4 \
    --masks_dir masks_cache/sweep/tennis_A \
    --dilate_kernel 13 --conf 0.2 \
    > logs/tennis_A.log 2>&1 &
echo "[GPU4] tennis_A (k=13, conf=0.2)           PID=$!"

# B: 更激进的 conf + 大 kernel
CUDA_VISIBLE_DEVICES=5 python scripts/run_part1.py \
    --video_dir /data2/shared/project3/tennis \
    --dataset tennis \
    --output outputs/sweep/tennis_B_k15_c015.mp4 \
    --masks_dir masks_cache/sweep/tennis_B \
    --dilate_kernel 15 --conf 0.15 \
    > logs/tennis_B.log 2>&1 &
echo "[GPU5] tennis_B (k=15, conf=0.15)          PID=$!"

# C: 大 kernel + 低 conf + 大窗口
CUDA_VISIBLE_DEVICES=6 python scripts/run_part1.py \
    --video_dir /data2/shared/project3/tennis \
    --dataset tennis \
    --output outputs/sweep/tennis_C_k15_c02_w20.mp4 \
    --masks_dir masks_cache/sweep/tennis_C \
    --dilate_kernel 15 --conf 0.2 --window 20 \
    > logs/tennis_C.log 2>&1 &
echo "[GPU6] tennis_C (k=15, conf=0.2, w=20)     PID=$!"

echo "=== 6 jobs launched, waiting for all to finish... ==="
wait
echo "=== Sweep done: $(date) ==="
