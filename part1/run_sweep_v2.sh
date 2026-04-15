#!/bin/bash
# Part 1 sweep v2 · 只用 GPU 1, 2 路并行分 3 批
# 补跑 bmx_A 之外的 5 路(bmx_A 已完成)

set -u
cd /data2/jli657/project3/part1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate part1_env

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs outputs/sweep masks_cache/sweep

run_batch() {
    local batch_name=$1
    shift
    echo "=== $batch_name start: $(date '+%H:%M:%S') ==="
    for cmd in "$@"; do
        eval "$cmd" &
    done
    wait
    echo "=== $batch_name done:  $(date '+%H:%M:%S') ==="
}

# ============ 批次1: bmx_B + tennis_A ============
run_batch "Batch 1" \
"python scripts/run_part1.py \
    --video_dir /data2/shared/project3/bmx-trees \
    --dataset bmx-trees \
    --output outputs/sweep/bmx_B_k13_m25.mp4 \
    --masks_dir masks_cache/sweep/bmx_B \
    --dilate_kernel 13 --motion_threshold 2.5 \
    > logs/bmx_B.log 2>&1" \
"python scripts/run_part1.py \
    --video_dir /data2/shared/project3/tennis \
    --dataset tennis \
    --output outputs/sweep/tennis_A_k13_c02.mp4 \
    --masks_dir masks_cache/sweep/tennis_A \
    --dilate_kernel 13 --conf 0.2 \
    > logs/tennis_A.log 2>&1"

# ============ 批次2: bmx_C + tennis_B ============
run_batch "Batch 2" \
"python scripts/run_part1.py \
    --video_dir /data2/shared/project3/bmx-trees \
    --dataset bmx-trees \
    --output outputs/sweep/bmx_C_k15_m25_w25.mp4 \
    --masks_dir masks_cache/sweep/bmx_C \
    --dilate_kernel 15 --motion_threshold 2.5 --window 25 \
    > logs/bmx_C.log 2>&1" \
"python scripts/run_part1.py \
    --video_dir /data2/shared/project3/tennis \
    --dataset tennis \
    --output outputs/sweep/tennis_B_k15_c015.mp4 \
    --masks_dir masks_cache/sweep/tennis_B \
    --dilate_kernel 15 --conf 0.15 \
    > logs/tennis_B.log 2>&1"

# ============ 批次3: tennis_C (单独, 剩下就一个) ============
run_batch "Batch 3" \
"python scripts/run_part1.py \
    --video_dir /data2/shared/project3/tennis \
    --dataset tennis \
    --output outputs/sweep/tennis_C_k15_c02_w20.mp4 \
    --masks_dir masks_cache/sweep/tennis_C \
    --dilate_kernel 15 --conf 0.2 --window 20 \
    > logs/tennis_C.log 2>&1"

echo ""
echo "=== ALL DONE: $(date '+%H:%M:%S') ==="
ls -lh outputs/sweep/
