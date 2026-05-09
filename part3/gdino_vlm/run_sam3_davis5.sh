#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/jli657/project3"
PY="/data2/jli657/envs/sam3_env/bin/python"
RUNNER="$ROOT/part3/gdino_vlm/run_gdino_mainline.py"
CFG_S1="$ROOT/part3/configs/gdino_vlm_mainline_sam3.yaml"
CFG_S2="$ROOT/part3/configs/gdino_vlm_mainline_sam3_stage2.yaml"
SEQS=("bmx-trees" "tennis" "blackswan" "car-shadow" "horsejump-low")

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

for seq in "${SEQS[@]}"; do
  echo "[run] SAM3 stage1 seq=${seq}"
  "$PY" "$RUNNER" \
    --config "$CFG_S1" \
    --sequence "$seq" \
    --stage stage1 \
    --output "$ROOT/part3/gdino_vlm/masks/sam3/stage1/$seq"
done

for seq in "${SEQS[@]}"; do
  echo "[run] SAM3 stage2 seq=${seq}"
  "$PY" "$RUNNER" \
    --config "$CFG_S2" \
    --sequence "$seq" \
    --stage stage2 \
    --output "$ROOT/part3/gdino_vlm/masks/sam3/stage2/$seq"
done

echo "[ok] SAM3 DAVIS5 stage1+stage2 done."
