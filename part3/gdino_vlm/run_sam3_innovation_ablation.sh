#!/usr/bin/env bash
set -euo pipefail

ROOT="/data3/jli657/project3"
PY="/data2/jli657/envs/sam3_env/bin/python"
RUNNER="$ROOT/part3/gdino_vlm/run_gdino_mainline.py"
SEQS=("bmx-trees" "tennis")

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

run_cfg () {
  local cfg="$1"
  local tag="$2"
  for seq in "${SEQS[@]}"; do
    echo "[run] innovation=${tag} seq=${seq}"
    "$PY" "$RUNNER" \
      --config "$cfg" \
      --sequence "$seq" \
      --stage stage2 \
      --output "$ROOT/part3/gdino_vlm/masks/sam3/innovation/$tag/$seq"
  done
}

run_cfg "$ROOT/part3/configs/gdino_vlm_sam3_innov_quality_gate.yaml" "quality_gate"
run_cfg "$ROOT/part3/configs/gdino_vlm_sam3_innov_o2o.yaml" "o2o_assoc"
run_cfg "$ROOT/part3/configs/gdino_vlm_sam3_innov_real_vlm.yaml" "real_vlm"

echo "[ok] SAM3 innovation ablation runs done."
