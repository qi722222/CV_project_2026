#!/usr/bin/env bash
set -euo pipefail

# :
#   part2/gen_masks_sam2.py 3maskDAVIS
#   eval/davis_eval_targets.yaml  >=5
#
# :
#   conda activate sam2_env
#   bash eval/run_missing_davis_masks.sh
#
# :
#   SAM2_DIR, YOLO_WEIGHT, DATA_ROOT, OUT_ROOT, SCRIPT_PATH, CUDA_VISIBLE_DEVICES

SAM2_DIR="${SAM2_DIR:-/data2/jli657/sam2}"
YOLO_WEIGHT="${YOLO_WEIGHT:-/home/jli657/my_storage2_1T/project3/yolov8x-seg.pt}"
DATA_ROOT="${DATA_ROOT:-/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p}"
OUT_ROOT="${OUT_ROOT:-/home/jli657/my_storage2_1T/project3/part2/masks_cache}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/jli657/my_storage2_1T/project3/part2/gen_masks_sam2.py}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_one() {
  local seq="$1"
  shift
  local classes=("$@")
  local out_dir="${OUT_ROOT}/${seq}"
  local video_dir="${DATA_ROOT}/${seq}"

  echo "=================================================="
  echo "[run] seq=${seq}"
  echo "[run] classes=${classes[*]}"
  echo "[run] video_dir=${video_dir}"
  echo "[run] out_dir=${out_dir}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${SCRIPT_PATH}" \
    --video "${video_dir}" \
    --output "${out_dir}" \
    --sam2_dir "${SAM2_DIR}" \
    --yolo_weight "${YOLO_WEIGHT}" \
    --classes "${classes[@]}" \
    --conf 0.3 \
    --config "configs/sam2.1/sam2.1_hiera_l.yaml" \
    --checkpoint "checkpoints/sam2.1_hiera_large.pt"
}

# COCO:
# bird=14, car=2, horse=17
run_one "blackswan" 14
run_one "car-shadow" 2
run_one "horsejump-low" 17

echo "[done] :"
echo "  python eval/eval_davis_masks.py --policy eval/davis_eval_targets.yaml --output_csv eval/results_davis_masks.csv"
