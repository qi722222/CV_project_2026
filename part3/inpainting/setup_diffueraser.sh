#!/usr/bin/env bash
# =============================================================================
# setup_diffueraser.sh — DiffuEraser 环境与权重一键配置
#
# 用法：bash part3/inpainting/setup_diffueraser.sh
#
# 执行步骤：
#   1. 克隆 DiffuEraser 仓库到 /data3/jli657/project3/part3/DiffuEraser_workspace/
#   2. 创建 conda 环境 diffueraser_env (Python 3.9)
#   3. 安装依赖
#   4. 下载 HuggingFace 权重 lixiaowen/diffuEraser
#   5. （可选）下载 SD v1.5 minimal (仅需 scheduler / text_encoder / tokenizer)
#
# 注意：
#   - 权重约 4-8 GB，首次下载需要时间
#   - 若已有 diffueraser_env 环境，脚本会直接用已有环境
#   - SD1.5 minimal 路径可与 controlnet_env 共享，如已存在会跳过
# =============================================================================
set -euo pipefail

WORKSPACE="/data3/jli657/project3/part3/DiffuEraser_workspace"
REPO_DIR="${WORKSPACE}/DiffuEraser"
ENV_NAME="diffueraser_env"
WEIGHTS_DIR="${REPO_DIR}/weights"
HF_MODEL="lixiaowen/diffuEraser"
SD15_DIR="${WEIGHTS_DIR}/stable-diffusion-v1-5"

echo "=== Step 1: Clone DiffuEraser ==="
mkdir -p "$WORKSPACE"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/lixiaowen-xw/DiffuEraser.git "$REPO_DIR"
    echo "[ok] cloned to $REPO_DIR"
else
    echo "[skip] already cloned"
    cd "$REPO_DIR" && git pull --ff-only || echo "[warn] git pull failed, continuing"
fi

echo ""
echo "=== Step 2: Create conda env ==="
if conda info --envs | grep -q "^${ENV_NAME}"; then
    echo "[skip] env ${ENV_NAME} already exists"
else
    conda create -y -n "$ENV_NAME" python=3.9.19
    echo "[ok] created ${ENV_NAME}"
fi

echo ""
echo "=== Step 3: Install requirements ==="
conda run -n "$ENV_NAME" pip install -r "${REPO_DIR}/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu121
# Ensure basic video/image tools available
conda run -n "$ENV_NAME" pip install opencv-python-headless imageio[ffmpeg] tqdm

echo ""
echo "=== Step 4: Download DiffuEraser weights ==="
mkdir -p "$WEIGHTS_DIR"
DIFFUERASER_DIR="${WEIGHTS_DIR}/diffuEraser"
if [ -d "$DIFFUERASER_DIR" ] && [ -f "${DIFFUERASER_DIR}/config.json" ]; then
    echo "[skip] DiffuEraser weights already downloaded"
else
    conda run -n "$ENV_NAME" python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${HF_MODEL}',
    local_dir='${DIFFUERASER_DIR}',
    ignore_patterns=['*.bin.index.json'],
)
print('[ok] downloaded to ${DIFFUERASER_DIR}')
"
fi

echo ""
echo "=== Step 5: Prepare SD v1.5 minimal ==="
# Check if controlnet_env already has SD1.5; if so, symlink to avoid double download
CONTROLNET_SD15="/data2/jli657/envs/controlnet_data/stable-diffusion-v1-5"
if [ -d "$CONTROLNET_SD15" ]; then
    echo "[skip] reusing SD1.5 from controlnet: $CONTROLNET_SD15"
    ln -sfn "$CONTROLNET_SD15" "$SD15_DIR" 2>/dev/null || true
elif [ -d "$SD15_DIR" ] && [ -f "${SD15_DIR}/model_index.json" ]; then
    echo "[skip] SD1.5 already present"
else
    echo "[downloading] SD1.5 minimal (scheduler/text_encoder/tokenizer/feature_extractor)..."
    conda run -n "$ENV_NAME" python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='runwayml/stable-diffusion-v1-5',
    local_dir='${SD15_DIR}',
    allow_patterns=[
        'feature_extractor/**',
        'model_index.json',
        'safety_checker/**',
        'scheduler/**',
        'text_encoder/**',
        'tokenizer/**',
    ],
)
print('[ok] SD1.5 minimal downloaded')
"
fi

echo ""
echo "=== Setup Complete ==="
echo "Repo:     $REPO_DIR"
echo "Env:      $ENV_NAME"
echo "Weights:  $WEIGHTS_DIR"
echo ""
echo "Next: bash part3/inpainting/prepare_diffueraser_inputs.sh tennis"
echo "Then: bash part3/inpainting/run_diffueraser_gtmask.sh tennis v1"
