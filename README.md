# Project 3: Video Object Removal & Inpainting
**AIAA 3201 — Introduction to Computer Vision, Spring 2026**

[English](#english) | [中文](#中文)

---

## English

### Overview
This repository contains the Part 2 pipeline for video object removal using **SAM 2.1** (mask generation) and **ProPainter** (video inpainting).

**Pipeline:**
```
Frame Sequence -> SAM 2.1 -> Mask PNGs -> cv2.dilate -> ProPainter -> Output MP4
```

### Environment Setup

> ⚠️ SAM2 and ProPainter have conflicting dependencies. Use **two separate conda environments**.

#### Environment 1: `sam2_env` (Mask Generation)

```bash
conda create -n sam2_env python=3.10 -y
conda activate sam2_env
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_sam2.txt
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
```

Download SAM 2.1 weights:
```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

#### Environment 2: `propainter_env` (Video Inpainting)

```bash
conda create -n propainter_env python=3.9 -y
conda activate propainter_env
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_propainter.txt
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter && pip install -r requirements.txt
```

ProPainter weights are downloaded automatically on first run.

### Usage

#### Step 1 — Generate Masks with SAM 2.1
```bash
conda activate sam2_env
python gen_masks_sam2.py \
    --video    /path/to/frames_folder \
    --output   /path/to/masks_cache/dataset_name \
    --sam2_dir /path/to/sam2 \
    --point_x  220 --point_y 120
```

#### Step 2 — Run ProPainter
```bash
conda activate propainter_env
python run_propainter.py \
    --video          /path/to/frames_folder \
    --masks          /path/to/masks_cache/dataset_name \
    --output         /path/to/outputs/dataset_name \
    --propainter_dir /path/to/ProPainter
```

Optional flags:
- `--dilate_kernel 13` — increase if edge artifacts remain (default: 9)
- `--resize_ratio 0.5` — reduce if GPU OOM

### Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `RuntimeError: Numpy is not available` | numpy>=2.0 | `pip install "numpy<2"` |
| `no images found` | SAM2 only reads JPEG | convert PNG to JPG first |
| ProPainter OOM | not enough VRAM | add `--resize_ratio 0.5` |
| weights download timeout | network issue | `wget --timeout=60 --tries=3 <url>` |
| edge artifacts in output | mask too tight | increase `--dilate_kernel` to 13 or 15 |

### Notes
- Tested on NVIDIA RTX A6000 (49GB VRAM), CUDA Driver 12.8, nvcc 12.4
- numpy must be `<2.0` in both environments
- tennis dataset is PNG format — convert to JPEG before running SAM2

---

## 中文

### 项目概述
本仓库包含视频目标移除的 Part 2 pipeline，使用 **SAM 2.1** 生成 mask，**ProPainter** 进行视频修复。

**流程图：**
```
帧序列 -> SAM 2.1 -> Mask PNG -> cv2.dilate 膨胀 -> ProPainter -> 输出 MP4
```

### 环境搭建

> ⚠️ SAM2 和 ProPainter 依赖冲突，必须使用**两个独立的 conda 环境**。

#### 环境一：`sam2_env`（生成 Mask）

```bash
conda create -n sam2_env python=3.10 -y
conda activate sam2_env
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_sam2.txt
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
```

下载 SAM 2.1 权重：
```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

#### 环境二：`propainter_env`（视频修复）

```bash
conda create -n propainter_env python=3.9 -y
conda activate propainter_env
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_propainter.txt
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter && pip install -r requirements.txt
```

ProPainter 权重第一次运行时自动下载。

### 使用方法

#### 第一步 — SAM 2.1 生成 Mask
```bash
conda activate sam2_env
python gen_masks_sam2.py \
    --video    /path/to/帧序列文件夹 \
    --output   /path/to/masks_cache/数据集名 \
    --sam2_dir /path/to/sam2 \
    --point_x  220 --point_y 120
```

参数说明：
- `--point_x / --point_y`：第一帧中要删除的目标的点击坐标
- 输出：二值 mask PNG（白色=要修复，黑色=保留）

#### 第二步 — ProPainter 视频修复
```bash
conda activate propainter_env
python run_propainter.py \
    --video          /path/to/帧序列文件夹 \
    --masks          /path/to/masks_cache/数据集名 \
    --output         /path/to/outputs/数据集名 \
    --propainter_dir /path/to/ProPainter
```

可选参数：
- `--dilate_kernel 13` — 边缘残影严重时调大（默认 9，可调到 13 或 15）
- `--resize_ratio 0.5` — 显存不足时降低分辨率

### 数据集

| 数据集 | 帧数 | 分辨率 | 格式 |
|---|---|---|---|
| bmx-trees | 80 | 432x240 | JPEG |
| tennis | 70 | 432x240 | PNG（需转 JPEG）|
| Wild Video | 30+ | — | — |

### 常见问题

| 问题 | 原因 | 解决方法 |
|---|---|---|
| `RuntimeError: Numpy is not available` | numpy>=2.0 | `pip install "numpy<2"` |
| `no images found` | SAM2 只读 JPEG | 先把 PNG 转成 JPG |
| ProPainter OOM | 显存不足 | 加 `--resize_ratio 0.5` |
| 权重下载超时 | 网络问题 | `wget --timeout=60 --tries=3 <url>` |
| 输出视频边缘有残影 | mask 膨胀不够 | 把 `--dilate_kernel` 调到 13 或 15 |

### 注意事项
- 测试环境：NVIDIA RTX A6000（49GB 显存），CUDA 驱动 12.8，nvcc 12.4
- 两个环境的 numpy 都必须 `<2.0`，装完 opencv 后需重新执行 `pip install "numpy<2"`
- tennis 数据集为 PNG 格式，运行 SAM2 前需先转成 JPEG
