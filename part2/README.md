# Part 2 — SOTA Pipeline (SAM 2.1 + ProPainter)

基于基础模型的视频目标移除。项目总览见 [`../README.md`](../README.md)，经典 baseline 见 [`../part1/README.md`](../part1/README.md)。

---

## 项目概述

Pipeline：
```
帧序列
    │
    ▼
YOLOv8x-Seg（仅第一帧）─► 各类物体的 bounding box
    │
    ▼
SAM 2.1（视频追踪）
    │  - 每个 bbox 作为一个 obj_id 的 prompt
    │  - SAM2 在所有帧上传播像素级精度的 mask
    │  - 多物体 mask 按帧取并集
    ▼
Mask PNG 序列（00000.png, 00001.png, ...）
    │
    ▼
ProPainter（双域光流 + Transformer 融合）
    │
    ▼
输出 MP4
```

**关键设计**：我们用 **YOLO 第一帧的 bbox 检测**作为 SAM2 的 prompt，不用手工点击。这样：
1. 无需人工标注
2. 与 Part 1 在语义上对齐（删的都是同样的目标类别）
3. 避免 SAM2 point prompt 的歧义（比如点在自行车上不会自动包括骑手）

---

## 环境搭建

> ⚠️ SAM2 和 ProPainter 依赖冲突，必须用**两个独立 conda 环境**。

### 环境一：`sam2_env`（生成 Mask）

```bash
conda create -n sam2_env python=3.10 -y
conda activate sam2_env

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_sam2.txt
pip install ultralytics --no-deps
pip install pandas seaborn

git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e .
```

下载 SAM 2.1 权重：
```bash
cd sam2/checkpoints
bash download_ckpts.sh
# 或者手动:
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 环境二：`propainter_env`（视频修复）

```bash
conda create -n propainter_env python=3.10 -y
conda activate propainter_env

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_propainter.txt

git clone https://github.com/sczhou/ProPainter.git
cd ProPainter && pip install -r requirements.txt
```

ProPainter 权重首次运行时自动下载。

---

## 使用方法

### 第一步 — YOLO + SAM 2.1 生成 Mask

```bash
conda activate sam2_env

CUDA_VISIBLE_DEVICES=1 python gen_masks_sam2.py \
    --video        /path/to/帧序列文件夹 \
    --output       /path/to/masks_cache/<数据集>_v3 \
    --sam2_dir     /path/to/sam2 \
    --yolo_weight  /path/to/yolov8x-seg.pt \
    --classes      0 1 \
    --conf         0.3 \
    --config       configs/sam2.1/sam2.1_hiera_l.yaml \
    --checkpoint   checkpoints/sam2.1_hiera_large.pt
```

#### 各数据集对应的类别 ID（COCO）

| 数据集 | `--classes` | 删什么 |
|---|---|---|
| `bmx-trees` | `0 1` | 人 + 自行车 |
| `tennis` | `0 38` | 人 + 网球拍 |
| Wild Video | `0` | 人 |
| `blackswan`（DAVIS）| `14` | 鸟 |
| `car-shadow`（DAVIS）| `2` | 车 |

COCO 类别速查：`person=0`、`bicycle=1`、`car=2`、`motorcycle=3`、`bird=14`、`cat=15`、`dog=16`、`horse=17`、`sports ball=32`、`tennis racket=38`。

如果 YOLO 在第一帧漏检某个目标，把 `--conf` 降到 `0.2` 或 `0.15`。

### 第二步 — ProPainter 视频修复

```bash
conda deactivate
conda activate propainter_env
cd /path/to/ProPainter

CUDA_VISIBLE_DEVICES=1 python inference_propainter.py \
    --video  /path/to/帧序列文件夹 \
    --mask   /path/to/masks_cache/<数据集>_v3 \
    --output /path/to/outputs/<数据集>_v3
```

可选参数：
- `--resize_ratio 0.5` — 显存不足时降分辨率
- `--neighbor_length 10` — 局部参考帧窗口
- `--ref_stride 10` — 全局参考帧步长

---

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `FileNotFoundError: sam2.1_hiera_tiny.pt` | 默认 checkpoint 不匹配 | 显式传 `--checkpoint checkpoints/sam2.1_hiera_large.pt` |
| `cannot import name '_C' from 'sam2'` 警告 | C 扩展未编译 | 可忽略，不影响 mask 质量 |
| 场景里只有部分物体被删 | YOLO 第一帧漏检了某个类别 | 加 `--classes` 包含该类别；降 `--conf` |
| 某些帧物体重新出现 | 第一帧 prompt 没锚住 | 换一个更清晰的第一帧；或添加多帧 prompt（进阶） |
| ProPainter OOM | 视频太长/分辨率太高 | `--resize_ratio 0.5` 或分段处理 |
| 背景泛色（前景颜色残留） | mask 边缘太紧 | 加大 dilation（默认 9×9）|
| SAM2 权重下载慢 | Meta CDN 国内访问慢 | 用 HuggingFace 镜像：`hf-mirror.com/facebook/sam2.1-hiera-large` |

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `gen_masks_sam2.py` | YOLO bbox 检测 + SAM 2.1 视频追踪 → 输出 mask PNG |
| `run_propainter.py` | ProPainter `inference_propainter.py` 的封装 |
| `requirements_sam2.txt` | `sam2_env` 的依赖 pin |
| `requirements_propainter.txt` | `propainter_env` 的依赖 pin |

---

## 硬件需求

测试环境 NVIDIA RTX A6000（49 GB）：
- SAM 2.1（Hiera-Large）跑 `bmx-trees` 80 帧：约 30 秒，峰值显存 ~8 GB
- ProPainter 跑 `bmx-trees` 80 帧 @ 854×480：约 2 分钟，峰值 ~12 GB