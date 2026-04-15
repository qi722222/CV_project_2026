# CV Project 2026: 视频目标移除与 Inpainting

**AIAA 3201 — Introduction to Computer Vision，Spring 2026**

本项目实现了一个完整的视频动态目标移除与背景修复 pipeline，对比了**经典 CV 方法**与**现代基础模型方法**两种思路。

> 📄 最终报告 (arXiv)：_即将上线_
> 🎬 演示视频：见 `videos.zip`（最终提交）

---

## TL;DR

| Part | 方法 | 状态 |
|---|---|---|
| **Part 1** | YOLOv8-Seg + Lucas-Kanade 光流 + 时序背景传播 + cv2.inpaint | ✅ 完成 |
| **Part 2** | YOLOv8-Seg（bbox 提示）→ SAM 2.1 → ProPainter | ✅ 完成 |
| **Part 3** | Stable Diffusion Inpainting 关键帧修复（Direction C） | 🚧 进行中 |

**核心结论**：Part 1 的时序传播方法在剧烈相机运动下会失效（如 `bmx-trees`），出现严重虚影；Part 2 的 ProPainter 通过光流引导传播能正确处理这种情况，体现了"学习式光流"相对"启发式像素借用"的根本优势。

---

## 仓库结构

```
project3/
├── README.md                  # 本文件
├── part1/                     # 经典 CV 方法 baseline
│   ├── README.md              # Part 1 环境与使用
│   ├── scripts/               # Pipeline 模块
│   │   ├── gen_masks_yolo.py
│   │   ├── inpaint_temporal.py
│   │   ├── run_part1.py
│   │   ├── compare.py
│   │   ├── make_compare.py
│   │   └── make_full_compare.py
│   └── run_sweep_v2.sh        # 参数扫描
├── part2/                     # SOTA pipeline (SAM2 + ProPainter)
│   ├── README.md              # Part 2 环境与使用
│   ├── gen_masks_sam2.py      # YOLO bbox → SAM 2.1
│   ├── run_propainter.py
│   └── requirements_*.txt
└── part3/                     # SD 关键帧修复（开发中）
```

---

## 快速上手

### 1. 克隆仓库

```bash
git clone https://github.com/qi722222/CV_project_2026.git
cd CV_project_2026
```

### 2. 配置环境

本项目需要**三个独立的 conda 环境**，因为底层模型有依赖冲突（主要是 `torch`、`mmcv`、`hydra` 等）：

```bash
# Part 1: YOLO + 经典 CV
conda create -n part1_env python=3.10 -y
conda activate part1_env
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics --no-deps
pip install pandas seaborn opencv-python scikit-image imageio imageio-ffmpeg numpy scipy

# Part 2: SAM 2.1 + ProPainter（需要两个独立环境）
# 详见 part2/README.md
```

### 3. 准备数据

数据集存放在任意目录，运行时通过 CLI 参数传路径。我们测试了以下数据集：

| 数据集 | 来源 | 帧数 | 备注 |
|---|---|---|---|
| `bmx-trees` | DAVIS | 80 | 相机跟拍主体（困难案例）|
| `tennis` | DAVIS | 70 | 多目标：球员 + 球拍 |
| Wild Video | 自录 | 30+ | 作业要求；建议固定机位 |
| DAVIS | https://davischallenge.org | 不定 | 可选，用于定量指标 |

**注意**：本仓库**不包含**数据集文件（体积大，且非我们的 IP）。请从 DAVIS 官网下载，自行录制 Wild Video。

### 4. 运行某个 Part

详见 [`part1/README.md`](part1/README.md) 与 [`part2/README.md`](part2/README.md)。

---

## 方法亮点

### Part 1 — 经典 CV Baseline
- **Mask**：YOLOv8x-Seg 检测 person/bicycle，Lucas-Kanade 光流过滤静态目标，膨胀覆盖运动模糊边缘
- **Inpaint**：时序背景传播（在邻近帧的同一坐标借用干净像素）+ `cv2.inpaint`（Telea 算法）作为 fallback
- **优势**：在静态相机场景下效果良好
- **局限**：相机运动会破坏"同坐标借像素"的基本假设 → 出现虚影

### Part 2 — SOTA Pipeline
- **Mask**：YOLOv8x-Seg 提供 bounding box 作为 SAM 2.1 的 prompt，由 SAM2 利用其 memory 模块在所有帧上传播像素级精度的 mask
- **Inpaint**：ProPainter 使用双域（光流 + 特征）传播 + 稀疏 Transformer 融合
- **关键设计**：我们用 **YOLO bbox → SAM2 prompt** 替代手工点击。这样做：(1) 与 Part 1 在语义上对齐（删的是同样的目标类别）(2) 无需人工标注 (3) 避免了 SAM2 point prompt 的歧义（比如点在自行车上不会自动包括骑手）
- **优势**：能处理相机运动和大面积遮挡

### Part 3 — 生成式精修（进行中）
- ProPainter 在"背景从未在视频里出现"的情况下会失效（例如一面墙一直被运动的人挡着）
- 我们探索用 Stable Diffusion Inpainting 修复关键帧，再用光流传播到其他帧

---

## 硬件

测试环境：
- 2× NVIDIA RTX A6000（每张 49 GB 显存）
- CUDA Driver 12.8，nvcc 12.4
- Ubuntu 22.04，Python 3.10

ProPainter 在 `bmx-trees` 全分辨率（80 帧，854×480）下需要约 12 GB 显存。OOM 时加 `--resize_ratio 0.5`。

---

## 致谢

本项目基于以下开源工作：
- [YOLOv8](https://github.com/ultralytics/ultralytics) — Ultralytics
- [SAM 2](https://github.com/facebookresearch/sam2) — Meta AI
- [ProPainter](https://github.com/sczhou/ProPainter) — Zhou et al.，ICCV 2023
- [DAVIS Dataset](https://davischallenge.org)

完整引用见报告。

---

## License

代码：MIT。预训练权重遵循其各自上游的 License。