# Part 3: GDINO/VLM + SAM2/SAM3 主线 + 创新点消融

## 概述

Part 3 在 Part 2 基础上引入两条创新方向，进一步提升 mask 质量：

- **Direction A（Mask Upgrade）**：使用 Grounding DINO (GDINO) 自动生成检测框，替代手工 YOLO bbox 提示，再经 SAM2/SAM3 生成像素级 mask；并消融 QualityGate / One-to-One (O2O) / RealVLM 三种创新开关。
- **Direction B（Better Mask via Foundation Models）**：使用 VGGT4D 进行无监督动态物体发现，获得粗 mask 后再用 SAM3 精化边缘。

**核心结论（DAVIS5 Macro JM）**

| Part | 方法 | DAVIS5 Macro JM |
|---|---|---:|
| Part 1 | YOLO + Lucas-Kanade + cv2.inpaint | 0.4922 |
| Part 2 | YOLO + SAM2 + ProPainter | 0.8451 |
| **Part 3 A+B Best** | GDINO/VLM+SAM3 ∪ VGGT4D+SAM3 refine + ProPainter | **0.9119** |

---

## 目录结构

```
part3/
├── README.md                       # 本文件
├── requirements_controlnet.txt     # Python 依赖
│
├── # ── Direction A 主线 ──────────────────────────────────────────
├── run_sam3_multiobject.py         # SAM3 multi-object mask（主线）
├── run_part3_refine.py             # ControlNet 关键帧精修
├── run_controlnet_ablation_5seq.py # ControlNet 消融实验
├── run_part3_sam3_rebuild.py       # SAM3 rebuild v1
│
├── # ── Direction B 主线 ──────────────────────────────────────────
├── run_direction_b_vggt4d.py       # VGGT4D 无监督发现
├── run_direction_b_sam3_refine.py  # VGGT4D + SAM3 精化
├── run_direction_b_comparison.py   # Direction B 对比评估
│
├── # ── GDINO/VLM 子实验 ──────────────────────────────────────────
├── gdino_vlm/
│   ├── run_gdino_mainline.py       # GDINO Stage1+Stage2 主线
│   ├── run_sam3_innovation_ablation.sh  # 创新点消融脚本
│   └── policies/                   # 评估策略 YAML
│
├── # ── inpaint-only 公平对比（GT mask 协议）─────────────────────
├── run_propainter_gtmask.py        # 纯 ProPainter + DAVIS GT mask
├── run_phase2_sdxl_all7.py         # SDXL kf5 + ProPainter + GT mask
├── run_phase3_lama_all7.py         # LaMa + ProPainter + GT mask
│
├── # ── 评估与整理 ─────────────────────────────────────────────────
├── evaluate_all.py                 # 统一量化评估
├── eval_unified_v2.py              # 综合评估 v2
├── build_part3_deliverables.py     # 生成交付目录结构
├── build_part3_result_table.py     # 生成完整结果表
├── build_part123_team_comparison.py # 生成团队对比表
│
├── configs/                        # 各序列配置 YAML
└── part3_deliverables/             # 实验台账（轻量元数据）
    ├── experiment_registry.csv/json
    ├── part3_results_full_table.md
    └── part123_team_comparison.md
```

---

## 快速使用

### 环境准备

```bash
# Part 3 主线：controlnet_env（SDXL / LaMa / ControlNet）
conda create -n controlnet_env python=3.10 -y
conda activate controlnet_env
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r part3/requirements_controlnet.txt

# inpaint 传播：propainter_env（和 Part 2 共用）
# 见 part2/README.md
```

### Direction A — SAM3 Multi-Object Mask

```bash
# 生成 DAVIS 5 序列的 SAM3 multi-object mask
conda run -n controlnet_env python3 part3/direction_a/run_sam3_multiobject.py \
  --sequences tennis bmx-trees blackswan car-shadow horsejump-low
```

### inpaint-only 公平对比（DAVIS GT mask 协议）

```bash
# 1. 纯 ProPainter（基线）
conda run -n propainter_env python3 part3/inpainting/run_propainter_gtmask.py \
  --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

# 2. LaMa + ProPainter
conda run -n controlnet_env python3 part3/inpainting/run_phase3_lama_all7.py \
  --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

# 3. SDXL kf5 + ProPainter（需要 GPU）
PYTHONUNBUFFERED=1 conda run -n controlnet_env python3 part3/inpainting/run_phase2_sdxl_all7.py \
  --gpu 0 --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

# 4. DiffuEraser（Direction C 首优先候选，首轮只跑 tennis）
# Step 0: 环境安装（首次运行）
bash part3/inpainting/setup_diffueraser.sh

# Step 1: 准备输入视频和 mask 视频
conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py \
  --seq tennis --version v1 --dilate_px 5

# Step 2: 运行 DiffuEraser 推理
conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py \
  --seq tennis --version v1

# Step 3: 评估 PSNR/SSIM
conda run -n controlnet_env python3 part3/eval/evaluate_all.py --seqs tennis

# Step 4: 注册到 deliverables
python3 part3/reporting/build_part3_deliverables.py \
  --manifest /data3/jli657/project3/part3/results/tennis/direction_c/diffueraser_gtmask_v1/run_manifest.json
```

### DiffuEraser v4–v8 扩展调参（消除白色虚影）

```bash
# v4: 硬贴回（推荐！已通过门槛 PSNR_proxy=36.28 > 34.88）
# 基于 v1 推理结果做后处理，mask 外用 DAVIS JPEG 原帧替换
conda run -n diffueraser_env python3 part3/inpainting/apply_hard_blend.py \
  --base_version v1 --out_version v4 --sequence tennis

# v5: 大 neighbor window（neighbor_length=20, subvideo=70）
# 需先准备输入（可复用 v1 的 input_video/mask）
DIFFUERASER_DIR=/data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser
OUT_DIR=/data3/jli657/project3/part3/results/tennis/direction_c/diffueraser_gtmask_v5
cd ${DIFFUERASER_DIR} && conda run -n diffueraser_env python3 run_diffueraser.py \
  --input_video ${OUT_DIR}/input_video.mp4 --input_mask ${OUT_DIR}/input_mask.mp4 \
  --video_length 70 --save_path ${OUT_DIR} \
  --base_model_path weights/stable-diffusion-v1-5 --vae_path weights/sd-vae-ft-mse \
  --diffueraser_path weights/diffuEraser --propainter_model_dir weights/propainter \
  --mask_dilation_iter 0 --neighbor_length 20 --subvideo_length 70

# v6: 高分辨率（max_img_size=1280）- 原分辨率推理
  # 同上命令，改 --max_img_size 1280

# v7: 最紧 mask（dilate_px=0, mask_dilation_iter=0）
conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py \
  --seq tennis --version v7 --dilate_px 0
  # 然后运行 DiffuEraser: --mask_dilation_iter 0

# v8: 密参考帧（ref_stride=5）
  # 同 v5 命令，改 --ref_stride 5 --mask_dilation_iter 0

# 评估所有版本
conda run -n diffueraser_env python3 part3/eval/evaluate_all.py --seq tennis

# 完整 ablation 报告
cat part3/part3_deliverables/direction_c_diffueraser_ablation.md
```

### 生成结果表

```bash
# 重建交付目录和实验台账
/data2/jli657/envs/controlnet_env/bin/python3 part3/reporting/build_part3_deliverables.py

# 生成完整结果表（CSV/MD/JSON）
/data2/jli657/envs/controlnet_env/bin/python3 part3/reporting/build_part3_result_table.py

# 生成团队对比表
/data2/jli657/envs/controlnet_env/bin/python3 part3/reporting/build_part123_team_comparison.py
```

---

## 实验台账

完整实验定义和结果见：

- `part3_deliverables/experiment_registry.csv` — 所有实验的配置和路径索引
- `part3_deliverables/part3_results_full_table.md` — Part3 内部完整结果表
- `part3_deliverables/part123_team_comparison.md` — Part1/2/3 团队对比表

---

## 实验规范（schema v2）

### 强制实验分类

| 类别 | 目的 | 主要指标 |
|---|---|---|
| `mask_only` | 只评 mask，不评修复 | JM / JR / F |
| `inpaint_only` | 固定同一套 mask，只比修复工具 | PSNR_proxy / PSNR_synthetic / SSIM |
| `full_pipeline` | 完整链路，mask+修复一起评价 | JM + PSNR + SSIM |

### 固定看结果顺序

1. `masked_in.mp4`：确认 mask 是否覆盖目标、有无误伤背景、是否时间抖动
2. `inpaint_out.mp4`：确认修复区域是否自然、是否闪烁、是否 hallucination
3. 指标（PSNR/SSIM/JM）：只有前两项合理，指标才能作为结论证据

### 版本化强约束

- 所有持续优化实验必须显式版本化：`method_v1 / v2 / v3`，**不允许覆盖旧版本**
- 每个版本必须在 `experiment_card.md` 中记录 `Version History` 区块
- 未进入 `part3_deliverables/` 且没有 `metrics.json + experiment_card.md` 的实验，一律视为未完成

### mask 协议说明

| 协议 | mask 来源 | 适用实验 |
|---|---|---|
| `davis_gt` | DAVIS annotation / GT mask | inpaint-only 公平对比（`_gtmask` 后缀目录）|
| `sam3_mask` | SAM3 预测 mask | full_pipeline 流程（mask+inpaint 一起评价）|
| `wild_existing_mask` | SAM3 shadow mask | wild_video-1person demo（无 GT，单独分组）|

**注意**：`inpaint_only` 对比必须统一 mask 来源才公平。旧版 `pure_propainter_fixed_mask` 等使用 SAM3 预测 mask，已标记为 `superseded`，请改用 `_gtmask` 版本。

### DiffuEraser 实验定位（Direction C 首优先）

- **方法类型**：视频 inpainting，基于 BrushNet + AnimateDiff 架构
- **相对 ProPainter**：ProPainter 是强视频补洞基线，保守修复；DiffuEraser 是 diffusion 方案，目标是更强生成能力和时间一致性
- **进入流程**：必须先进入 `inpaint_only`，通过门槛后才进入 `full_pipeline`
- **首轮序列**：tennis（70帧，计算量可控，已有完整 GT mask 和 ProPainter 基线）
- **门槛**：工程稳定 + 视觉不明显劣于 `pure_propainter_gtmask` + PSNR/SSIM 至少一项不退化

---

## 数据路径

本仓库不包含数据集和模型权重（体积大、非本项目 IP）。路径配置：

| 数据 | 路径 |
|---|---|
| DAVIS 帧 | `/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/<seq>` |
| DAVIS GT mask | `/home/jli657/shared_data/project3/DAVIS/Annotations/480p/<seq>` |
| SAM3 权重 | `/data3/jli657/project3/weights/sam3/sam3.pt` |
| HF 模型缓存 | `/data3/jli657/hf_cache` |
