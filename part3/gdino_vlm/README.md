# GDINO / VLM Mainline Runner

本目录是 Part3 主线实现：`VLM -> detector(GDINO/YOLO) -> segmentor(SAM2/SAM3) -> mask/mp4`。

## 当前状态（已完成）

- SAM2 与 SAM3 后端都可切换（配置项 `segmentor.backend`）。
- 已完成 SAM3 全量跑数：5 序列 × Stage1 + Stage2（含 `mask png + run_meta + mask/overlay mp4`）。
- 已重建统一主表：`part3/gdino_vlm/part123_sam2_sam3_compare.md`。
- 已完成创新点开关消融：`part3/gdino_vlm/sam3_innovation_ablation.md`。

## 关键脚本

- `run_gdino_mainline.py`：主入口（支持 `sam2/sam3`）。
- `run_sam3_davis5.sh`：SAM3 5 序列 Stage1+Stage2 一键跑。
- `run_sam3_innovation_ablation.sh`：创新点 A/B/C 开关实验。
- `build_unified_compare_csv.py`：重建 Part1/Part2/Part3(SAM2+SAM3) 总表。
- `build_gdino_ablation_v2.py`：输出带版本字段的 `gdino_ablation.csv`。

## 环境建议

推荐将 Part3 放在独立环境 `sam3_env`（已验证可用）：

```bash
conda create -y -n sam3_env --clone /home/jli657/.conda/envs/gdino_env
PATH="/data2/jli657/envs/sam3_env/bin:$PATH" pip install --upgrade \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

若网络受限，建议加：

```bash
HF_ENDPOINT="https://hf-mirror.com"
```

## SAM2 / SAM3 切换方式

- SAM2：使用 `part3/configs/gdino_vlm_mainline.yaml`，`segmentor.backend: sam2`
- SAM3 Stage1：`part3/configs/gdino_vlm_mainline_sam3.yaml`
- SAM3 Stage2：`part3/configs/gdino_vlm_mainline_sam3_stage2.yaml`

示例：

```bash
HF_ENDPOINT="https://hf-mirror.com" \
PATH="/data2/jli657/envs/sam3_env/bin:$PATH" \
python part3/gdino_vlm/run_gdino_mainline.py \
  --config part3/configs/gdino_vlm_mainline_sam3.yaml \
  --sequence tennis \
  --stage stage1 \
  --output part3/gdino_vlm/masks/sam3/stage1/tennis
```

## 结果叙事（当前建议）

- 现有数据下，主文保持 SAM2 主线更稳（宏平均更高）。
- SAM3 作为“升级尝试与边界结论”写入，不回避负结果。
- 创新点里 O2O 关联有正增益（DAVIS2 上优于 baseline）。

详见：
- `part3/gdino_vlm/part123_sam2_sam3_compare.md`
- `part3/gdino_vlm/sam3_decision.md`
- `part3/gdino_vlm/sam3_innovation_ablation.md`

## VLM 接入现状

- 已有真实 VLM 最小接入开关：`vlm.real_prompt.enable`。
- 失败自动回退规则 prompt；`run_meta.json` 中记录 `prompt_source`。
- 当前实验里 RealVLM 路径可运行，但在已测序列上未带来额外数值收益。
