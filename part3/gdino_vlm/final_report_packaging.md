# Final Report Packaging (SAM2/SAM3)

## 主文叙事决策

- 主线方法：`Part3-SAM2-Stage1`（当前 5 序列宏平均更优）。
- SAM3 位置：升级尝试与边界结论（已全量接入并给出负结果分析）。
- 创新点主推：`O2O temporal association`（在 DAVIS2 子集上有正增益）。

## 可直接引用的结果文件

- 统一总表：`part3/gdino_vlm/part123_sam2_sam3_compare.csv`
- 统一总表（Markdown）：`part3/gdino_vlm/part123_sam2_sam3_compare.md`
- 后端区分消融：`part3/gdino_vlm/gdino_ablation.csv`
- 消融说明：`part3/gdino_vlm/ablation_summary.md`
- 创新开关消融：`part3/gdino_vlm/sam3_innovation_ablation.md`
- SAM3 决策记录：`part3/gdino_vlm/sam3_decision.md`

## 可视化产物（SAM3）

- Stage1 mask/overlay mp4 根目录：`part3/gdino_vlm/outputs/sam3/*_stage1/`
- Stage2 mask/overlay mp4 根目录：`part3/gdino_vlm/outputs/sam3/*_stage2/`
- 对应 mask png 根目录：`part3/gdino_vlm/masks/sam3/`

## 可视化与评估产物（ControlNet 5序列三组对照）

- 汇总结论与总表：
  - `part3/outputs/controlnet/ablation_5seq/ablation_summary.md`
  - `part3/outputs/controlnet/ablation_5seq/summary_5seq.csv`
  - `part3/outputs/controlnet/ablation_5seq/summary_5seq.json`
- 逐序列评估目录（含 `metrics_summary.json` + `metrics_per_frame.csv` + `gate_log.json`）：
  - `part3/outputs/controlnet/ablation_5seq/<seq>/eval/`
- 逐序列可视化视频（3方法）：
  - `part3/outputs/controlnet/ablation_5seq/<seq>/propainter_pure/<seq>/inpaint_out.mp4`
  - `part3/outputs/controlnet/ablation_5seq/<seq>/propainter_hybrid/hybrid_frames/inpaint_out.mp4`
  - `part3/outputs/controlnet/ablation_5seq/<seq>/propainter_hybrid_tc/hybrid_tc_frames/inpaint_out.mp4`

> 当前结论：`hybrid` 存在稳定负增益；`hybrid_temporal_consistency` 可显著回升并逼近 `pure_propainter`，但默认推荐仍为 `pure_propainter`，ControlNet 分支定位为条件式增强而非主干替代。

## 小成本冲刺复验（2序列 tuned）

- tuned 总表：
  - `part3/outputs/controlnet/ablation_2seq_tuned/summary_2seq_tuned.csv`
  - `part3/outputs/controlnet/ablation_2seq_tuned/summary_2seq_tuned.json`
- tuned 结论：
  - `part3/outputs/controlnet/ablation_2seq_tuned/validation_summary.md`
- 门控校准日志（含 default vs calibrated 对比）：
  - `part3/outputs/controlnet/ablation_2seq_tuned/<seq>/eval/<method>/gate_log.json`

> 小成本冲刺结论：门控决策可从“全回退 pure”变为“可选 hybrid_tc”，但指标未达到正增益验收线，因此不继续扩到5序列 tuned 全量重跑；最终默认主干仍为 `pure_propainter`。

## 报告可直接粘贴文本

- 失败机理与边界条件（中文稿）：
  - `part3/outputs/controlnet/report_ready_failure_analysis_cn.md`

## 报告建议写法

- 先给统一主表（Part1/Part2/Part3-SAM2/SAM3），说明 SAM2 仍为当前最优主线。
- 再给 SAM3 全量接入结果，强调“方法贡献不依赖 backbone 偶然提升”。
- 最后给创新点开关消融，突出 O2O 的增益与 QualityGate/RealVLM 的边界条件。
