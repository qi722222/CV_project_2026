# SAM3 Gate 决策记录（已执行）

## 最终执行结果

- 已完成：`SAM3` 全量接入（5 序列、Stage1+Stage2、mask+mp4 产物）。
- 统一主表见：`part3/gdino_vlm/part123_sam2_sam3_compare.md`
- 关键宏平均（IoU/J）：
  - Part3 SAM2 Stage1：`0.866043`
  - Part3 SAM2 Stage2：`0.808324`
  - Part3 SAM3 Stage1：`0.739991`
  - Part3 SAM3 Stage2：`0.687136`

## 叙事决策（用于 final report）

- **主文主线：保留 SAM2**（当前数据下整体更优，尤其 `bmx-trees`）。
- **SAM3 定位：升级尝试与边界结论**  
  已验证可切换、可复现，但在当前配置上未超过 SAM2。
- **口径要求**：不做指标口径漂移；SAM3 结果以负结果分析+创新开关消融呈现。

## 失败分析重点

- `bmx-trees`：SAM3 Stage1/Stage2 均显著偏低，是当前主要瓶颈序列。
- Stage2 在 SAM3 下继续下降，说明 “稀疏重锚 + 现有关联策略” 仍需更稳健设计。
- 创新消融显示 O2O 关联有提升潜力（见 `sam3_innovation_ablation.md`）。

## 复现实验入口

- 全量跑数脚本：`part3/gdino_vlm/run_sam3_davis5.sh`
- 创新开关脚本：`part3/gdino_vlm/run_sam3_innovation_ablation.sh`
- 评估脚本：`eval/eval_davis_masks.py`
