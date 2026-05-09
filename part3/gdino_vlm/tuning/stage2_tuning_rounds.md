# Stage2 Tuning Rounds 对比（round2）

基线与判定口径：

- Stage1 baseline（`eval_stage1.csv`）Macro IoU/J = **0.866043**，`bmx-trees` IoU/J = **0.745507**
- 判定“超过 Stage1”口径：**同时满足** `Macro IoU/J > Stage1` 且 `bmx-trees IoU/J > Stage1`

| 方案 | Macro IoU/J | 是否超过 Stage1(Macro) | bmx-trees IoU/J | 是否超过 Stage1(bmx) | 验收判定 |
|---|---:|:---:|---:|:---:|:---:|
| Stage1 baseline | 0.866043 | - | 0.745507 | - | baseline |
| round0（当前 stage2） | 0.808324 | 否 | 0.581130 | 否 | 未通过 |
| round1 | 0.840480 | 否 | 0.680926 | 否 | 未通过 |
| round2 | 0.792504 | 否 | 0.411754 | 否 | 未通过 |

## 最佳 round 选择

- 由于 round0 / round1 / round2 均未超过 Stage1，**本轮 Stage2 未通过验收**。
- 按约束保持正式结果不变：`part3/gdino_vlm/eval_stage2.csv` 不替换。
- 若仅看 Stage2 rounds 内部，round1 的 Macro IoU/J 最高（0.840480），但仍低于 Stage1。
