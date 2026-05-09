# GDINO 消融总结（含 SAM2/SAM3 版本字段）

| Variant | Macro IoU/J | Macro F |
|---|---:|---:|
| Part2 YOLO baseline | 0.845106 | 0.859767 |
| Part3 SAM2 Stage1 | 0.866043 | 0.880528 |
| Part3 SAM2 Stage2 | 0.808324 | 0.830129 |
| Part3 SAM3 Stage1 | 0.739991 | 0.736974 |
| Part3 SAM3 Stage2 | 0.687136 | 0.709031 |

- 观察：SAM3 Stage1 在 tennis / blackswan / car-shadow 保持较高精度，但 bmx-trees 上明显弱于 SAM2 Stage1。
- 观察：SAM3 Stage2 当前低于 SAM3 Stage1，说明稀疏重锚在现参数下仍有退化风险。

- CSV: `/data3/jli657/project3/part3/gdino_vlm/gdino_ablation.csv`