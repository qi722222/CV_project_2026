# Stage2 验收基线冻结

本文件用于冻结 Stage2 冲刺的比较口径。后续所有 Stage2 结果均以本页为基线，不得替换。

## 固定基线文件

- `Part2` 基线: `/data3/jli657/project3/eval/results_davis_masks.csv`
  - sha256: `1403ef3d2dec08f01e7732659eea68c9c7ac4d524adac183236dd2ca521b7a10`
- `Part3-Stage1` 基线: `/data3/jli657/project3/part3/gdino_vlm/eval_stage1.csv`
  - sha256: `835fe75741e0a1c650efd77537a27f1d2bc12635b9930b38eeb0c0656ca1902d`
- `Part1/2/3-S1` 对照总表: `/data3/jli657/project3/part3/gdino_vlm/part123_davis_compare.csv`
  - sha256: `b2854c7c6a5cd43ed2871d4f45d47fd77acd2fddaf8a02dd50109c7fe64e76b8`

## Stage1 固定指标（5 序列）

来自 `eval_stage1.csv`:

- bmx-trees: IoU/J `0.745507`, F `0.796067`
- tennis: IoU/J `0.931044`, F `0.959095`
- blackswan: IoU/J `0.955155`, F `0.928346`
- car-shadow: IoU/J `0.974851`, F `0.949308`
- horsejump-low: IoU/J `0.723656`, F `0.769825`

宏平均:

- IoU/J `0.8660`
- F `0.8805`

## 唯一验收标准

Stage2 参数轮次必须满足以下任一条件，才可判定“超过 Stage1”:

1. 宏平均 IoU/J 高于 `0.8660`，且无明显质量崩塌序列。
2. 难例 `bmx-trees` IoU/J 高于 `0.745507`，且其余序列无显著退化。

若多轮调参后仍不满足，Stage2 保持“探索增强”定位，不替换 Stage1 主线。
