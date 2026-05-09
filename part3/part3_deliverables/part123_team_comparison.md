# Part1 / Part2 / Part3 团队对比表

这份表是给队友直接看结果用的，不是完整实验台账。

## 核心结论（DAVIS5 Macro JM）

| Part | 方法 | DAVIS5 Macro JM |
|---|---|---:|
| **Part 1** | YOLO+Lucas-Kanade+cv2.inpaint | **0.4922** |
| **Part 2** | YOLO+SAM2+ProPainter | **0.8451** |
| **Part 3 A+B Best** | (GDINO/VLM+SAM3) ∪ (VGGT4D+SAM3 refine) +ProPainter | **0.9119** |

> Part2 → Part3 mask 质量提升 +7.9pp（相对提升约 +9.4%）

## 1. Mask 质量对比（JM）

- `Part1`：最早版本
- `Part2`：YOLO+SAM2 基线
- `Part3-A`：SAM3 multi-object 主线
- `Part3-B1`：VGGT4D 原始动态发现
- `Part3-B5`：VGGT4D + SAM3 refine
- `Part3-A+B Best`：当前按序列选优的最好结果

| Sequence | Part1 | Part2 | Part3-A | Part3-B1 | Part3-B5 | Part3-A+B Best |
|---|---:|---:|---:|---:|---:|---:|
| tennis | 0.5800 | 0.9320 | 0.9468 | 0.7571 | 0.8735 | 0.9468 |
| blackswan | 0.5118 | 0.9551 | 0.9546 | 0.2082 | 0.9558 | 0.9558 |
| horsejump-low | 0.6176 | 0.7235 | 0.8574 | 0.6438 | 0.9331 | 0.9331 |
| bmx-trees | 0.3466 | 0.6403 | 0.6308 | 0.4421 | 0.6887 | 0.7455 |
| car-shadow | 0.4048 | 0.9746 | 0.8910 | 0.7589 | 0.9785 | 0.9785 |
| DAVIS5_Macro | 0.4922 | 0.8451 | 0.8561 | 0.5620 | 0.8859 | 0.9119 |

## 2. 最终视频结果对比（mask_JM / PSNR_proxy / SSIM）

这里只放最终流程里最常用的几条路线，便于队友快速看：

- `Part2_Baseline`：Part2 YOLO+SAM2+ProPainter
- `Part3_A`：Part3 SAM3+ProPainter
- `Part3_B1`：Part3 VGGT4D 原始 mask（无视频分数）
- `Part3_B5`：Part3 VGGT4D+SAM3 refine（当前 unified 里无视频分数）
- `Part3_ABBest`：Part3 A+B 最优完整流程

| Sequence | P2 maskJM | P2 PSNR | P2 SSIM | P3-A maskJM | P3-A PSNR | P3-A SSIM | P3-B5 maskJM | P3-ABBest maskJM | P3-ABBest PSNR | P3-ABBest SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tennis | 0.9320 | 32.5143 | 0.8998 | 0.9468 | 33.5709 | 0.9160 | 0.8735 | 0.9468 | 35.1285 | 0.9264 |
| blackswan | 0.9550 | nan | nan | 0.9549 | 33.4755 | 0.8844 | 0.9558 | 0.9558 | 33.4822 | 0.8844 |
| horsejump-low | 0.7230 | nan | nan | 0.8574 | 33.5237 | 0.9103 | 0.9331 | 0.9331 | 33.4754 | 0.9085 |
| bmx-trees | 0.6400 | 32.3113 | 0.9229 | 0.7455 | 35.1406 | 0.9429 | 0.6887 | 0.7455 | 35.2173 | 0.9432 |
| car-shadow | 0.9750 | nan | nan | 0.9749 | 31.7644 | 0.9121 | 0.9785 | 0.9785 | 37.4591 | 0.9193 |
| koala | nan | nan | nan | 0.9482 | 33.8071 | 0.8026 | 0.9373 | 0.9482 | 32.2091 | 0.8346 |

## 3. Direction C / DiffuEraser 补充结论（inpaint_only）

这部分只比较修复工具，mask 协议统一使用 `DAVIS annotation / GT mask`，因此属于 `inpaint_only` 公平对比，不混入不同方法各自 mask 的影响。

DiffuEraser 共验证 v1-v8 八个版本，完整横向表见 `direction_c_diffueraser_ablation.md`。核心结论如下：

| Version | 主要变化 | PSNR_proxy | PSNR_synth | 结论 |
|---|---|---:|---:|---|
| v1 | 默认 DiffuEraser + GT mask | 31.32 | 8.42 | 未过，出现 soft-blending 泄漏 / 白色虚影 |
| v2 | 调 guidance / steps | 31.22 | 8.42 | 未过 |
| v3 | 降分辨率 + dilation=4 | 30.62 | 8.50 | 更差 |
| **v4** | **hard-blend：mask 外硬贴回 DAVIS 原帧** | **36.28** | **32.88** | **通过门槛，白色虚影消除** |
| v5 | neighbor_length=20, subvideo=70 | 31.32 | 8.42 | 未过 |
| v6 | max_img_size=1280 | 31.32 | 8.43 | 未过 |
| v7 | 最紧 mask：dilate_px=0 | 32.46 | 13.10 | 有改善但未过 |
| v8 | ref_stride=5 | 31.31 | 8.41 | 未过 |

结论：DiffuEraser 原生输出的问题主要不是 mask 错，而是 inpainting 阶段的 soft-blending 会污染非 mask 区域，造成白色虚影和较低 PSNR_proxy。v4 的 hard-blend 后处理把 mask 外像素强制替换回 DAVIS 原帧，PSNR_proxy 从约 31.3 提升到 36.28，超过 `pure_propainter_gtmask` 基线 34.88。后续如果继续使用 DiffuEraser，应优先采用 v4 hard-blend 作为 Direction C 的候选方案，再扩展到 `bmx-trees` / `car-shadow` 验证泛化性。


## 3. 结论

- **Mask 质量**：Part1 0.4922 → Part2 0.8451 → Part3 A+B Best **0.9119**（DAVIS5 Macro JM）
- `Direction A`（GDINO/VLM + SAM3）在大多数序列表现稳定；`Direction B`（VGGT4D + SAM3 refine）在 `horsejump-low`、`car-shadow` 有额外增益。
- `Part3-A+B Best` 是按序列选优的融合结果，若需要单一方法请使用 `Part3-A`（SAM3 multi-object）。
- `inpaint_only` 对比现已统一为 DAVIS GT mask 协议（见 `pure_propainter_gtmask` / `sdxl_kf5_gtmask_propainter` / `lama_gtmask_propainter`）。

完整实验台账见：`part3_deliverables/experiment_registry.csv`
