# Part3 Deliverables

这个目录是所有 Part3 实验的可审计交付入口。任何实验只有在这里有目录、`metrics.json` 和 `experiment_card.md`，才算真正完成。

---

## 实验完成定义（强约束）

一个实验被视为"完成"，必须满足以下全部条件：

1. `part3_deliverables/<seq>/<exp_id>/` 目录存在
2. `experiment_card.md` 存在，且四问已填写
3. `metrics.json` 存在（即使指标为空，也需要文件存在）
4. `masked_in.mp4` 可查看（确认 mask 覆盖正确）
5. 如果是 `inpaint_only` 或 `full_pipeline`，`inpaint_out.mp4` 也必须存在

未满足以上条件的实验目录，视为"进行中"或"待完成"，不应引用为最终结果。

---

## 固定看结果顺序（强约束）

1. **先看 `masked_in.mp4`**：确认 mask 是否覆盖目标、有无误伤背景、是否时间抖动
2. **再看 `inpaint_out.mp4`**：确认修复区域是否自然、是否闪烁、是否 hallucination、非 mask 区域有无污染
3. **最后看指标**：只有前两个视频入口都合理，`PSNR / SSIM / JM / JR / F` 才能作为结论证据

**不要直接跳到指标**。指标高但视觉差的实验，应标记 `needs_review`，不能作为最终正向结论。

---

## 三类实验分类

| 类别 | 目的 | 主要指标 | 典型 Direction |
|---|---|---|---|
| `mask_only` | 只评 mask，不评修复 | JM / JR / F | Direction A, B |
| `inpaint_only` | 固定同一套 mask，只比修复工具 | PSNR_proxy / PSNR_synthetic / SSIM | Direction C |
| `full_pipeline` | 完整链路，mask+修复一起评价 | JM + PSNR + SSIM | A+B+C 融合 |

`inpaint_only` 比较必须统一 mask 来源：DAVIS 序列使用 DAVIS GT mask，不得混用 SAM3 预测 mask。

---

## 版本化规范（schema v2）

- 所有持续优化实验必须显式版本化：`method_v1 / v2 / v3`
- 不允许覆盖旧版本。新版本必须写入新目录
- 每个版本的 `experiment_card.md` 必须包含 `Version History` 区块
- 新实验完成后必须写出 `run_manifest.json`，通过 builder 自动注册

---

## 审计状态说明

| 状态 | 含义 |
|---|---|
| `reference` | 历史参照基线，必须保留 |
| `stable` | 当前可直接用于汇报的稳定结果 |
| `promising` | 有正向信号，值得继续精进 |
| `exploratory` | 探索性结果，能说明问题，但不宜直接当最终结论 |
| `legacy` | 旧版路线，保留做对照 |
| `superseded` | 被后续版本替代，仍保留痕迹 |
| `partial_or_failed` / `failed` | 有明显失败或不完整问题 |
| `needs_review` | 指标异常或视觉质量存疑，需要人工核查 |

---

## DiffuEraser 验证进度（Direction C 首优先）

| 阶段 | 状态 | 说明 |
|---|---|---|
| 环境安装 | ✅ 已完成 | conda env: `diffueraser_env` |
| 输入准备 | ✅ 已完成 | tennis: `input_video.mp4 + input_mask.mp4`（70帧，854x480）|
| 权重下载 | ⏳ 进行中 | diffuEraser / SD1.5 / sd-vae-ft-mse / PCM_Weights |
| smoke test | ⏳ 待运行 | 环境导入验证 |
| v1 推理 | ⏳ 待运行 | tennis 首轮 inpaint_only |
| 指标评估 | ⏳ 待运行 | 与 pure_propainter_gtmask 比较 |

---

## 文件结构

```
part3_deliverables/
├── README.md                        # 本文件
├── experiment_registry.csv          # 所有实验的配置和路径索引
├── experiment_registry.json
├── part3_results_full_table.csv     # 完整结果表（含 version/mask_protocol 列）
├── part3_results_full_table.md
├── part3_results_full_table.json
├── part123_team_comparison.md       # 面向队友的简洁对比表
└── <sequence>/                      # 每个序列一个目录
    ├── 00_readme.md                 # 本序列下所有实验摘要
    └── <exp_id>/                    # 每个实验一个目录
        ├── experiment_card.md       # 实验卡片（含四问 + Version History）
        ├── metrics.json             # 指标 + schema v2 元数据
        ├── command.sh               # 可复现命令
        ├── run_manifest.json        # (新实验) 自动注册 manifest
        ├── masked_in.mp4            # mask 覆盖预览
        ├── inpaint_out.mp4          # 修复结果
        ├── mask_frames/             # mask 帧目录（软链接）
        └── <script>.py              # 运行脚本（软链接）
```

---

*本地详细说明见 `../DELIVERABLES_GUIDE_CN.md`（未跟踪文件，仅本地）*
