# SAM3 创新点开关消融

| Sequence | Baseline-S2 | +QualityGate | +O2O | +RealVLM |
|---|---:|---:|---:|---:|
| bmx-trees | 0.365081 | 0.365081 | 0.478006 | 0.365081 |
| tennis | 0.765717 | 0.765717 | 0.785951 | 0.765717 |

## Macro IoU/J

- Baseline-S2: `0.565399`
- +QualityGate: `0.565399` (delta `+0.000000`)
- +O2O: `0.631978` (delta `+0.066580`)
- +RealVLM: `0.565399` (delta `+0.000000`)

## 结论与失败分析

- 创新点B（O2O）在两条序列都提升，尤其 `bmx-trees` 提升更明显，说明多目标/快速运动下，避免重复匹配能减少ID混叠。
- 创新点A（QualityGate）与基线完全一致，说明当前阈值配置下门控基本未触发有效重锚路径，后续需结合更激进阈值或质量信号。
- 创新点C（RealVLM）与基线一致：`prompt_source=real_vlm` 已生效，但生成prompt与原规则prompt接近，未带来额外收益。
- 失败片段建议优先查看 `outputs/sam3/*bmx-trees*` 的 overlay 视频，高速形变时仍有掩码断裂与边界抖动。