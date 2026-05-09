# Part3 完整结果整理表

这张表来自 `part3_deliverables/experiment_registry.json` 和每个方法目录下的 `metrics.json`。

## 怎么读

- `mask_only`：只看 mask 指标，不能直接拿 PSNR/SSIM 比。
- `inpaint_only`：固定 mask 后比较修复工具。
- `full_pipeline`：mask + 修复工具一起看，是最终视频效果。
- `failed / partial_or_failed / superseded / legacy`：保留用于追溯，不建议直接作为主结果引用。

## 汇总

- 总实验行数：`193`
- `mask_only`：`102`
- `inpaint_only`：`65`
- `full_pipeline`：`26`

## 完整表

| sequence | method_id | ver | mask_protocol | type | status | JM/mask_JM | PSNR_proxy | SSIM | baseline | next_decision | 结论 |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| tennis | `part2_mask_baseline` | legacy |  | mask_only | reference | 0.9320 |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| bmx-trees | `part2_mask_baseline` | legacy |  | mask_only | reference | 0.6403 |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| blackswan | `part2_mask_baseline` | legacy |  | mask_only | reference | 0.9551 |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| car-shadow | `part2_mask_baseline` | legacy |  | mask_only | reference | 0.9746 |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| horsejump-low | `part2_mask_baseline` | legacy |  | mask_only | reference | 0.7235 |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| koala | `part2_mask_baseline` | legacy |  | mask_only | reference |  |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| wild_video-1person | `part2_mask_baseline` | legacy |  | mask_only | reference |  |  |  |  |  | 它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。 |
| tennis | `part2_baseline_full_pipeline` | legacy |  | full_pipeline | reference |  | 32.5143 | nan |  |  | 它不是最先进，但它很重要，因为很多新方法必须先超过它才有说服力。 |
| bmx-trees | `part2_baseline_full_pipeline` | legacy |  | full_pipeline | reference |  |  |  |  |  | 它不是最先进，但它很重要，因为很多新方法必须先超过它才有说服力。 |
| wild_video-1person | `part2_baseline_full_pipeline` | legacy |  | full_pipeline | reference |  |  |  |  |  | 它不是最先进，但它很重要，因为很多新方法必须先超过它才有说服力。 |
| tennis | `official_sam3_video_mask_only` | legacy |  | mask_only | exploratory | 0.8796 |  |  |  |  | 它证明了官方 SAM3 能跑通，但单 prompt 版本不是最终最强方案。 |
| bmx-trees | `official_sam3_video_mask_only` | legacy |  | mask_only | exploratory | 0.5193 |  |  |  |  | 它证明了官方 SAM3 能跑通，但单 prompt 版本不是最终最强方案。 |
| blackswan | `official_sam3_video_mask_only` | legacy |  | mask_only | exploratory | 0.9543 |  |  |  |  | 它证明了官方 SAM3 能跑通，但单 prompt 版本不是最终最强方案。 |
| car-shadow | `official_sam3_video_mask_only` | legacy |  | mask_only | exploratory | 0.9008 |  |  |  |  | 它证明了官方 SAM3 能跑通，但单 prompt 版本不是最终最强方案。 |
| horsejump-low | `official_sam3_video_mask_only` | legacy |  | mask_only | exploratory | 0.7281 |  |  |  |  | 它证明了官方 SAM3 能跑通，但单 prompt 版本不是最终最强方案。 |
| tennis | `official_sam3_best_mask_only` | legacy |  | mask_only | exploratory | 0.8824 |  |  |  |  | 它说明 prompt 确实重要，但仅靠 prompt 调整还不够。 |
| bmx-trees | `official_sam3_best_mask_only` | legacy |  | mask_only | exploratory | 0.5175 |  |  |  |  | 它说明 prompt 确实重要，但仅靠 prompt 调整还不够。 |
| blackswan | `official_sam3_best_mask_only` | legacy |  | mask_only | exploratory | 0.9548 |  |  |  |  | 它说明 prompt 确实重要，但仅靠 prompt 调整还不够。 |
| car-shadow | `official_sam3_best_mask_only` | legacy |  | mask_only | exploratory | 0.9008 |  |  |  |  | 它说明 prompt 确实重要，但仅靠 prompt 调整还不够。 |
| horsejump-low | `official_sam3_best_mask_only` | legacy |  | mask_only | exploratory | 0.7281 |  |  |  |  | 它说明 prompt 确实重要，但仅靠 prompt 调整还不够。 |
| tennis | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable | 0.9468 |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| bmx-trees | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable | 0.6308 |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| blackswan | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable | 0.9546 |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| car-shadow | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable | 0.8910 |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| horsejump-low | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable | 0.8574 |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| koala | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable | 0.9482 |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| wild_video-1person | `sam3_multiobj_mask_only` | legacy |  | mask_only | stable |  |  |  |  |  | 这是目前 Direction A 最有代表性的 SAM3 路线之一。 |
| tennis | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  | 33.5709 | nan |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| bmx-trees | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  |  |  |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| blackswan | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  |  |  |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| car-shadow | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  |  |  |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| horsejump-low | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  |  |  |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| koala | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  |  |  |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| wild_video-1person | `sam3_multiobj_propainter_full_pipeline` | legacy |  | full_pipeline | stable |  |  |  |  |  | 它代表了 Direction A 在最终视频上的主要交付版本。 |
| tennis | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory | 0.7860 |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| bmx-trees | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory | 0.4690 |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| blackswan | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory |  |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| car-shadow | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory |  |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| horsejump-low | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory |  |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| koala | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory | 0.7518 |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| bear | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory | 0.9571 |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| camel | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory | 0.9693 |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| wild_video-1person | `sam3_rebuild_v1_mask_only` | legacy |  | mask_only | exploratory |  |  |  |  |  | 它是重要的重建实验，但不是所有序列最终都用它。 |
| tennis | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| bmx-trees | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| blackswan | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| car-shadow | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| horsejump-low | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| koala | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| bear | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| camel | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| wild_video-1person | `sam3_rebuild_v1_propainter_full_pipeline` | legacy |  | full_pipeline | exploratory |  |  |  |  |  | 它更像一条对照和重建路线，不是目前最统一的最终版本。 |
| tennis | `gdino_sam2_stage1_mask_only` | legacy |  | mask_only | stable | 0.9310 |  |  |  |  | 在当前数据里，它依然是很强的参照，尤其 bmx-trees 上比 SAM3 更稳。 |
| bmx-trees | `gdino_sam2_stage1_mask_only` | legacy |  | mask_only | stable | 0.7455 |  |  |  |  | 在当前数据里，它依然是很强的参照，尤其 bmx-trees 上比 SAM3 更稳。 |
| blackswan | `gdino_sam2_stage1_mask_only` | legacy |  | mask_only | stable | 0.9552 |  |  |  |  | 在当前数据里，它依然是很强的参照，尤其 bmx-trees 上比 SAM3 更稳。 |
| car-shadow | `gdino_sam2_stage1_mask_only` | legacy |  | mask_only | stable | 0.9749 |  |  |  |  | 在当前数据里，它依然是很强的参照，尤其 bmx-trees 上比 SAM3 更稳。 |
| horsejump-low | `gdino_sam2_stage1_mask_only` | legacy |  | mask_only | stable | 0.7237 |  |  |  |  | 在当前数据里，它依然是很强的参照，尤其 bmx-trees 上比 SAM3 更稳。 |
| tennis | `gdino_sam2_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.8411 |  |  |  |  | 当前整体不如 Stage1 稳，说明重锚不是越多越好。 |
| bmx-trees | `gdino_sam2_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.5811 |  |  |  |  | 当前整体不如 Stage1 稳，说明重锚不是越多越好。 |
| blackswan | `gdino_sam2_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.9542 |  |  |  |  | 当前整体不如 Stage1 稳，说明重锚不是越多越好。 |
| car-shadow | `gdino_sam2_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.9730 |  |  |  |  | 当前整体不如 Stage1 稳，说明重锚不是越多越好。 |
| horsejump-low | `gdino_sam2_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.6922 |  |  |  |  | 当前整体不如 Stage1 稳，说明重锚不是越多越好。 |
| tennis | `gdino_sam3_stage1_mask_only` | legacy |  | mask_only | exploratory | 0.8255 |  |  |  |  | 它在部分序列有亮点，但整体还没有超过 GDINO+SAM2 Stage1。 |
| bmx-trees | `gdino_sam3_stage1_mask_only` | legacy |  | mask_only | exploratory | 0.1843 |  |  |  |  | 它在部分序列有亮点，但整体还没有超过 GDINO+SAM2 Stage1。 |
| blackswan | `gdino_sam3_stage1_mask_only` | legacy |  | mask_only | exploratory | 0.9560 |  |  |  |  | 它在部分序列有亮点，但整体还没有超过 GDINO+SAM2 Stage1。 |
| car-shadow | `gdino_sam3_stage1_mask_only` | legacy |  | mask_only | exploratory | 0.9732 |  |  |  |  | 它在部分序列有亮点，但整体还没有超过 GDINO+SAM2 Stage1。 |
| horsejump-low | `gdino_sam3_stage1_mask_only` | legacy |  | mask_only | exploratory | 0.7609 |  |  |  |  | 它在部分序列有亮点，但整体还没有超过 GDINO+SAM2 Stage1。 |
| tennis | `gdino_sam3_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.7657 |  |  |  |  | 当前它比 GDINO+SAM3 Stage1 还低，说明现有重锚设计不够稳。 |
| bmx-trees | `gdino_sam3_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.3651 |  |  |  |  | 当前它比 GDINO+SAM3 Stage1 还低，说明现有重锚设计不够稳。 |
| blackswan | `gdino_sam3_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.9561 |  |  |  |  | 当前它比 GDINO+SAM3 Stage1 还低，说明现有重锚设计不够稳。 |
| car-shadow | `gdino_sam3_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.7122 |  |  |  |  | 当前它比 GDINO+SAM3 Stage1 还低，说明现有重锚设计不够稳。 |
| horsejump-low | `gdino_sam3_stage2_mask_only` | legacy |  | mask_only | exploratory | 0.6366 |  |  |  |  | 当前它比 GDINO+SAM3 Stage1 还低，说明现有重锚设计不够稳。 |
| tennis | `gdino_sam3_o2o_mask_only` | legacy |  | mask_only | promising | 0.7860 |  |  |  |  | 这是目前 GDINO+SAM3 创新里最有正向信号的一条。 |
| bmx-trees | `gdino_sam3_o2o_mask_only` | legacy |  | mask_only | promising | 0.4780 |  |  |  |  | 这是目前 GDINO+SAM3 创新里最有正向信号的一条。 |
| tennis | `gdino_sam3_quality_gate_mask_only` | legacy |  | mask_only | exploratory | 0.7657 |  |  |  |  | 现有阈值下几乎没带来增益，更像一次未充分触发的尝试。 |
| bmx-trees | `gdino_sam3_quality_gate_mask_only` | legacy |  | mask_only | exploratory | 0.3651 |  |  |  |  | 现有阈值下几乎没带来增益，更像一次未充分触发的尝试。 |
| tennis | `gdino_sam3_real_vlm_mask_only` | legacy |  | mask_only | exploratory | 0.7657 |  |  |  |  | 当前已验证能跑通，但已测序列上没有额外数值收益。 |
| bmx-trees | `gdino_sam3_real_vlm_mask_only` | legacy |  | mask_only | exploratory | 0.3651 |  |  |  |  | 当前已验证能跑通，但已测序列上没有额外数值收益。 |
| tennis | `direction_a_mask_fusion_mask_only` | legacy |  | mask_only | stable | 0.9468 |  |  |  |  | 这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。 |
| bmx-trees | `direction_a_mask_fusion_mask_only` | legacy |  | mask_only | stable | 0.7455 |  |  |  |  | 这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。 |
| blackswan | `direction_a_mask_fusion_mask_only` | legacy |  | mask_only | stable | 0.9549 |  |  |  |  | 这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。 |
| car-shadow | `direction_a_mask_fusion_mask_only` | legacy |  | mask_only | stable | 0.9749 |  |  |  |  | 这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。 |
| horsejump-low | `direction_a_mask_fusion_mask_only` | legacy |  | mask_only | stable | 0.8574 |  |  |  |  | 这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。 |
| koala | `direction_a_mask_fusion_mask_only` | legacy |  | mask_only | stable | 0.9482 |  |  |  |  | 这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。 |
| car-shadow | `direction_a_shadow_geom_scale_sweep_mask_only` | legacy |  | mask_only | failed_or_exploratory | 0.8231 |  |  |  |  | 当前扫参结果整体不如已有主线，说明这个先验还没有调到能正向贡献的程度。 |
| tennis | `vggt4d_vggt_mask_only` | legacy |  | mask_only | exploratory | 0.7571 |  |  |  |  | 它覆盖面有启发，但原始边缘比较粗，单独用还不够强。 |
| bmx-trees | `vggt4d_vggt_mask_only` | legacy |  | mask_only | exploratory | 0.4421 |  |  |  |  | 它覆盖面有启发，但原始边缘比较粗，单独用还不够强。 |
| blackswan | `vggt4d_vggt_mask_only` | legacy |  | mask_only | exploratory | 0.2082 |  |  |  |  | 它覆盖面有启发，但原始边缘比较粗，单独用还不够强。 |
| car-shadow | `vggt4d_vggt_mask_only` | legacy |  | mask_only | exploratory | 0.7589 |  |  |  |  | 它覆盖面有启发，但原始边缘比较粗，单独用还不够强。 |
| horsejump-low | `vggt4d_vggt_mask_only` | legacy |  | mask_only | exploratory | 0.6438 |  |  |  |  | 它覆盖面有启发，但原始边缘比较粗，单独用还不够强。 |
| koala | `vggt4d_vggt_mask_only` | legacy |  | mask_only | exploratory | 0.2353 |  |  |  |  | 它覆盖面有启发，但原始边缘比较粗，单独用还不够强。 |
| tennis | `pi3_transplant_v3_mask_only` | legacy |  | mask_only | failed | 0.0001 |  |  |  |  | 当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。 |
| bmx-trees | `pi3_transplant_v3_mask_only` | legacy |  | mask_only | failed | 0.0003 |  |  |  |  | 当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。 |
| blackswan | `pi3_transplant_v3_mask_only` | legacy |  | mask_only | failed | 0.0000 |  |  |  |  | 当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。 |
| car-shadow | `pi3_transplant_v3_mask_only` | legacy |  | mask_only | failed | 0.0001 |  |  |  |  | 当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。 |
| horsejump-low | `pi3_transplant_v3_mask_only` | legacy |  | mask_only | failed | 0.0006 |  |  |  |  | 当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。 |
| koala | `pi3_transplant_v3_mask_only` | legacy |  | mask_only | failed | 0.0568 |  |  |  |  | 当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。 |
| tennis | `vggt4d_sam3_refine_v2_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。 |
| bmx-trees | `vggt4d_sam3_refine_v2_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。 |
| blackswan | `vggt4d_sam3_refine_v2_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。 |
| car-shadow | `vggt4d_sam3_refine_v2_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。 |
| horsejump-low | `vggt4d_sam3_refine_v2_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。 |
| koala | `vggt4d_sam3_refine_v2_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。 |
| tennis | `vggt4d_sam3_refine_v3_mask_only` | legacy |  | mask_only | partial_or_failed | 0.0000 |  |  |  |  | 它是中间版本，应该保留在台账里，但不能当成最终结论。 |
| bmx-trees | `vggt4d_sam3_refine_v3_mask_only` | legacy |  | mask_only | partial_or_failed | 0.0000 |  |  |  |  | 它是中间版本，应该保留在台账里，但不能当成最终结论。 |
| blackswan | `vggt4d_sam3_refine_v3_mask_only` | legacy |  | mask_only | partial_or_failed | 0.0000 |  |  |  |  | 它是中间版本，应该保留在台账里，但不能当成最终结论。 |
| car-shadow | `vggt4d_sam3_refine_v3_mask_only` | legacy |  | mask_only | partial_or_failed | 0.0000 |  |  |  |  | 它是中间版本，应该保留在台账里，但不能当成最终结论。 |
| horsejump-low | `vggt4d_sam3_refine_v3_mask_only` | legacy |  | mask_only | partial_or_failed | 0.0000 |  |  |  |  | 它是中间版本，应该保留在台账里，但不能当成最终结论。 |
| koala | `vggt4d_sam3_refine_v3_mask_only` | legacy |  | mask_only | partial_or_failed | 0.0000 |  |  |  |  | 它是中间版本，应该保留在台账里，但不能当成最终结论。 |
| tennis | `vggt4d_sam3_refine_v4_mask_only` | legacy |  | mask_only | superseded | 0.0000 |  |  |  |  | 它接近可用，但最终汇总主要还是用 v5。 |
| bmx-trees | `vggt4d_sam3_refine_v4_mask_only` | legacy |  | mask_only | superseded | 0.5396 |  |  |  |  | 它接近可用，但最终汇总主要还是用 v5。 |
| blackswan | `vggt4d_sam3_refine_v4_mask_only` | legacy |  | mask_only | superseded | 0.9558 |  |  |  |  | 它接近可用，但最终汇总主要还是用 v5。 |
| car-shadow | `vggt4d_sam3_refine_v4_mask_only` | legacy |  | mask_only | superseded | 0.0013 |  |  |  |  | 它接近可用，但最终汇总主要还是用 v5。 |
| horsejump-low | `vggt4d_sam3_refine_v4_mask_only` | legacy |  | mask_only | superseded | 0.0002 |  |  |  |  | 它接近可用，但最终汇总主要还是用 v5。 |
| koala | `vggt4d_sam3_refine_v4_mask_only` | legacy |  | mask_only | superseded | 0.9441 |  |  |  |  | 它接近可用，但最终汇总主要还是用 v5。 |
| tennis | `vggt4d_sam3_refine_v5_mask_only` | legacy |  | mask_only | stable | 0.8735 |  |  |  |  | 这是 Direction B 当前最有说服力的正向结果。 |
| bmx-trees | `vggt4d_sam3_refine_v5_mask_only` | legacy |  | mask_only | stable | 0.6887 |  |  |  |  | 这是 Direction B 当前最有说服力的正向结果。 |
| blackswan | `vggt4d_sam3_refine_v5_mask_only` | legacy |  | mask_only | stable | 0.9558 |  |  |  |  | 这是 Direction B 当前最有说服力的正向结果。 |
| car-shadow | `vggt4d_sam3_refine_v5_mask_only` | legacy |  | mask_only | stable | 0.9785 |  |  |  |  | 这是 Direction B 当前最有说服力的正向结果。 |
| horsejump-low | `vggt4d_sam3_refine_v5_mask_only` | legacy |  | mask_only | stable | 0.9331 |  |  |  |  | 这是 Direction B 当前最有说服力的正向结果。 |
| koala | `vggt4d_sam3_refine_v5_mask_only` | legacy |  | mask_only | stable | 0.9373 |  |  |  |  | 这是 Direction B 当前最有说服力的正向结果。 |
| tennis | `pure_propainter_gtmask` | legacy |  | inpaint_only | reference |  | 34.9782 | 0.9298 |  |  | 这是 GT mask 协议下 inpaint-only 的基准。 |
| bmx-trees | `pure_propainter_gtmask` | legacy |  | inpaint_only | reference |  |  |  |  |  | 这是 GT mask 协议下 inpaint-only 的基准。 |
| blackswan | `pure_propainter_gtmask` | legacy |  | inpaint_only | reference |  |  |  |  |  | 这是 GT mask 协议下 inpaint-only 的基准。 |
| koala | `pure_propainter_gtmask` | legacy |  | inpaint_only | reference |  |  |  |  |  | 这是 GT mask 协议下 inpaint-only 的基准。 |
| horsejump-low | `pure_propainter_gtmask` | legacy |  | inpaint_only | reference |  |  |  |  |  | 这是 GT mask 协议下 inpaint-only 的基准。 |
| car-shadow | `pure_propainter_gtmask` | legacy |  | inpaint_only | reference |  |  |  |  |  | 这是 GT mask 协议下 inpaint-only 的基准。 |
| tennis | `sdxl_kf5_gtmask_propainter` | legacy |  | inpaint_only | stable |  | 37.5066 | 0.9726 |  |  | GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。 |
| bmx-trees | `sdxl_kf5_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。 |
| blackswan | `sdxl_kf5_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。 |
| koala | `sdxl_kf5_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。 |
| horsejump-low | `sdxl_kf5_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。 |
| car-shadow | `sdxl_kf5_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。 |
| tennis | `lama_gtmask_propainter` | legacy |  | inpaint_only | stable |  | 37.6036 | 0.9720 |  |  | GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。 |
| bmx-trees | `lama_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。 |
| blackswan | `lama_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。 |
| koala | `lama_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。 |
| horsejump-low | `lama_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。 |
| car-shadow | `lama_gtmask_propainter` | legacy |  | inpaint_only | stable |  |  |  |  |  | GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。 |
| tennis | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  | 34.8786 | nan |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| bmx-trees | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| blackswan | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| car-shadow | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| horsejump-low | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| koala | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| wild_video-1person | `pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是 inpaint-only 的强基线，经常是最难被超过的一条。 |
| tennis | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  | 34.2062 | nan |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| bmx-trees | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| blackswan | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| car-shadow | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| horsejump-low | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| koala | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| wild_video-1person | `sdxl_kf5_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。 |
| tennis | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  | 34.6778 | nan |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| bmx-trees | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| blackswan | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| car-shadow | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| horsejump-low | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| koala | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| wild_video-1person | `lama_propainter_fixed_mask` | legacy |  | inpaint_only | stable |  |  |  |  |  | 它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。 |
| tennis | `controlnet_pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable | 1.0000 | 34.9782 | 0.9945 |  |  | 它是 ControlNet 消融里的基线方法，经常被推荐为最终选择。 |
| bmx-trees | `controlnet_pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable | 1.0000 | 35.0082 | 0.9975 |  |  | 它是 ControlNet 消融里的基线方法，经常被推荐为最终选择。 |
| koala | `controlnet_pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable | 1.0000 | 30.6999 | 0.9920 |  |  | 它是 ControlNet 消融里的基线方法，经常被推荐为最终选择。 |
| bear | `controlnet_pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable | 1.0000 | 31.0590 | 0.9875 |  |  | 它是 ControlNet 消融里的基线方法，经常被推荐为最终选择。 |
| camel | `controlnet_pure_propainter_fixed_mask` | legacy |  | inpaint_only | stable | 1.0000 | 30.2338 | 0.9933 |  |  | 它是 ControlNet 消融里的基线方法，经常被推荐为最终选择。 |
| tennis | `controlnet_hybrid_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 34.6397 | 0.9929 |  |  | 它能展示思路，但当前大多数情况下不如纯 ProPainter 稳。 |
| bmx-trees | `controlnet_hybrid_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 34.5878 | 0.9962 |  |  | 它能展示思路，但当前大多数情况下不如纯 ProPainter 稳。 |
| koala | `controlnet_hybrid_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 30.6104 | 0.9918 |  |  | 它能展示思路，但当前大多数情况下不如纯 ProPainter 稳。 |
| bear | `controlnet_hybrid_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 30.7198 | 0.9831 |  |  | 它能展示思路，但当前大多数情况下不如纯 ProPainter 稳。 |
| camel | `controlnet_hybrid_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 29.9429 | 0.9917 |  |  | 它能展示思路，但当前大多数情况下不如纯 ProPainter 稳。 |
| tennis | `controlnet_hybrid_tc_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 34.9748 | 0.9945 |  |  | 这是个合理尝试，但目前仍然很难整体超过纯 ProPainter。 |
| bmx-trees | `controlnet_hybrid_tc_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 35.0024 | 0.9975 |  |  | 这是个合理尝试，但目前仍然很难整体超过纯 ProPainter。 |
| koala | `controlnet_hybrid_tc_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 30.7103 | 0.9920 |  |  | 这是个合理尝试，但目前仍然很难整体超过纯 ProPainter。 |
| bear | `controlnet_hybrid_tc_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 31.0667 | 0.9876 |  |  | 这是个合理尝试，但目前仍然很难整体超过纯 ProPainter。 |
| camel | `controlnet_hybrid_tc_propainter_fixed_mask` | legacy |  | inpaint_only | exploratory | 1.0000 | 30.2324 | 0.9933 |  |  | 这是个合理尝试，但目前仍然很难整体超过纯 ProPainter。 |
| tennis | `sdxl_interval10_legacy_fixed_mask` | legacy |  | inpaint_only | legacy | 0.7860 | 32.6079 | 0.9925 |  |  | 它应该保留在台账里，但属于旧版对照，不是现在主推配置。 |
| bmx-trees | `sdxl_interval10_legacy_fixed_mask` | legacy |  | inpaint_only | legacy | 0.4690 | 32.1659 | 0.9960 |  |  | 它应该保留在台账里，但属于旧版对照，不是现在主推配置。 |
| koala | `koala_void_full_pipeline` | legacy |  | full_pipeline | exploratory |  | 13.1300 |  |  |  | 它可以作为创新方向保留，但当前更像探索性结果，不适合直接当稳定主线。 |
| tennis | `a_plus_b_best_full_pipeline` | legacy |  | full_pipeline | stable | 0.9468 | 35.1285 | 0.9264 |  |  | 它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。 |
| bmx-trees | `a_plus_b_best_full_pipeline` | legacy |  | full_pipeline | stable | 0.7455 | 35.2173 | 0.9432 |  |  | 它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。 |
| blackswan | `a_plus_b_best_full_pipeline` | legacy |  | full_pipeline | stable | 0.9558 | 33.4822 | 0.8844 |  |  | 它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。 |
| car-shadow | `a_plus_b_best_full_pipeline` | legacy |  | full_pipeline | stable | 0.9785 | 37.4591 | 0.9193 |  |  | 它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。 |
| horsejump-low | `a_plus_b_best_full_pipeline` | legacy |  | full_pipeline | stable | 0.9331 | 33.4754 | 0.9085 |  |  | 它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。 |
| koala | `a_plus_b_best_full_pipeline` | legacy |  | full_pipeline | stable | 0.9482 | 32.2091 | 0.8346 |  |  | 它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。 |
| tennis | `diffueraser_gtmask_v1` | v1 | davis_gt | inpaint_only | exploratory |  |  |  | pure_propainter_gtmask | 三版停止条件已触发（v1/v2/v3 均低于基线 3.5-4.5 dB）。DiffuEraser 不扩序列，不进 full_pipeline。归档为 exploratory。后续可探索修改 soft-blending 策略或使用更大 neighbor_length。 | v1 最终状态（三版停止决策后）：PSNR_proxy=31.42 vs 基线 34.98（-3.56dB），SSIM=0.897 vs 0.930（-0.033）。v2/v3 均未改善。根本原因：DiffuEraser soft-blending 污染非 mask 区域。决策：停止扩序列，不进 full_pipeline，归档 exploratory。PSNR_synthetic (mask 区域) ≈ ProPainter，mask 内修复质量相当但无提升。 |
| tennis | `diffueraser_gtmask_v3` | v3 | davis_gt | inpaint_only | exploratory |  |  |  | diffueraser_gtmask_v2 | 三版停止条件已触发。DiffuEraser 不扩序列，不进 full_pipeline。归档为 exploratory。后续可探索修改 blending 策略或更大 neighbor_length。 | DiffuEraser tennis 三版验证均未过门槛（v1: -3.56dB, v2: -3.66dB, v3: -4.56dB）。根本原因：soft-blended diffusion 输出影响非 mask 区域。决策：停止扩序列，不进 full_pipeline，归档为 exploratory 供参考。 |
| tennis | `diffueraser_gtmask_v4` | v4 | davis_gt | inpaint_only | exploratory |  |  |  | diffueraser_gtmask_v1 | v4 通过 PSNR_proxy 门槛（36.28 ≥ 34.88）。下一步：扩到 bmx-trees 和 car-shadow 验证泛化性。 | v4 结果（硬贴回，从 DAVIS JPEG 直接取非 mask 像素）：PSNR_proxy=36.28（+1.40dB vs 基线 34.88），通过 PSNR_proxy 门槛！PSNR_synthetic=32.88（mask 区域修复质量高，与 ProPainter prior 一致）。白色虚影问题已通过硬贴回解决。决策：v4 通过门槛，可扩到其他序列。 |
| tennis | `diffueraser_gtmask_v5` | v5 | davis_gt | inpaint_only | exploratory |  |  |  | pure_propainter_gtmask | 未过门槛，归档 exploratory。soft-blending 根本问题需 v4 类硬贴回解决。 | v5 结果：PSNR_proxy=31.32（与 v1 31.32 相近），PSNR_synthetic=8.42。扩大上下文窗口未显著改善 PSNR_proxy，根本原因仍是 soft-blending 泄漏。决策：归档 exploratory，不扩序列。 |
| tennis | `diffueraser_gtmask_v6` | v6 | davis_gt | inpaint_only | exploratory |  |  |  | pure_propainter_gtmask | 未过门槛，归档 exploratory。soft-blending 根本问题与分辨率无关。 | v6 结果：PSNR_proxy=31.32（与 v1 相近），PSNR_synthetic=8.43。原分辨率推理未改善 PSNR_proxy，确认根本问题是 soft-blending 架构，与分辨率无关。决策：归档 exploratory。 |
| tennis | `diffueraser_gtmask_v7` | v7 | davis_gt | inpaint_only | exploratory |  |  |  | pure_propainter_gtmask | 未过门槛（32.46 < 34.88），但有改善趋势。归档 exploratory。若需进一步探索可结合 v4 硬贴回思路。 | v7 结果：PSNR_proxy=32.46（-2.43dB vs 基线 34.88），PSNR_synthetic=13.10。最紧 mask 有所改善（vs v1 -3.56dB），但仍未过门槛。PSNR_synthetic 13.10 比 v1 的 8.42 高，说明更紧 mask 使 mask 内更多区域保留了原始像素。决策：有改善趋势，但仍归档 exploratory。 |
| tennis | `diffueraser_gtmask_v8` | v8 | davis_gt | inpaint_only | exploratory |  |  |  | pure_propainter_gtmask | 未过门槛，归档 exploratory。8 版 DiffuEraser 调参探索完毕。结论：仅 v4（硬贴回）通过门槛，其余版本均因 soft-blending 架构问题失败。 | v8 结果：PSNR_proxy=31.31（与 v1 相近），PSNR_synthetic=8.41。密参考帧未改善 PSNR_proxy，soft-blending 根本问题不受 ref_stride 影响。决策：归档 exploratory。8 版调参方向已全部探索完毕。 |
| tennis | `diffueraser_smoke_v1` | v1 | davis_gt | inpaint_only | exploratory |  |  |  | pure_propainter_gtmask | smoke 通过后运行完整 v1 推理 | 待 smoke test 运行后更新 |
| tennis | `diffueraser_gtmask_v2` | v2 | davis_gt | inpaint_only | exploratory |  |  |  | diffueraser_gtmask_v1 | DiffuEraser tennis 三版验证结论（2026-05）：
v1 (dilation=0): PSNR_proxy=31.42 (-3.56dB vs 基线 34.98), SSIM=0.897
v2 (dilation=8): PSNR_proxy=31.32 (-3.66dB), SSIM=0.894 — 与 v1 无显著差异
v3 (dilation=4, max_img_size=640): PSNR_proxy=30.42 (-4.56dB), SSIM=0.871 — 更差
根本原因：DiffuEraser 使用 soft-blended diffusion 输出，边界 feathering 会轻微污染非 mask 区域。
PSNR_synthetic（mask 区域）三版均≈8.35，与 ProPainter 持平，说明 mask 内修复质量相当但无提升。
决策：停止扩序列，归档为 exploratory。不进入 full_pipeline 主线。
后续可考虑：(1) 修改 blending 策略使其保留背景; (2) 探索更大 neighbor_length 参数。 | v2 (dilation=8, size=960): PSNR_proxy=31.32, SSIM=0.8941 — 系统性低于基线 3.5-4.5 dB。 |
