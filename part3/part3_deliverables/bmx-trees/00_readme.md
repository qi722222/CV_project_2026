# bmx-trees

- 这个素材难点：树枝和骑手/自行车交错，目标细长且运动快，是目前最难的弱点序列。
- 先看 `masked_in.mp4` 判断 mask，再看 `inpaint_out.mp4` 判断修复。
- `mask_only` 看遮罩质量；`inpaint_only` 看固定 mask 下修复工具；`full_pipeline` 看最终交付视频。

## 当前已整理的方法

- `part2_mask_baseline` | mask_only | reference | Part2 YOLO+SAM2 基线 mask
  这条方法一句话：这是 Part2 的历史基线，只看遮罩本身，不看后续修复。
- `part2_baseline_full_pipeline` | full_pipeline | reference | Part2 YOLO+SAM2+ProPainter 基线完整流程
  这条方法一句话：这是旧基线：YOLO+SAM2 先出 mask，再用 ProPainter 修复。
- `official_sam3_video_mask_only` | mask_only | exploratory | 官方 SAM3 单 prompt 视频分割
  这条方法一句话：这是最直接的官方 SAM3 文本 prompt 版本，一条 prompt 直接做视频分割。
- `official_sam3_best_mask_only` | mask_only | exploratory | 官方 SAM3 best-prompt 版本
  这条方法一句话：这是官方 SAM3 的调过 prompt 版本，用来回答“只换 prompt 能不能变好”。
- `sam3_multiobj_mask_only` | mask_only | stable | SAM3 multi-object 联合 prompt mask
  这条方法一句话：这是 Part3 Direction A 的主力版本：多个 prompt 一起提示，再把所有对象的 mask 取并集。
- `sam3_multiobj_propainter_full_pipeline` | full_pipeline | stable | SAM3 multi-object + ProPainter 完整流程
  这条方法一句话：这是把 SAM3 multi-object mask 接到 ProPainter 后的完整视频结果。
- `sam3_rebuild_v1_mask_only` | mask_only | exploratory | SAM3 rebuild v1 mask
  这条方法一句话：这是后来重建的一条 SAM3 主线，和 multi-object 不是完全同一个产物来源。
- `sam3_rebuild_v1_propainter_full_pipeline` | full_pipeline | exploratory | SAM3 rebuild v1 + ProPainter
  这条方法一句话：这是 rebuild v1 接 ProPainter 的完整流程，用来和 multi-object 路线分开看。
- `gdino_sam2_stage1_mask_only` | mask_only | stable | GDINO + SAM2 Stage1 mask
  这条方法一句话：GDINO 先根据文字找框，再交给 SAM2 做 mask 和传播。这是比较稳的历史主线。
- `gdino_sam2_stage2_mask_only` | mask_only | exploratory | GDINO + SAM2 Stage2 稀疏重锚 mask
  这条方法一句话：这条路线尝试在视频中间重新找框，想解决长视频漂移问题。
- `gdino_sam3_stage1_mask_only` | mask_only | exploratory | GDINO + SAM3 Stage1 mask
  这条方法一句话：GDINO 负责找框，SAM3 负责分割和传播，用来回答“把 SAM2 换成 SAM3 是否整体变强”。
- `gdino_sam3_stage2_mask_only` | mask_only | exploratory | GDINO + SAM3 Stage2 稀疏重锚 mask
  这条方法一句话：这是 GDINO+SAM3 再加稀疏重锚的版本，原本想提高长时稳定性。
- `gdino_sam3_o2o_mask_only` | mask_only | promising | GDINO + SAM3 + O2O 时序关联
  这条方法一句话：这条创新是给相邻帧做更严格的一对一关联，减少 ID 混乱和错配。
- `gdino_sam3_quality_gate_mask_only` | mask_only | exploratory | GDINO + SAM3 + QualityGate
  这条方法一句话：QualityGate 想做的是：质量太差时，不要瞎更新。
- `gdino_sam3_real_vlm_mask_only` | mask_only | exploratory | GDINO + SAM3 + RealVLM prompt
  这条方法一句话：这条路线用真实 VLM caption 来生成 prompt，而不是完全手写规则。
- `direction_a_mask_fusion_mask_only` | mask_only | stable | Direction A 三路/两路 mask 融合
  这条方法一句话：这是把不同 mask 路线按序列或逐帧策略做融合的版本。
- `vggt4d_vggt_mask_only` | mask_only | exploratory | VGGT4D 原始动态发现 mask
  这条方法一句话：这是 Direction B 的原始 baseline：不靠文字 prompt，直接从 3D foundation model 的时序动态线索里找运动物体。
- `pi3_transplant_v3_mask_only` | mask_only | failed | Pi3 transplant v3 mask
  这条方法一句话：这是把 VGGT4D 的思路移植到 Pi3 上的尝试，理论上想用更强 backbone 换更好 mask。
- `vggt4d_sam3_refine_v2_mask_only` | mask_only | superseded | VGGT4D + SAM3 refine v2
  这条方法一句话：这条路线是把 VGGT4D 粗 mask 当作 SAM3 的空间提示来精修边界。
- `vggt4d_sam3_refine_v3_mask_only` | mask_only | partial_or_failed | VGGT4D + SAM3 refine v3
  这条方法一句话：这是 refine 的中间版本，部分序列仍有 prompt 失败或不稳定问题。
- `vggt4d_sam3_refine_v4_mask_only` | mask_only | superseded | VGGT4D + SAM3 refine v4
  这条方法一句话：这是 refine 的较成熟版本，已经比早期版本稳很多。
- `vggt4d_sam3_refine_v5_mask_only` | mask_only | stable | VGGT4D + SAM3 refine v5
  这条方法一句话：这是当前 Direction B 最终保留的 refine 版本：先自动发现动态区域，再用 SAM3 把边缘精修。
- `pure_propainter_gtmask` | inpaint_only | reference | 纯 ProPainter（DAVIS GT mask 统一口径）
  这条方法一句话：所有 DAVIS 序列统一使用 DAVIS annotation / GT mask 作为输入，只比较 ProPainter 自己的修复能力。这是公平 inpaint-only 对比的基线。
- `sdxl_kf5_gtmask_propainter` | inpaint_only | stable | SDXL kf5 + ProPainter（DAVIS GT mask 统一口径）
  这条方法一句话：DAVIS 序列统一使用 DAVIS GT mask；先用 SDXL 修关键帧，再让 ProPainter 传播到全视频。与 pure_propainter_gtmask 做公平对比。
- `lama_gtmask_propainter` | inpaint_only | stable | LaMa + ProPainter（DAVIS GT mask 统一口径）
  这条方法一句话：DAVIS 序列统一使用 DAVIS GT mask；先用 LaMa 修关键区域，再交给 ProPainter 做时序传播。与 pure_propainter_gtmask 做公平对比。
- `pure_propainter_fixed_mask` | inpaint_only | stable | 纯 ProPainter（旧版 SAM3 mask，legacy）
  这条方法一句话：这里固定的是同一套 mask，只比较 ProPainter 自己的修复能力。
- `sdxl_kf5_propainter_fixed_mask` | inpaint_only | stable | SDXL kf5 + ProPainter（旧版 SAM3 mask，legacy）
  这条方法一句话：这是先用 SDXL 修关键帧，再让 ProPainter 传播到全视频的路线。
- `lama_propainter_fixed_mask` | inpaint_only | stable | LaMa + ProPainter（旧版 SAM3 mask，legacy）
  这条方法一句话：这是先用 LaMa 修关键区域，再交给 ProPainter 做时序传播。
- `controlnet_pure_propainter_fixed_mask` | inpaint_only | stable | ControlNet 消融：纯 ProPainter
  这条方法一句话：这是在 ControlNet 消融里保留原始帧、直接走 ProPainter 的对照组。
- `controlnet_hybrid_propainter_fixed_mask` | inpaint_only | exploratory | ControlNet 消融：hybrid
  这条方法一句话：这是先做关键帧生成式修复，再与原视频拼成 hybrid frames 后交给 ProPainter。
- `controlnet_hybrid_tc_propainter_fixed_mask` | inpaint_only | exploratory | ControlNet 消融：hybrid + temporal consistency
  这条方法一句话：这是在 hybrid 的基础上，再加一层更保守的时序一致性处理。
- `sdxl_interval10_legacy_fixed_mask` | inpaint_only | legacy | 旧版 SDXL interval10 + ProPainter
  这条方法一句话：这是更早一版的 SDXL 关键帧路线，间隔更大，只覆盖两条序列。
- `a_plus_b_best_full_pipeline` | full_pipeline | stable | A+B Best 按序列选优完整流程
  这条方法一句话：这是当前最强的汇总版本：每个序列选当下最好的 mask 来源，再接 ProPainter。
