# wild_video-1person

- 这个素材难点：真实视频场景，人物、书包、阴影都会互相干扰，没有 DAVIS 那么干净。
- 先看 `masked_in.mp4` 判断 mask，再看 `inpaint_out.mp4` 判断修复。
- `mask_only` 看遮罩质量；`inpaint_only` 看固定 mask 下修复工具；`full_pipeline` 看最终交付视频。

## 当前已整理的方法

- `part2_mask_baseline` | mask_only | reference | Part2 YOLO+SAM2 基线 mask
  这条方法一句话：这是 Part2 的历史基线，只看遮罩本身，不看后续修复。
- `part2_baseline_full_pipeline` | full_pipeline | reference | Part2 YOLO+SAM2+ProPainter 基线完整流程
  这条方法一句话：这是旧基线：YOLO+SAM2 先出 mask，再用 ProPainter 修复。
- `sam3_multiobj_mask_only` | mask_only | stable | SAM3 multi-object 联合 prompt mask
  这条方法一句话：这是 Part3 Direction A 的主力版本：多个 prompt 一起提示，再把所有对象的 mask 取并集。
- `sam3_multiobj_propainter_full_pipeline` | full_pipeline | stable | SAM3 multi-object + ProPainter 完整流程
  这条方法一句话：这是把 SAM3 multi-object mask 接到 ProPainter 后的完整视频结果。
- `sam3_rebuild_v1_mask_only` | mask_only | exploratory | SAM3 rebuild v1 mask
  这条方法一句话：这是后来重建的一条 SAM3 主线，和 multi-object 不是完全同一个产物来源。
- `sam3_rebuild_v1_propainter_full_pipeline` | full_pipeline | exploratory | SAM3 rebuild v1 + ProPainter
  这条方法一句话：这是 rebuild v1 接 ProPainter 的完整流程，用来和 multi-object 路线分开看。
- `pure_propainter_fixed_mask` | inpaint_only | stable | 纯 ProPainter（旧版 SAM3 mask，legacy）
  这条方法一句话：这里固定的是同一套 mask，只比较 ProPainter 自己的修复能力。
- `sdxl_kf5_propainter_fixed_mask` | inpaint_only | stable | SDXL kf5 + ProPainter（旧版 SAM3 mask，legacy）
  这条方法一句话：这是先用 SDXL 修关键帧，再让 ProPainter 传播到全视频的路线。
- `lama_propainter_fixed_mask` | inpaint_only | stable | LaMa + ProPainter（旧版 SAM3 mask，legacy）
  这条方法一句话：这是先用 LaMa 修关键区域，再交给 ProPainter 做时序传播。
