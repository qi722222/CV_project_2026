# bear

- 这个素材难点：动物轮廓较大，但毛边和自然背景纹理容易混淆。
- 先看 `masked_in.mp4` 判断 mask，再看 `inpaint_out.mp4` 判断修复。
- `mask_only` 看遮罩质量；`inpaint_only` 看固定 mask 下修复工具；`full_pipeline` 看最终交付视频。

## 当前已整理的方法

- `sam3_rebuild_v1_mask_only` | mask_only | exploratory | SAM3 rebuild v1 mask
  这条方法一句话：这是后来重建的一条 SAM3 主线，和 multi-object 不是完全同一个产物来源。
- `sam3_rebuild_v1_propainter_full_pipeline` | full_pipeline | exploratory | SAM3 rebuild v1 + ProPainter
  这条方法一句话：这是 rebuild v1 接 ProPainter 的完整流程，用来和 multi-object 路线分开看。
- `controlnet_pure_propainter_fixed_mask` | inpaint_only | stable | ControlNet 消融：纯 ProPainter
  这条方法一句话：这是在 ControlNet 消融里保留原始帧、直接走 ProPainter 的对照组。
- `controlnet_hybrid_propainter_fixed_mask` | inpaint_only | exploratory | ControlNet 消融：hybrid
  这条方法一句话：这是先做关键帧生成式修复，再与原视频拼成 hybrid frames 后交给 ProPainter。
- `controlnet_hybrid_tc_propainter_fixed_mask` | inpaint_only | exploratory | ControlNet 消融：hybrid + temporal consistency
  这条方法一句话：这是在 hybrid 的基础上，再加一层更保守的时序一致性处理。
