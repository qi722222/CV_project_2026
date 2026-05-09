# SDXL kf5 + ProPainter（wild demo，SAM3 shadow mask）

## 这是什么

wild 视频没有 DAVIS GT，继续用 SAM3 shadow mask；仅作为 demo 展示用，不参与 DAVIS GT 公平对比。

## 这条实验属于哪一类

- `inpaint_only`
- 家族：Direction C（Inpainting）
- 审计状态：`stable`

## 你看这个实验时重点看什么

检查修复视觉效果，不能直接和 DAVIS GT 组的数字混排。

## 当前结论

Demo 分组，mask_protocol=wild_existing_mask，不纳入 GT inpaint 公平对比表。

## 关键路径

- `mask_frames`: `/data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/wild_video-1person_with_shadow`
- `source_output_dir`: `/data3/jli657/project3/part3/results/wild_video-1person/direction_c/sdxl_kf5_propainter`
- `inpaint_out`: `/data3/jli657/project3/part3/results/wild_video-1person/direction_c/sdxl_kf5_propainter/inpaint_out.mp4`
- `masked_in`: `/data3/jli657/project3/part3/results/wild_video-1person/direction_c/sdxl_kf5_propainter/masked_in.mp4`
- `script_path`: `/data3/jli657/project3/part3/run_phase2_sdxl_all7.py`

## 指标摘录

- 当前未在统一指标表里找到对应数值。
