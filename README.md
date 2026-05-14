# CV Project 2026: Video Object Removal and Inpainting

**AIAA 3201 -- Introduction to Computer Vision, Spring 2026**

This project implements a complete video object removal and background restoration pipeline,
comparing classical CV methods against modern foundation-model approaches.

Report: `report/project3_video_object_removal_report.tex`

---

## Summary

| Part | Method | Status |
|---|---|---|
| Part 1 | YOLOv8-Seg + Lucas-Kanade optical flow + temporal background propagation + cv2.inpaint | Done |
| Part 2 | YOLOv8-Seg (bbox prompt) -> SAM 2.1 -> ProPainter | Done |
| Part 3 | GDINO/VLM + SAM2/SAM3 + DiffuEraser + MiniMax-Remover + shadow-aware masks | Done |

**Core finding**: Part 1 temporal propagation fails under strong camera motion (e.g. `bmx-trees`),
producing ghosting. Part 2 ProPainter handles this via flow-guided propagation.
Part 3 improves mask completeness with semantic prompting (SAM3/GDINO/VLM),
geometry-motion refinement (VGGT4D+SAM3), and native video diffusion (MiniMax-Remover).

---

## Repository Structure

```
project3/
+-- README.md
+-- part1/         Classical CV baseline (YOLOv8 + LK flow + cv2.inpaint)
+-- part2/         AI pipeline (SAM2 + ProPainter)
+-- part3/         Foundation-model pipeline
|   +-- direction_a/    SAM3/GDINO/VLM mask upgrade
|   +-- direction_b/    VGGT4D + SAM3 refinement
|   +-- inpainting/     ProPainter, DiffuEraser, MiniMax-Remover scripts
|   +-- eval/           Per-experiment evaluation
|   +-- reporting/      Result collection
|   +-- configs/        Sequence-level YAML configs
|   +-- gdino_vlm/      GDINO/VLM sub-experiments
+-- eval/          Unified DAVIS evaluation entry point
+-- report/        LaTeX report source
```

---

## Part 3 Results (DAVIS5 Macro JM)

| Part | Method | Macro JM |
|---|---|---:|
| Part 1 | YOLO + LK + cv2.inpaint | 0.4922 |
| Part 2 | YOLO + SAM2 + ProPainter | 0.8451 |
| Part 3-A | SAM3 multi-object union prompts | 0.8561 |
| Part 3-B5 | VGGT4D + SAM3 refine | 0.8859 |
| **Part 3 A+B Best** | Scene-adaptive routing | **0.9119** |

---

## Quick Start

Clone the repo and set up conda environments as described in each part README.
Download DAVIS from https://davischallenge.org. See `part3/README.md` for Part 3 usage.

Run unified DAVIS evaluation:

```bash
conda activate part1_env
python eval/eval_davis_masks.py --policy eval/davis_eval_targets.yaml \
  --output_csv eval/results_davis_masks.csv
```

---

## Hardware

- 2x NVIDIA RTX A6000 (49 GB VRAM each), CUDA 12.8, Ubuntu 22.04, Python 3.10
- ProPainter on `bmx-trees` (80 frames, 854x480) needs ~12 GB VRAM. Add `--resize_ratio 0.5` if OOM.

---

## Acknowledgements

YOLOv8 (Ultralytics), SAM 2/3 (Meta AI), ProPainter (Zhou et al., ICCV 2023),
DiffuEraser (Li et al., 2025), MiniMax-Remover (zibojia/minimax-remover), VGGT4D (Hu et al., 2025),
DAVIS (Pont-Tuset et al.). Full citations in the report.

## License

Code: MIT. Pre-trained weights follow their respective upstream licenses.
