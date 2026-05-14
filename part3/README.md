# Part 3: Foundation-Model Video Object Removal

## Overview

Part 3 extends the Part 2 pipeline with three innovation directions to improve mask quality and
inpainting completeness:

- **Direction A (SAM3 Mask Upgrade)**: Replace fixed YOLO category detection with open-vocabulary
  prompting. Grounding-DINO (GDINO) generates text-conditioned boxes, which are passed to
  SAM2/SAM3 for pixel-level masks. Ablation of QualityGate / One-to-One (O2O) / RealVLM
  innovations is included.
- **Direction B (Geometry+Motion Mask)**: VGGT4D provides zero-shot dynamic object segmentation
  from motion and geometry cues. SAM3 refines the resulting coarse masks.
- **Direction C (Inpainting Ablation, fixed GT mask)**: Compares ProPainter, LaMa, SDXL keyframe
  repair, ControlNet, and DiffuEraser under identical DAVIS GT masks to isolate the inpainting
  backend from mask quality.
- **Direction D (MiniMax-Remover + Shadow-Aware Masks)**: Native video diffusion inpainter
  (MiniMax-Remover) paired with improved shadow-aware masks (`shadow_edge_v1`, `shadow_object_v2`)
  for more complete shadow removal. Qualitative results available; pending full quantitative eval.

**DAVIS5 Macro JM results:**

| Part | Method | Macro JM |
|---|---|---:|
| Part 1 | YOLO + Lucas-Kanade + cv2.inpaint | 0.4922 |
| Part 2 | YOLO + SAM2 + ProPainter | 0.8451 |
| **Part 3 A+B Best** | GDINO/VLM+SAM3 union + VGGT4D+SAM3 refine + ProPainter | **0.9119** |

---

## Directory Structure

```
part3/
+-- README.md
+-- requirements_controlnet.txt
+-- direction_a/
|   +-- run_sam3_multiobject.py         SAM3 multi-object mask (main pipeline)
|   +-- run_part3_refine.py             ControlNet keyframe refinement
|   +-- run_controlnet_ablation_5seq.py ControlNet ablation
|   +-- run_part3_sam3_rebuild.py       SAM3 rebuild v1
|   +-- run_sam3_prompt_search.py       Prompt policy search
|   +-- extend_shadow_mask.py           Shadow geometry extension (Direction A shadow)
+-- direction_b/
|   +-- run_direction_b_vggt4d.py       VGGT4D unsupervised discovery
|   +-- run_direction_b_sam3_refine.py  VGGT4D + SAM3 boundary refinement
|   +-- run_direction_b_comparison.py   Direction B evaluation
|   +-- run_direction_b_vlm_sam3.py     VLM-guided SAM3 variant
+-- inpainting/
|   +-- run_propainter_gtmask.py        ProPainter + DAVIS GT mask
|   +-- run_diffueraser_gtmask.py       DiffuEraser + GT mask (v8/v9 final)
|   +-- apply_hard_blend.py             DiffuEraser hard-blend post-processing fix
|   +-- run_minimax_remover_gtmask.py   MiniMax-Remover + GT/shadow-aware masks
|   +-- prepare_minimax_masks.py        Prepare shadow-edge masks for MiniMax
|   +-- prepare_minimax_shadow_object_masks.py  Improved shadow+object masks (v2)
|   +-- run_objectclear_gtmask.py       ObjectClear exploratory run
|   +-- run_phase2_sdxl_all7.py         SDXL kf5 + ProPainter
|   +-- run_phase3_lama_all7.py         LaMa + ProPainter
|   +-- setup_diffueraser.sh            DiffuEraser environment setup
+-- eval/
|   +-- evaluate_all.py                 Unified PSNR/SSIM evaluation
|   +-- fair_psnr_eval.py               Fair inpaint-only evaluation
+-- reporting/
|   +-- build_part3_deliverables.py     Generate deliverable directory structure
|   +-- build_part3_result_table.py     Generate result tables
+-- configs/                            Per-sequence YAML configs
+-- gdino_vlm/                          GDINO/VLM sub-experiments
```

---

## Quick Usage

### Direction A -- SAM3 Multi-Object Mask

```bash
conda run -n controlnet_env python3 part3/direction_a/run_sam3_multiobject.py \
  --sequences tennis bmx-trees blackswan car-shadow horsejump-low
```

### Direction C -- Inpainting Comparison (DAVIS GT mask)

```bash
# ProPainter baseline
conda run -n propainter_env python3 part3/inpainting/run_propainter_gtmask.py \
  --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

# LaMa + ProPainter
conda run -n controlnet_env python3 part3/inpainting/run_phase3_lama_all7.py \
  --seqs tennis bmx-trees blackswan koala horsejump-low car-shadow

# DiffuEraser v9 (corrected hard-blend)
# Step 1: prepare inputs (dilate_px=0)
conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py \
  --seq tennis --version v8 --dilate_px 0
# Step 2: run inference (see script for per-sequence frame counts)
# Step 3: apply corrected hard-blend
conda run -n diffueraser_env python3 part3/inpainting/apply_hard_blend.py \
  --base_version v8 --out_version v9 --sequence tennis
```

### Direction D -- MiniMax-Remover

```bash
# GT mask run
conda run -p /path/to/minimax_env python3 part3/inpainting/run_minimax_remover_gtmask.py \
  --seq car-shadow --version v1

# Shadow-aware mask (shadow_object_v2)
# Step 1: generate improved shadow masks
conda run -p /path/to/minimax_env python3 part3/inpainting/prepare_minimax_shadow_object_masks.py \
  --seq car-shadow
# Step 2: run MiniMax with the shadow mask
conda run -p /path/to/minimax_env python3 part3/inpainting/run_minimax_remover_gtmask.py \
  --seq car-shadow --version shadow_object_v2 \
  --mask_dir /path/to/outputs/minimax_masks/shadow_object_v2/car-shadow
```

---

## DiffuEraser Version History (key milestones)

| Version | Description | PSNR proxy (tennis) | Status |
|---|---|---:|---|
| v1 | Default DiffuEraser, soft-blend leakage | 31.32 | Superseded |
| v4 | Incorrect hard-blend (threshold bug) | 36.28* | **Invalid** (object pasted back) |
| v8 | Improved inference params (ref_stride=5) | 31.31 | Intermediate |
| **v9** | v8 + corrected hard-blend (threshold >0) | 34.02 | **Final candidate** |
| v11 | car-shadow specific (dilate=5px, mdi=4) | 34.57 | car-shadow final |

> v4 is invalid: the high score resulted from pasting the original object back.
> v9 is the standard candidate for 4/5 DAVIS sequences; v11 for car-shadow.

---

## Evaluation

```bash
conda activate controlnet_env
python3 part3/eval/evaluate_all.py --seqs tennis bmx-trees
```

Full inpainting comparison results: `part3/evaluation_summary.csv`
