# Part3 Deliverables

 Part3 `metrics.json`  `experiment_card.md`

---

##

""

1. `part3_deliverables/<seq>/<exp_id>/`
2. `experiment_card.md`
3. `metrics.json`
4. `masked_in.mp4`  mask
5.  `inpaint_only`  `full_pipeline``inpaint_out.mp4`

""""

---

##

1. ** `masked_in.mp4`** mask
2. ** `inpaint_out.mp4`** hallucination mask
3. ****`PSNR / SSIM / JM / JR / F`

**** `needs_review`

---

##

|  |  |  |  Direction |
|---|---|---|---|
| `mask_only` |  mask | JM / JR / F | Direction A, B |
| `inpaint_only` |  mask | PSNR_proxy / PSNR_synthetic / SSIM | Direction C |
| `full_pipeline` | mask+ | JM + PSNR + SSIM | A+B+C  |

`inpaint_only`  mask DAVIS  DAVIS GT mask SAM3  mask

---

## schema v2

- `method_v1 / v2 / v3`
-
-  `experiment_card.md`  `Version History`
-  `run_manifest.json` builder

---

##

|  |  |
|---|---|
| `reference` |  |
| `stable` |  |
| `promising` |  |
| `exploratory` |  |
| `legacy` |  |
| `superseded` |  |
| `partial_or_failed` / `failed` |  |
| `needs_review` |  |

---

## DiffuEraser Direction C

|  |  |  |
|---|---|---|
|  | ✅  | conda env: `diffueraser_env` |
|  | ✅  | tennis: `input_video.mp4 + input_mask.mp4`70854x480|
|  | ⏳  | diffuEraser / SD1.5 / sd-vae-ft-mse / PCM_Weights |
| smoke test | ⏳  |  |
| v1  | ⏳  | tennis  inpaint_only |
|  | ⏳  |  pure_propainter_gtmask  |

---

##

```
part3_deliverables/
├── README.md                        #
├── experiment_registry.csv          #
├── experiment_registry.json
├── part3_results_full_table.csv     #  version/mask_protocol
├── part3_results_full_table.md
├── part3_results_full_table.json
├── part123_team_comparison.md       #
└── <sequence>/                      #
    ├── 00_readme.md                 #
    └── <exp_id>/                    #
        ├── experiment_card.md       #  + Version History
        ├── metrics.json             #  + schema v2
        ├── command.sh               #
        ├── run_manifest.json        # ()  manifest
        ├── masked_in.mp4            # mask
        ├── inpaint_out.mp4          #
        ├── mask_frames/             # mask
        └── <script>.py              #
```

---

* `../DELIVERABLES_GUIDE_CN.md`*
