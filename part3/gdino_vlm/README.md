# GDINO / VLM Mainline Runner

 Part3 `VLM -> detector(GDINO/YOLO) -> segmentor(SAM2/SAM3) -> mask/mp4`

##

- SAM2  SAM3  `segmentor.backend`
-  SAM3 5  × Stage1 + Stage2 `mask png + run_meta + mask/overlay mp4`
- `part3/gdino_vlm/part123_sam2_sam3_compare.md`
- `part3/gdino_vlm/sam3_innovation_ablation.md`

##

- `run_gdino_mainline.py` `sam2/sam3`
- `run_sam3_davis5.sh`SAM3 5  Stage1+Stage2
- `run_sam3_innovation_ablation.sh` A/B/C
- `build_unified_compare_csv.py` Part1/Part2/Part3(SAM2+SAM3)
- `build_gdino_ablation_v2.py` `gdino_ablation.csv`

##

 Part3  `sam3_env`

```bash
conda create -y -n sam3_env --clone /home/jli657/.conda/envs/gdino_env
PATH="/data2/jli657/envs/sam3_env/bin:$PATH" pip install --upgrade \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```


```bash
HF_ENDPOINT="https://hf-mirror.com"
```

## SAM2 / SAM3

- SAM2 `part3/configs/gdino_vlm_mainline.yaml``segmentor.backend: sam2`
- SAM3 Stage1`part3/configs/gdino_vlm_mainline_sam3.yaml`
- SAM3 Stage2`part3/configs/gdino_vlm_mainline_sam3_stage2.yaml`


```bash
HF_ENDPOINT="https://hf-mirror.com" \
PATH="/data2/jli657/envs/sam3_env/bin:$PATH" \
python part3/gdino_vlm/run_gdino_mainline.py \
  --config part3/configs/gdino_vlm_mainline_sam3.yaml \
  --sequence tennis \
  --stage stage1 \
  --output part3/gdino_vlm/masks/sam3/stage1/tennis
```

##

-  SAM2
- SAM3
-  O2O DAVIS2  baseline

- `part3/gdino_vlm/part123_sam2_sam3_compare.md`
- `part3/gdino_vlm/sam3_decision.md`
- `part3/gdino_vlm/sam3_innovation_ablation.md`

## VLM

-  VLM `vlm.real_prompt.enable`
-  prompt`run_meta.json`  `prompt_source`
-  RealVLM
