# EvalDAVIS GT



1. `davis_eval_targets.yaml`GTGDINOre-anchor
2. `eval_davis_masks.py` `IoU/J, JR, F` CSV
3. `validate_davis_policy.py` policy

##

-  `part1_env`
  - `opencv-python`
  - `Pillow`
  - `PyYAML`
  - `numpy`



```bash
pip install opencv-python Pillow pyyaml numpy
```

##

 policy YAML

```bash
python eval/validate_davis_policy.py --policy eval/davis_eval_targets.yaml --check_paths
```



```bash
python eval/eval_davis_masks.py \
  --policy eval/davis_eval_targets.yaml \
  --output_csv eval/results_davis_masks.csv
```

 JSON

```bash
python eval/eval_davis_masks.py \
  --policy eval/davis_eval_targets.yaml \
  --output_csv eval/results_davis_masks.csv \
  --output_json eval/results_davis_masks.json
```

/

```bash
python eval/eval_davis_masks.py \
  --policy eval/davis_eval_targets.yaml \
  --output_csv eval/results_davis_masks.csv \
  --strict
```

 `blackswan/car-shadow/horsejump-low`

```bash
conda activate sam2_env
bash eval/run_missing_davis_masks.sh
```

## policy

- `defaults.gt_root`DAVIS `Annotations/480p`
- `defaults.pred_root`mask
- `defaults.iou_threshold_for_jr`JR
- `defaults.boundary_tolerance_px`F
- `defaults.gdino_reanchor_stride`Stage2K
- `sequences[].eval_mode`
  - `union_all_instances`GT
  - `instance_ids`ID
- `sequences[].pred_subdir`mask
- `sequences[].prompt_text_for_gdino`Main line
- `sequences[].gdino_reanchor_stride`Stage2K

##

- `IoU/J_mean`mask IoU
- `JR_at_tau`IoU >= tau
- `F_mean`F

DAVIS GT
