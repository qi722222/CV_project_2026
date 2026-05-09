# Part3 Rebuild (SAM3 Mainline)

本流程是从头重建的 Part3 主线：

`GDINO + VLM prompt -> SAM3 propagation (stage2 sparse re-anchor) -> ProPainter`

不包含 ControlNet（本轮只做 Main line）。

## 配置文件

- DAVIS5: `part3/configs/sam3_rebuild_mainline_davis5.yaml`
- Wild: `part3/configs/sam3_rebuild_mainline_wild.yaml`
- 对应 policy:
  - `part3/configs/sam3_rebuild_policy_davis5.yaml`
  - `part3/configs/sam3_rebuild_policy_wild.yaml`

## 一键运行

```bash
cd /home/jli657/my_storage2_1T/project3

/data2/jli657/envs/gdino_env/bin/python part3/run_part3_sam3_rebuild.py \
  --mode all \
  --stage stage2 \
  --gdino_python /data2/jli657/envs/gdino_env/bin/python \
  --propainter_python /data2/jli657/envs/propainter_env/bin/python \
  --propainter_dir /home/jli657/my_storage2_1T/ProPainter
```

## 输出

- masks: `part3/outputs/sam3_rebuild/masks/<group>/<sequence>/`
- inpaint: `part3/outputs/sam3_rebuild/propainter/<group>/<sequence>/inpaint_out.mp4`
- manifest: `part3/outputs/sam3_rebuild/rebuild_manifest.json`
