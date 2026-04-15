# Part 1 — 经典 CV Baseline

YOLOv8-Seg + Lucas-Kanade 光流 + 时序背景传播 + cv2.inpaint。

这是经典 CV 方法 baseline。项目总览见 [`../README.md`](../README.md)，SOTA 对照见 [`../part2/README.md`](../part2/README.md)。

---

## Pipeline

```
帧序列
    │
    ▼
YOLOv8x-Seg ─► 各类实例 mask（person、bicycle 等）
    │
    ▼
Lucas-Kanade 光流 ─► 过滤静态物体
    │
    ▼
cv2.dilate（kernel 9~15）─► 覆盖运动模糊边缘
    │
    ▼
时序背景传播（前后 ±N 帧窗口）
    │  ├─► 在邻近帧的同一坐标借用干净像素
    │  └─► fallback：cv2.inpaint（Telea）处理永远不可见的像素
    ▼
输出 MP4
```

---

## 环境配置

```bash
conda create -n part1_env python=3.10 -y
conda activate part1_env

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics --no-deps      # --no-deps 防止覆盖 torch
pip install pandas seaborn              # ultralytics 运行时依赖
pip install opencv-python scikit-image imageio imageio-ffmpeg numpy scipy
```

验证：
```bash
python -c "from ultralytics import YOLO; import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

---

## 权重

YOLOv8x-Seg（约 140 MB）首次运行会自动下载。如果想用本地副本：

```bash
# 手动下载（自动下载慢的话）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt
mv yolov8x-seg.pt /path/to/your/weights/
```

我们用最大的 `yolov8x-seg.pt` 以获得最好的 mask 质量。小模型（`yolov8n-seg.pt` 等）速度更快但 mask 噪声更多。

---

## 使用方法

### 单个数据集

```bash
conda activate part1_env

python scripts/run_part1.py \
    --video_dir   /path/to/帧序列文件夹 \
    --dataset     bmx-trees \
    --output      outputs/bmx-trees.mp4
```

### 关键参数

| 参数 | 默认值 | 什么时候调 |
|---|---|---|
| `--dilate_kernel` | 9 | 出现边缘虚影时调到 13 或 15 |
| `--motion_threshold` | 1.5 | 相机有明显运动时调到 2.5 或 3.0（如 bmx-trees）|
| `--conf` | 0.3 | YOLO 漏检某些帧时调到 0.2（如 tennis 球员特殊姿势）|
| `--window` | 15 | 加大 dilation 后需要填补的像素更多，调到 25 |

### 各数据集推荐参数

基于参数扫描结果（见 `run_sweep_v2.sh`）：

```bash
# bmx-trees: 相机剧烈运动 → 放宽运动阈值 + 加大膨胀
python scripts/run_part1.py \
    --video_dir /path/to/bmx-trees \
    --dataset bmx-trees \
    --output outputs/bmx-trees.mp4 \
    --dilate_kernel 15 --motion_threshold 2.5 --window 25

# tennis: YOLO 漏检若干帧 → 降低置信度 + 加大膨胀
python scripts/run_part1.py \
    --video_dir /path/to/tennis \
    --dataset tennis \
    --output outputs/tennis.mp4 \
    --dilate_kernel 15 --conf 0.2 --window 20
```

### 指定要删除的类别

编辑 `prompts.json`：
```json
{
  "bmx-trees": ["person", "bicycle"],
  "tennis":    ["person"],
  "wild":      ["person"]
}
```

---

## 脚本说明

| 脚本 | 用途 |
|---|---|
| `scripts/gen_masks_yolo.py` | YOLO 推理 + LK 运动过滤 + 膨胀 → 输出 mask PNG |
| `scripts/inpaint_temporal.py` | 时序传播 + cv2.inpaint fallback |
| `scripts/run_part1.py` | 主入口，串联以上模块 |
| `scripts/compare.py` | Part 1 与 Part 2 mask 的 IoU 一致性分析 |
| `scripts/make_compare.py` | OpenCV 实现的三路并排视频（无需 ffmpeg）|
| `scripts/make_full_compare.py` | 原帧 \| Part 1 \| Part 2 三路并排 |
| `run_sweep_v2.sh` | 并行跑多组参数组合 |

---

## 已知局限

| 现象 | 原因 | 解决/规避 |
|---|---|---|
| `bmx-trees` 出现虚影 | 相机跟拍主体，"同坐标借像素"假设被破坏 | 这是时序传播方法的**根本局限**。相机运动场景请用 Part 2 |
| 某些帧物体重新出现 | YOLO 在困难姿势下漏检 | 降低 `--conf`，或加 mask 时域平滑 |
| 长视频 OOM | 所有帧一次性加载 | 分块处理，或用 `np.memmap` |
| CPU 上很慢 | 没检测到 CUDA | 确认 `CUDA_VISIBLE_DEVICES` 设对了；脚本会**静默 fallback 到 CPU** |

---

## 性能注意事项

⚠️ **重要**：`run_part1.py` 在指定的 CUDA 设备不存在时**会静默 fallback 到 CPU**（比如 `CUDA_VISIBLE_DEVICES` 设错）。CPU 上每帧约 12 秒（GPU 是 0.05 秒）。**一定要看 log 里 `it/s` 的数字是不是正常**来判断是否真的用上了 GPU。

```bash
# 总是显式设置
export CUDA_VISIBLE_DEVICES=1   # 用空闲的那张 GPU
```