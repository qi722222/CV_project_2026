"""
extend_shadow_mask.py — 为 wild_video-1person 的 person mask 添加地面阴影扩展

策略：
  1. 加载每帧 SAM3 生成的 person mask
  2. 找到人物脚底（mask 最低非零行）及宽度
  3. 在脚底以下方向用垂直拉伸椭圆核做形态学膨胀，模拟地面投影阴影
  4. union(person_mask, shadow_region) 作为最终 mask
  5. 可视化：生成 debug 对比图（原始 mask vs 增强 mask）
  6. 输出到 --output_dir

用法:
  conda run -n propainter_env python3 part3/extend_shadow_mask.py \
    --input_dir  /data3/jli657/project3/part3/outputs/sam3_multiobj/masks_final/wild_video-1person \
    --output_dir /data3/jli657/project3/part3/outputs/sam3_multiobj/masks_shadow/wild_video-1person \
    --shadow_down 0.25 \
    --shadow_lateral 0.15 \
    --blur_sigma 4
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory with binary mask PNGs (0 or 255)")
    p.add_argument("--output_dir", required=True, help="Directory to write extended mask PNGs")
    p.add_argument("--shadow_down", type=float, default=0.28,
                   help="Shadow vertical extent as fraction of person height (default 0.28)")
    p.add_argument("--shadow_lateral", type=float, default=0.12,
                   help="Shadow horizontal expansion as fraction of person width (default 0.12)")
    p.add_argument("--shadow_direction", type=float, default=0.05,
                   help="Shadow horizontal offset as fraction of person width (positive = right, default 0.05)")
    p.add_argument("--blur_sigma", type=float, default=5.0,
                   help="Gaussian blur sigma for shadow softening")
    p.add_argument("--shadow_threshold", type=float, default=64,
                   help="Threshold to binarize blurred shadow (0-255)")
    p.add_argument("--debug_dir", default="",
                   help="If set, save comparison images here")
    return p.parse_args()


def extend_single_mask(mask: np.ndarray, shadow_down: float, shadow_lateral: float,
                        shadow_direction: float, blur_sigma: float, shadow_threshold: float) -> np.ndarray:
    """
    mask: H x W, values 0 or 255
    Returns extended mask of same shape.
    """
    H, W = mask.shape
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        return mask.copy()

    top_row = int(nz[0].min())
    bottom_row = int(nz[0].max())
    left_col = int(nz[1].min())
    right_col = int(nz[1].max())

    person_h = bottom_row - top_row + 1
    person_w = right_col - left_col + 1

    # Shadow ellipse parameters
    shadow_h = max(10, int(person_h * shadow_down))
    shadow_w_half = max(8, int(person_w * (0.5 + shadow_lateral)))
    shadow_center_x = int((left_col + right_col) / 2 + person_w * shadow_direction)
    shadow_center_y = min(H - 1, bottom_row + shadow_h // 2)

    # Draw filled ellipse on a blank canvas → shadow region
    shadow_canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(
        shadow_canvas,
        center=(shadow_center_x, shadow_center_y),
        axes=(shadow_w_half, shadow_h // 2),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    # Only keep the shadow BELOW the feet (avoid bleeding into person body)
    shadow_canvas[:bottom_row, :] = 0

    # Soften shadow edges with Gaussian blur
    if blur_sigma > 0:
        shadow_blurred = cv2.GaussianBlur(shadow_canvas.astype(np.float32), (0, 0), sigmaX=blur_sigma * 1.5, sigmaY=blur_sigma * 0.8)
        shadow_binary = (shadow_blurred > shadow_threshold).astype(np.uint8) * 255
    else:
        shadow_binary = shadow_canvas

    # Union: person mask + shadow
    extended = np.maximum(mask, shadow_binary)
    return extended


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(input_dir.glob("*.png"), key=lambda p: p.stem)
    if not mask_files:
        print(f"[error] No PNG files found in {input_dir}")
        return

    print(f"Processing {len(mask_files)} masks from {input_dir}")
    print(f"Shadow params: down={args.shadow_down}, lateral={args.shadow_lateral}, "
          f"direction={args.shadow_direction}, blur={args.blur_sigma}")

    shadow_added = 0
    for i, mpath in enumerate(mask_files):
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [warn] Cannot read {mpath.name}, skipping")
            continue

        # Ensure binary
        mask = (mask > 127).astype(np.uint8) * 255

        extended = extend_single_mask(
            mask,
            shadow_down=args.shadow_down,
            shadow_lateral=args.shadow_lateral,
            shadow_direction=args.shadow_direction,
            blur_sigma=args.blur_sigma,
            shadow_threshold=args.shadow_threshold,
        )

        if extended.sum() > mask.sum():
            shadow_added += 1

        out_path = output_dir / mpath.name
        Image.fromarray(extended).save(str(out_path))

        # Debug: save side-by-side comparison for every 30th frame
        if debug_dir and i % 30 == 0:
            comparison = np.hstack([mask, extended])
            cv2.imwrite(str(debug_dir / f"debug_{mpath.stem}.png"), comparison)

    print(f"Done. Extended {shadow_added}/{len(mask_files)} masks with shadow region.")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
