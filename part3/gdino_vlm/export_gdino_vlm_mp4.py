"""
从逐帧 mask PNG + DAVIS 帧目录（或可选 inpaint 视频）导出报告用 mp4。
不依赖 ffmpeg，使用 OpenCV VideoWriter（与 part1 make_full_compare 同构）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _list_masks(mask_dir: Path) -> List[Path]:
    return sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png" and p.stem.isdigit())


def _list_frames(frame_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in frame_dir.iterdir() if p.suffix.lower() in exts)


def _read_video_frames(path: Path, max_n: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out: List[np.ndarray] = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        out.append(fr)
        if max_n is not None and len(out) >= max_n:
            break
    cap.release()
    return out


def export_mask_overlay_bundle(
    masks_dir: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    inpaint_video: Optional[Path] = None,
    fps: float = 24.0,
    overlay_alpha: float = 0.45,
) -> dict:
    """
    写入 output_dir: mask.mp4, overlay.mp4, manifest.json；
    若提供 inpaint_video 且可读，则额外写入 side_by_side.mp4（原 | overlay | inpaint）。
    返回 manifest 字典。
    """
    masks_dir = masks_dir.resolve()
    frames_dir = frames_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = _list_masks(masks_dir)
    frames = _list_frames(frames_dir)
    if not masks:
        raise FileNotFoundError(f"未找到逐帧 mask PNG: {masks_dir}")
    if not frames:
        raise FileNotFoundError(f"未找到帧图像: {frames_dir}")

    n = min(len(masks), len(frames))
    if n != len(masks) or n != len(frames):
        # 允许截断到公共长度，但在 manifest 中记录
        pass

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"无法读取首帧: {frames[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_mask = cv2.VideoWriter(str(output_dir / "mask.mp4"), fourcc, fps, (w, h), True)
    vw_overlay = cv2.VideoWriter(str(output_dir / "overlay.mp4"), fourcc, fps, (w, h), True)

    inpaint_frames: Optional[List[np.ndarray]] = None
    if inpaint_video is not None and inpaint_video.is_file():
        inpaint_frames = _read_video_frames(inpaint_video.resolve(), max_n=n)
        if len(inpaint_frames) >= n:
            inpaint_frames = inpaint_frames[:n]
        else:
            inpaint_frames = None  # 太短则不做三列

    vw_side: Optional[cv2.VideoWriter] = None
    if inpaint_frames is not None and len(inpaint_frames) == n:
        side_w = w * 3
        vw_side = cv2.VideoWriter(str(output_dir / "side_by_side.mp4"), fourcc, fps, (side_w, h), True)

    for i in range(n):
        fr = cv2.imread(str(frames[i]))
        if fr is None:
            raise RuntimeError(f"无法读取帧: {frames[i]}")
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h))
        m = cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"无法读取 mask: {masks[i]}")
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m > 127).astype(np.float32)

        mask_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        vw_mask.write(mask_bgr)

        overlay = fr.astype(np.float32)
        red = np.zeros_like(overlay)
        red[:, :, 2] = 255.0
        a = overlay_alpha * m_bin[..., None]
        out = overlay * (1.0 - a) + red * a
        vw_overlay.write(np.clip(out, 0, 255).astype(np.uint8))

        if vw_side is not None and inpaint_frames is not None:
            inp = inpaint_frames[i]
            if inp.shape[0] != h or inp.shape[1] != w:
                inp = cv2.resize(inp, (w, h))
            side = np.hstack([fr, np.clip(out, 0, 255).astype(np.uint8), inp])
            vw_side.write(side)

    vw_mask.release()
    vw_overlay.release()
    if vw_side is not None:
        vw_side.release()

    manifest = {
        "num_frames_written": int(n),
        "fps": fps,
        "masks_dir": str(masks_dir),
        "frames_dir": str(frames_dir),
        "inpaint_video": str(inpaint_video) if inpaint_video else None,
        "side_by_side_semantics": "col1=original,col2=mask_overlay,col3=inpaint_video_if_provided",
        "outputs": {
            "mask_mp4": str(output_dir / "mask.mp4"),
            "overlay_mp4": str(output_dir / "overlay.mp4"),
            "side_by_side_mp4": str(output_dir / "side_by_side.mp4")
            if (output_dir / "side_by_side.mp4").is_file()
            else None,
        },
        "mask_png_count": len(masks),
        "frame_file_count": len(frames),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出 mask / overlay / 可选三列 mp4")
    p.add_argument("--masks_dir", required=True, type=Path)
    p.add_argument("--frames_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path, help="如 part3/gdino_vlm/outputs/tennis_stage1")
    p.add_argument("--inpaint_video", type=Path, default=None)
    p.add_argument("--fps", type=float, default=24.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    m = export_mask_overlay_bundle(
        args.masks_dir,
        args.frames_dir,
        args.output_dir,
        inpaint_video=args.inpaint_video,
        fps=args.fps,
    )
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
