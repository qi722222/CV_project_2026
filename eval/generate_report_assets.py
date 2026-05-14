"""
generate_report_assets.py
-------------------------

1)  report_assets/
2) Original / Part1 / Part2 / Part3
3) limitations
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report assets")
    parser.add_argument("--project_root", default="/home/jli657/my_storage2_1T/project3")
    parser.add_argument("--output_dir", default="/home/jli657/my_storage2_1T/project3/report_assets")
    return parser.parse_args()


def read_frame_from_video(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def read_frame_from_dir(frames_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    if not frames_dir.exists():
        return None
    files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not files:
        return None
    frame_idx = min(max(0, frame_idx), len(files) - 1)
    return cv2.imread(str(files[frame_idx]))


def make_panel(img: Optional[np.ndarray], title: str, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    if img is None:
        canvas = np.full((h, w, 3), 25, dtype=np.uint8)
        cv2.putText(canvas, "N/A", (w // 2 - 25, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    else:
        canvas = cv2.resize(img, (w, h))
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_qual_figure(
    out_path: Path,
    orig: Optional[np.ndarray],
    p1: Optional[np.ndarray],
    p2: Optional[np.ndarray],
    p3: Optional[np.ndarray],
) -> None:
    size = (420, 236)
    panels = [
        make_panel(orig, "Original", size),
        make_panel(p1, "Part1", size),
        make_panel(p2, "Part2", size),
        make_panel(p3, "Part3", size),
    ]
    stacked = np.hstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), stacked)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    out_root = Path(args.output_dir)
    figs_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1)
    src_table = root / "eval" / "davis_results_table.md"
    if src_table.exists():
        shutil.copy2(src_table, out_root / "table_main_davis.md")

    # 2)
    dataset_cfg = {
        "bmx-trees": {
            "frame_idx": 20,
            "orig_dir": Path("/home/jli657/shared_data/project3/bmx-trees"),
            "part1_video": root / "part1" / "outputs" / "bmx-trees.mp4",
            "part2_video": root / "part2" / "outputs" / "bmx-trees" / "bmx-trees" / "inpaint_out.mp4",
            "part3_dir": None,
        },
        "tennis": {
            "frame_idx": 35,
            "orig_dir": Path("/home/jli657/shared_data/project3/tennis"),
            "part1_video": root / "part1" / "outputs" / "tennis.mp4",
            "part2_video": root / "part2" / "outputs" / "tennis_v3" / "tennis" / "inpaint_out.mp4",
            "part3_dir": root / "part3" / "outputs" / "tennis" / "refined_keyframes",
        },
        "wild_video-1person": {
            "frame_idx": 15,
            "orig_dir": root / "wild_frames" / "wild_video-1person",
            "part1_video": root / "part1" / "outputs" / "wild_video-1person.mp4",
            "part2_video": root / "part2" / "outputs" / "wild_video-1person" / "wild_video-1person" / "inpaint_out.mp4",
            "part3_dir": None,
        },
    }

    for name, cfg in dataset_cfg.items():
        fid = int(cfg["frame_idx"])
        orig = read_frame_from_dir(cfg["orig_dir"], fid)
        p1 = read_frame_from_video(cfg["part1_video"], fid)
        p2 = read_frame_from_video(cfg["part2_video"], fid)

        p3 = None
        p3_dir = cfg.get("part3_dir")
        if p3_dir is not None:
            p3 = read_frame_from_dir(Path(p3_dir), fid)
        if p3 is None:
            # fallback: part3part2
            p3 = p2.copy() if p2 is not None else None

        save_qual_figure(figs_dir / f"qual_{name}.png", orig, p1, p2, p3)

    # 3) limitations
    limitations = out_root / "limitations_draft.md"
    limitations.write_text(
        "\n".join(
            [
                "# Limitations Draft",
                "",
                "- DAVIS  mask-GT  J/F/IoU  clean GT  proxy",
                "- GDINO  yolo_fallback",
                "- Stage2  tennis ",
                "- Part3  keyframe skeletoncopy_fallback  SD+ControlNet ",
            ]
        ),
        encoding="utf-8",
    )

    # 4)
    manifest = out_root / "assets_manifest.md"
    manifest.write_text(
        "\n".join(
            [
                "# Report Assets Manifest",
                "",
                "- Main table: `report_assets/table_main_davis.md`",
                "- Qual figures:",
                "  - `report_assets/figures/qual_bmx-trees.png`",
                "  - `report_assets/figures/qual_tennis.png`",
                "  - `report_assets/figures/qual_wild_video-1person.png`",
                "- Limitations draft: `report_assets/limitations_draft.md`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[ok] report assets generated at: {out_root}")


if __name__ == "__main__":
    main()
