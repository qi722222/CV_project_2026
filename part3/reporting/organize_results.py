"""
organize_results.py — Phase 0: 建立统一 results/ 目录并 symlink 已有输出

统一结构:
  /data3/jli657/project3/part3/results/<seq>/
    mask_frames/         -> masks_final/<seq>/
    direction_a/
      sam3_propainter/
        inpaint_out.mp4
        masked_in.mp4
    direction_c/
      pure_propainter/   (controlnet 消融里的 propainter_pure 结果)
      sdxl_kf5_propainter/
      lama_propainter/   (待跑)
    part2_baseline/
      inpaint_out.mp4
      masked_in.mp4
"""

import os
import sys
from pathlib import Path

RESULTS = Path("/data3/jli657/project3/part3/results")
OUTPUTS = Path("/data3/jli657/project3/part3/outputs")
PART2   = Path("/data3/jli657/project3/part2/outputs")

def symlink(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"  [SKIP] src missing: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return  # already correct
        dst.unlink()
    dst.symlink_to(src)
    print(f"  [link] {dst.relative_to(RESULTS)} -> {src}")

def symlink_dir(src, dst):
    """Symlink a whole directory."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"  [SKIP] dir src missing: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)
    print(f"  [dir-link] {dst.relative_to(RESULTS)} -> {src}")

SEQUENCES = {
    "tennis": {
        "dir_a_src": OUTPUTS / "sam3_multiobj/propainter/tennis/tennis",
        "pure_pp_src": OUTPUTS / "controlnet/ablation_5seq/tennis/propainter_pure/tennis",
        "sdxl_kf5_src": None,  # not done for tennis yet
        "part2_src": PART2 / "tennis_v3/tennis",
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/tennis",
    },
    "koala": {
        "dir_a_src": OUTPUTS / "sam3_multiobj/propainter/koala/koala",
        "pure_pp_src": OUTPUTS / "controlnet/ablation_5seq/koala/propainter_pure/koala",
        "sdxl_kf5_src": OUTPUTS / "keyframe_inpaint_ablation/sdxl/koala/interval5/propainter_output/frames",
        "part2_src": None,  # Part2 did not process koala
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/koala",
    },
    "wild_video-1person": {
        "dir_a_src": OUTPUTS / "sam3_multiobj/propainter_shadow_v2/wild_video-1person",
        "pure_pp_src": None,  # no controlnet ablation for wild
        "sdxl_kf5_src": None,
        "part2_src": PART2 / "wild_video-1person/wild_video-1person",
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/wild_video-1person",
    },
    "bmx-trees": {
        "dir_a_src": None,  # to be created in Phase 1
        "pure_pp_src": OUTPUTS / "controlnet/ablation_5seq/bmx-trees/propainter_pure/bmx-trees",
        "sdxl_kf5_src": None,  # to be fixed in Phase 2
        "part2_src": PART2 / "bmx-trees_v2/bmx-trees",
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/bmx-trees",
    },
    "blackswan": {
        "dir_a_src": None,  # to be created in Phase 1
        "pure_pp_src": None,
        "sdxl_kf5_src": None,
        "part2_src": None,  # Part2 did not process blackswan inpainting
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/blackswan",
    },
    "horsejump-low": {
        "dir_a_src": None,  # to be created in Phase 1
        "pure_pp_src": None,
        "sdxl_kf5_src": None,
        "part2_src": None,
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/horsejump-low",
    },
    "car-shadow": {
        "dir_a_src": None,  # to be created in Phase 1
        "pure_pp_src": None,
        "sdxl_kf5_src": None,
        "part2_src": None,
        "mask_src": OUTPUTS / "sam3_multiobj/masks_final/car-shadow",
    },
}


def setup_sequence(seq_name, cfg):
    print(f"\n=== {seq_name} ===")
    base = RESULTS / seq_name

    # mask_frames
    if cfg["mask_src"]:
        symlink_dir(cfg["mask_src"], base / "mask_frames")

    # Direction A
    if cfg["dir_a_src"]:
        da = base / "direction_a/sam3_propainter"
        for f in ["inpaint_out.mp4", "masked_in.mp4"]:
            symlink(cfg["dir_a_src"] / f, da / f)

    # Direction C: pure_propainter
    if cfg["pure_pp_src"]:
        dcp = base / "direction_c/pure_propainter"
        for f in ["inpaint_out.mp4", "masked_in.mp4"]:
            symlink(cfg["pure_pp_src"] / f, dcp / f)

    # Direction C: sdxl_kf5_propainter
    if cfg["sdxl_kf5_src"]:
        dcs = base / "direction_c/sdxl_kf5_propainter"
        for f in ["inpaint_out.mp4", "masked_in.mp4"]:
            symlink(cfg["sdxl_kf5_src"] / f, dcs / f)

    # Part2 baseline
    if cfg["part2_src"]:
        p2 = base / "part2_baseline"
        for f in ["inpaint_out.mp4", "masked_in.mp4"]:
            symlink(cfg["part2_src"] / f, p2 / f)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for seq, cfg in SEQUENCES.items():
        setup_sequence(seq, cfg)

    print("\n\n=== Directory structure ===")
    for seq in sorted(SEQUENCES.keys()):
        base = RESULTS / seq
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.mp4")):
            exists = "OK" if (p.is_symlink() and p.resolve().exists()) else "BROKEN"
            print(f"  [{exists}] {p.relative_to(RESULTS)}")


if __name__ == "__main__":
    main()
