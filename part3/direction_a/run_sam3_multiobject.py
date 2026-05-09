"""
run_sam3_multiobject.py

Direction A: Official SAM3 video text prompt — 多目标联合提示版本

核心改进（相比 run_official_sam3_video.py）：
  1. 读取 prompt_scope.yaml，按序列自动选择 video_root 和 prompt 候选集
  2. 支持多 prompt 联合：对每个候选 prompt 分别调用 add_prompt，
     合并所有对象的 mask（union），覆盖多目标
  3. 支持 --search_mode：对每个候选 prompt 评分，输出最优 prompt 和 JM
  4. 支持 wild_video-1person 等非 DAVIS 序列
  5. 输出目录结构与 eval_davis_masks.py 兼容

使用方式:
  # 全量运行（按 prompt_scope.yaml 最优 prompt）
  python part3/run_sam3_multiobject.py \
    --scope_yaml part3/configs/prompt_scope.yaml \
    --sequences tennis koala wild_video-1person bmx-trees blackswan car-shadow horsejump-low \
    --output_root /data3/jli657/project3/part3/outputs/sam3_multiobj/masks \
    --checkpoint /data3/jli657/project3/weights/sam3/sam3.pt

  # Prompt 搜索模式（输出每个候选 prompt 的 JM，基于 GT 评估）
  python part3/run_sam3_multiobject.py \
    --scope_yaml part3/configs/prompt_scope.yaml \
    --sequences tennis koala \
    --output_root /data3/jli657/project3/part3/outputs/sam3_multiobj/search \
    --checkpoint /data3/jli657/project3/weights/sam3/sam3.pt \
    --search_mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from PIL import Image

# Official SAM3 repo path
SAM3_REPO = "/data3/jli657/sam3"

# Triton cache to avoid home disk full
os.environ.setdefault("TRITON_CACHE_DIR", "/data3/jli657/tmp/triton_cache")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direction A: SAM3 multi-object video text prompt")
    p.add_argument("--scope_yaml", default="part3/configs/prompt_scope.yaml",
                   help="prompt_scope.yaml path")
    p.add_argument("--sequences", nargs="+", default=None,
                   help="Sequences to process (default: all in scope_yaml)")
    p.add_argument("--output_root",
                   default="/data3/jli657/project3/part3/outputs/sam3_multiobj/masks",
                   help="Output root for mask PNGs")
    p.add_argument("--checkpoint",
                   default="/data3/jli657/project3/weights/sam3/sam3.pt",
                   help="SAM3 checkpoint path")
    p.add_argument("--frame_idx", type=int, default=0,
                   help="Anchor frame index for text prompt")
    p.add_argument("--gpu", type=int, default=0, help="GPU index")
    p.add_argument("--score_threshold", type=float, default=0.3,
                   help="Confidence threshold for mask extraction")
    p.add_argument("--search_mode", action="store_true",
                   help="Run all candidate prompts per sequence, output best")
    p.add_argument("--search_output_root",
                   default="/data3/jli657/project3/part3/outputs/sam3_multiobj/search",
                   help="Search mode output root")
    p.add_argument("--gt_root",
                   default="/home/jli657/shared_data/project3/DAVIS/Annotations/480p",
                   help="DAVIS GT annotation root (for search mode eval)")
    return p.parse_args()


def load_scope(yaml_path: str) -> List[dict]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("sequences", [])


def load_frames(video_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem,
    )


def extract_mask_from_outputs(
    outputs: dict, frame_h: int, frame_w: int, score_threshold: float = 0.3
) -> np.ndarray:
    """合并所有置信度超阈值目标的 binary mask。"""
    combined = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if not outputs:
        return combined

    if "out_binary_masks" in outputs:
        binary_masks = outputs["out_binary_masks"]
        probs = outputs.get("out_probs", None)
        if binary_masks is None or binary_masks.size == 0:
            return combined
        for i in range(binary_masks.shape[0]):
            if probs is not None and float(probs[i]) < score_threshold:
                continue
            mask = binary_masks[i]
            if mask.shape != (frame_h, frame_w):
                mask = cv2.resize(
                    mask.astype(np.float32), (frame_w, frame_h),
                    interpolation=cv2.INTER_LINEAR,
                ) > 0.5
            combined = np.maximum(combined, mask.astype(np.uint8) * 255)
        return combined

    # Fallback legacy format
    import torch
    items = outputs.items() if isinstance(outputs, dict) else enumerate(outputs)
    for _, obj_data in items:
        if obj_data is None:
            continue
        masks = obj_data.get("masks", None) if isinstance(obj_data, dict) else obj_data
        if masks is None:
            continue
        if isinstance(masks, torch.Tensor):
            mask_np = masks.squeeze().cpu().numpy()
        elif isinstance(masks, np.ndarray):
            mask_np = masks.squeeze()
        else:
            continue
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.shape != (frame_h, frame_w):
            mask_np = cv2.resize(mask_np.astype(np.float32), (frame_w, frame_h),
                                 interpolation=cv2.INTER_LINEAR)
        combined = np.maximum(combined, (mask_np > 0.5).astype(np.uint8) * 255)
    return combined


def run_sequence_with_prompts(
    predictor,
    seq_name: str,
    video_dir: Path,
    output_dir: Path,
    prompt_texts: List[str],
    frame_idx: int = 0,
    score_threshold: float = 0.3,
) -> dict:
    """
    对单个序列运行 SAM3 视频 text prompt（支持多个 prompt 联合）。
    每个 prompt 独立 add_prompt，propagate 后合并所有帧 mask（union）。
    """
    frames = load_frames(video_dir)
    if not frames:
        raise FileNotFoundError(f"No frames in {video_dir}")

    first_img = cv2.imread(str(frames[0]))
    frame_h, frame_w = first_img.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{seq_name}] frames={len(frames)}, prompts={prompt_texts}, anchor={frame_idx}")

    # 对每个 prompt 分别运行，累积 per-frame mask
    # 最终对所有 prompt 结果取 union
    combined_per_frame: Dict[int, np.ndarray] = {}

    for prompt_text in prompt_texts:
        print(f"  [{seq_name}] adding prompt: '{prompt_text}'")
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(video_dir))
        )
        session_id = response["session_id"]

        predictor.handle_request(
            request=dict(type="reset_session", session_id=session_id)
        )

        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                text=prompt_text,
            )
        )

        outputs_per_frame: Dict[int, dict] = {}
        for prop_response in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            fidx = prop_response["frame_index"]
            outputs_per_frame[fidx] = prop_response["outputs"]

        print(f"  [{seq_name}] '{prompt_text}': propagated {len(outputs_per_frame)} frames")

        for fidx, out in outputs_per_frame.items():
            mask = extract_mask_from_outputs(out, frame_h, frame_w, score_threshold)
            if fidx not in combined_per_frame:
                combined_per_frame[fidx] = np.zeros((frame_h, frame_w), dtype=np.uint8)
            combined_per_frame[fidx] = np.maximum(combined_per_frame[fidx], mask)

        try:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        except Exception:
            pass

    # Save masks
    png_count = 0
    for i, frame_path in enumerate(frames):
        out_path = output_dir / f"{frame_path.stem}.png"
        mask = combined_per_frame.get(i, np.zeros((frame_h, frame_w), dtype=np.uint8))
        Image.fromarray(mask).save(str(out_path))
        png_count += 1

    return {
        "sequence_name": seq_name,
        "prompt_texts": prompt_texts,
        "num_frames": len(frames),
        "png_count": png_count,
        "masks_dir": str(output_dir),
        "prompt_source": "official_sam3_text_multiobj",
    }


def compute_jm_for_dir(pred_dir: Path, gt_dir: Path) -> float:
    """快速计算 pred_dir vs gt_dir 的 mean IoU（JM）用于 prompt 搜索。

    注意：DAVIS GT mask 使用实例 ID（如 38、75）而非 255；需用 > 0 判断前景。
    预测 mask 使用 255，用 > 127 判断。
    """
    ious = []
    gt_files = sorted(gt_dir.glob("*.png"), key=lambda p: p.stem)
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = np.array(Image.open(gt_path).convert("L")) > 0   # DAVIS: instance ID > 0 = foreground
        pred = np.array(Image.open(pred_path).convert("L")) > 127  # pred: binary 255
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def main() -> None:
    args = parse_args()

    scope = load_scope(args.scope_yaml)
    scope_by_name = {s["sequence_name"]: s for s in scope}

    target_seqs = args.sequences or [s["sequence_name"] for s in scope]

    if SAM3_REPO not in sys.path:
        sys.path.insert(0, SAM3_REPO)

    print("Loading official SAM3 video predictor...")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=args.checkpoint,
        gpus_to_use=[args.gpu],
        strict_state_dict_loading=False,
    )
    print("SAM3 loaded.")

    output_root = Path(args.output_root)
    all_meta = []

    for seq in target_seqs:
        if seq not in scope_by_name:
            print(f"[warn] {seq} not in scope_yaml, skipping")
            continue

        seq_cfg = scope_by_name[seq]
        video_dir = Path(seq_cfg["video_root"])
        candidates: List[str] = seq_cfg.get("sam3_official_prompts", ["object"])

        if not video_dir.exists():
            print(f"[warn] video dir not found: {video_dir}, skipping")
            continue

        if args.search_mode:
            # ---- Search mode: try each candidate prompt individually ----
            search_root = Path(args.search_output_root)
            best_prompt = candidates[0]
            best_jm = -1.0
            search_results = []

            gt_dir = Path(args.gt_root) / seq
            has_gt = gt_dir.exists()

            for cand in candidates:
                safe = cand.replace(" ", "_").replace("/", "_")
                cand_dir = search_root / f"{seq}_{safe}"
                try:
                    meta = run_sequence_with_prompts(
                        predictor=predictor,
                        seq_name=seq,
                        video_dir=video_dir,
                        output_dir=cand_dir,
                        prompt_texts=[cand],
                        frame_idx=args.frame_idx,
                        score_threshold=args.score_threshold,
                    )
                    jm = compute_jm_for_dir(cand_dir, gt_dir) if has_gt else -1.0
                    meta["jm"] = jm
                    search_results.append({"prompt": cand, "jm": jm, "dir": str(cand_dir)})
                    print(f"  [{seq}] '{cand}' -> JM={jm:.4f}")
                    if jm > best_jm:
                        best_jm = jm
                        best_prompt = cand
                except Exception as exc:
                    import traceback
                    print(f"  [{seq}] '{cand}' -> ERROR: {exc}")
                    traceback.print_exc()
                    search_results.append({"prompt": cand, "jm": -1.0, "error": str(exc)})

            all_meta.append({
                "sequence_name": seq,
                "search_results": search_results,
                "best_prompt": best_prompt,
                "best_jm": best_jm,
            })
            print(f"[{seq}] BEST: '{best_prompt}' JM={best_jm:.4f}")

        else:
            # ---- Normal mode: use ALL candidate prompts (union = multi-object) ----
            output_dir = output_root / seq
            try:
                meta = run_sequence_with_prompts(
                    predictor=predictor,
                    seq_name=seq,
                    video_dir=video_dir,
                    output_dir=output_dir,
                    prompt_texts=candidates,
                    frame_idx=args.frame_idx,
                    score_threshold=args.score_threshold,
                )
                all_meta.append({"status": "ok", **meta})
                print(f"[ok] {seq}: {meta['png_count']} masks saved")
            except Exception as exc:
                import traceback
                print(f"[error] {seq}: {exc}")
                traceback.print_exc()
                all_meta.append({"status": "error", "sequence_name": seq, "error": str(exc)})

    manifest_path = output_root / "manifest_multiobj.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n[save] manifest: {manifest_path}")

    if args.search_mode:
        # Also save best prompts JSON
        best_out = {
            r["sequence_name"]: {
                "best_prompt": r.get("best_prompt"),
                "best_jm": r.get("best_jm"),
            }
            for r in all_meta
        }
        best_path = Path(args.search_output_root) / "search_best_multiobj.json"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_path, "w") as f:
            json.dump(best_out, f, indent=2)
        print(f"[save] best prompts: {best_path}")


if __name__ == "__main__":
    main()
