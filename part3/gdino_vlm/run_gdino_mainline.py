"""
run_gdino_mainline.py
---------------------
Main line (Stage1 / Stage2):
  VLM prompt -> (GroundingDINO or fallback detector) boxes -> SAM2 propagation -> mask outputs

设计目标:
- Stage1: first-frame box
- Stage2: sparse re-anchor every K frames with simple IoU smoothing
- 若GDINO依赖不可用，自动fallback到YOLO并记录metadata，保证链路可运行
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
@dataclass
class RunMeta:
    stage: str
    sequence_name: str
    detector_requested: str
    detector_actual: str
    prompt: str
    num_frames: int
    num_anchor_frames: int
    num_boxes_total: int
    segmentor_backend: str
    prompt_source: str
    innovation_quality_gate: bool
    innovation_o2o_association: bool
    innovation_real_vlm: bool
    raw_caption: str
    normalized_prompt_tokens: List[str]
    prompt_quality_guard_triggered: bool
    prompt_fallback_reason: str


@dataclass
class PromptDecision:
    prompt: str
    source: str
    raw_caption: str
    normalized_tokens: List[str]
    quality_guard_triggered: bool
    fallback_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GDINO/VLM -> SAM2 mainline runner")
    parser.add_argument("--config", default="part3/configs/gdino_vlm_mainline.yaml")
    parser.add_argument("--sequence", required=True, help="sequence_name in config/policy")
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--output", required=True, help="mask输出目录")
    parser.add_argument(
        "--no-export-mp4",
        action="store_true",
        help="跳过导出 mask.mp4 / overlay.mp4（默认在跑完后写入 part3/gdino_vlm/outputs/<seq>_<stage>/）",
    )
    parser.add_argument(
        "--inpaint-video",
        default="",
        help="可选：ProPainter inpaint_out.mp4，用于生成 side_by_side.mp4",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置格式错误: {path}")
    return data


def list_frames(video_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in video_dir.iterdir() if p.suffix.lower() in exts])


def is_over_generic_prompt(prompt: str) -> bool:
    t = prompt.strip().lower()
    if not t:
        return True
    # 过泛词：会让开放词汇检测退化为“大类全检”，难以区分具体移除目标
    return t in {"object", "objects", "thing", "things", "person"}


def infer_prompt(cfg: Dict, seq_cfg: Dict, first_frame: Path) -> PromptDecision:
    vlm = cfg.get("vlm", {})
    real_vlm = vlm.get("real_prompt", {})
    use_real_vlm = bool(real_vlm.get("enable", False))
    force_override = bool(real_vlm.get("force_override_policy_prompt", False))
    user_intent = vlm.get("user_intent", "").strip()
    fixed = seq_cfg.get("prompt_text_for_gdino", "").strip()

    def policy_decision(reason: str = "") -> PromptDecision:
        return PromptDecision(
            prompt=fixed if fixed else "person",
            source="policy" if fixed else "rule_based",
            raw_caption="",
            normalized_tokens=[],
            quality_guard_triggered=False,
            fallback_reason=reason,
        )

    if fixed and not (use_real_vlm and force_override):
        return policy_decision()

    if use_real_vlm:
        prompt, raw_caption, normalized_tokens = infer_prompt_from_real_vlm(real_vlm, first_frame, user_intent)
        if prompt:
            if is_over_generic_prompt(prompt):
                if fixed:
                    return PromptDecision(
                        prompt=fixed,
                        source="policy_fallback_after_quality_guard",
                        raw_caption=raw_caption,
                        normalized_tokens=normalized_tokens,
                        quality_guard_triggered=True,
                        fallback_reason="real_vlm_prompt_over_generic",
                    )
                return PromptDecision(
                    prompt="person",
                    source="rule_based_fallback_after_quality_guard",
                    raw_caption=raw_caption,
                    normalized_tokens=normalized_tokens,
                    quality_guard_triggered=True,
                    fallback_reason="real_vlm_prompt_over_generic",
                )
            return PromptDecision(
                prompt=prompt,
                source="real_vlm",
                raw_caption=raw_caption,
                normalized_tokens=normalized_tokens,
                quality_guard_triggered=False,
                fallback_reason="",
            )
        if fixed:
            return PromptDecision(
                prompt=fixed,
                source="policy_fallback_after_real_vlm",
                raw_caption=raw_caption,
                normalized_tokens=normalized_tokens,
                quality_guard_triggered=False,
                fallback_reason="real_vlm_empty_output",
            )
    # MVP: rule-based prompt generation from user intent
    if not user_intent:
        return PromptDecision(
            prompt="person",
            source="rule_based",
            raw_caption="",
            normalized_tokens=[],
            quality_guard_triggered=False,
            fallback_reason="",
        )
    # 简易映射，防止VLM链路缺失时无prompt可用
    text = user_intent.lower()
    if "bicycle" in text or "bike" in text:
        return PromptDecision("person . bicycle", "rule_based", "", [], False, "")
    if "tennis" in text:
        return PromptDecision("person . tennis racket", "rule_based", "", [], False, "")
    if "car" in text:
        return PromptDecision("car", "rule_based", "", [], False, "")
    if "horse" in text:
        return PromptDecision("horse", "rule_based", "", [], False, "")
    return PromptDecision("person", "rule_based", "", [], False, "")


def infer_prompt_from_real_vlm(real_vlm_cfg: Dict, image_path: Path, user_intent: str) -> Tuple[str, str, List[str]]:
    model_name = str(real_vlm_cfg.get("model_name", "Salesforce/blip-image-captioning-base"))
    max_new_tokens = int(real_vlm_cfg.get("max_new_tokens", 24))
    try:
        from PIL import Image
        from transformers import pipeline  # type: ignore

        pipe = pipeline("image-to-text", model=model_name)
        out = pipe(Image.open(str(image_path)).convert("RGB"), max_new_tokens=max_new_tokens)
        caption = ""
        if out and isinstance(out, list):
            caption = str(out[0].get("generated_text", "")).lower().strip()
        text = f"{user_intent.lower()} {caption}".strip()
        tokens = []
        # 人物类
        if any(w in text for w in ("person", "man", "woman", "pedestrian", "athlete", "player", "rider")):
            tokens.append("person")
        # 运动器材
        if any(w in text for w in ("bicycle", "bike", "cycle")):
            tokens.append("bicycle")
        if any(w in text for w in ("tennis", "racket")):
            tokens.extend(["person", "tennis racket"])
        if any(w in text for w in ("tennis ball", "ball")) and "tennis" in text:
            tokens.append("tennis ball")
        # 车辆
        if any(w in text for w in ("car", "vehicle", "automobile")):
            tokens.append("car")
        # 动物
        if any(w in text for w in ("horse", "equestrian")):
            tokens.append("horse")
        if any(w in text for w in ("bird", "swan")):
            tokens.append("bird")
        if any(w in text for w in ("koala", "koala bear")):
            tokens.append("koala")
        if "bear" in text and "koala" not in text:
            tokens.append("bear")
        if "camel" in text:
            tokens.append("camel")
        # 随身物品（wild_video-1person 场景）
        if any(w in text for w in ("bag", "backpack", "handbag", "luggage", "purse", "suitcase")):
            tokens.append("bag")
        # 去重保序
        uniq: List[str] = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        print(f"[VLM] caption='{caption}' → tokens={uniq}")
        return " . ".join(uniq), caption, uniq
    except Exception as exc:
        print(f"[warn] real VLM prompt failed, fallback to rule/policy: {exc}")
        return "", "", []


def coco_classes_from_prompt(prompt: str) -> List[int]:
    p = prompt.lower()
    ids: List[int] = []
    mapping = [
        ("person", 0),
        ("bicycle", 1),
        ("car", 2),
        ("bird", 14),
        ("horse", 17),
        ("backpack", 24),
        ("handbag", 26),
        ("suitcase", 28),
        ("tennis racket", 38),
        ("bag", 24),
        # koala/bear/camel 不在 COCO，跳过
    ]
    for key, cid in mapping:
        if key in p:
            ids.append(cid)
    return sorted(set(ids)) or [0]


def try_load_gdino(cfg: Dict):
    gd = cfg.get("gdino", {})
    model_cfg = gd.get("model_config_path", "")
    ckpt = gd.get("checkpoint_path", "")
    try:
        from groundingdino.util.inference import load_model, load_image, predict  # type: ignore
    except Exception:
        return None
    if not model_cfg or not ckpt:
        return None
    model = load_model(model_cfg, ckpt)
    return {"model": model, "load_image": load_image, "predict": predict}


def detect_boxes_gdino(gdino_obj, image_path: Path, prompt: str, box_thr: float, text_thr: float) -> np.ndarray:
    load_image = gdino_obj["load_image"]
    predict = gdino_obj["predict"]
    model = gdino_obj["model"]
    image_source, image = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=box_thr,
        text_threshold=text_thr,
    )
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = image_source.shape[:2]
    # gdino boxes是cx,cy,w,h归一化
    boxes_xyxy = []
    for b in boxes:
        cx, cy, bw, bh = b.tolist()
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        boxes_xyxy.append([x1, y1, x2, y2])
    return np.array(boxes_xyxy, dtype=np.float32)


def detect_boxes_yolo(yolo_model, image_path: Path, classes: List[int], conf: float) -> np.ndarray:
    results = yolo_model(str(image_path), classes=classes, conf=conf)
    if len(results) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def smooth_boxes(
    prev_boxes: np.ndarray,
    new_boxes: np.ndarray,
    *,
    alpha: float = 0.6,
    min_iou_for_update: float = 0.0,
    one_to_one: bool = False,
) -> np.ndarray:
    if len(prev_boxes) == 0:
        return new_boxes
    if len(new_boxes) == 0:
        return prev_boxes
    smoothed = []
    used_new = set()
    for idx, pb in enumerate(prev_boxes):
        best_iou = -1.0
        best_nb = pb
        best_j = -1
        for j, nb in enumerate(new_boxes):
            if one_to_one and j in used_new:
                continue
            cur = iou_xyxy(pb, nb)
            if cur > best_iou:
                best_iou = cur
                best_nb = nb
                best_j = j
        if best_iou < float(min_iou_for_update):
            # 新框与历史框关联太弱时，不更新该对象，抑制错误重锚
            smoothed.append(pb)
            continue
        if one_to_one and best_j >= 0:
            used_new.add(best_j)
        smoothed.append(alpha * pb + (1.0 - alpha) * best_nb)
    return np.array(smoothed, dtype=np.float32)


def mean_best_iou(prev_boxes: np.ndarray, new_boxes: np.ndarray) -> float:
    if len(prev_boxes) == 0 or len(new_boxes) == 0:
        return 0.0
    vals: List[float] = []
    for pb in prev_boxes:
        vals.append(max(iou_xyxy(pb, nb) for nb in new_boxes))
    return float(np.mean(vals)) if vals else 0.0


def extract_boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for m in masks:
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def sam3_segment_frame(sam3_model, frame_path: Path, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取帧: {frame_path}")
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)
    bboxes = boxes.tolist()
    results = sam3_model.predict(source=str(frame_path), bboxes=bboxes, imgsz=1036, verbose=False)
    if not results:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取帧: {frame_path}")
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)
    masks_obj = results[0].masks
    if masks_obj is None or masks_obj.data is None:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取帧: {frame_path}")
        h, w = img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), np.zeros((0, 4), dtype=np.float32)
    masks = masks_obj.data.detach().cpu().numpy().astype(np.uint8)
    combined = (masks > 0).any(axis=0).astype(np.uint8) * 255
    new_boxes = extract_boxes_from_masks(masks > 0)
    return combined, new_boxes


def match_box_count(boxes: np.ndarray, target_n: int) -> np.ndarray:
    if target_n <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    if len(boxes) == 0:
        return np.zeros((target_n, 4), dtype=np.float32)
    if len(boxes) == target_n:
        return boxes.astype(np.float32)
    if len(boxes) > target_n:
        return boxes[:target_n].astype(np.float32)
    # 不足则重复最后一个框
    last = boxes[-1:]
    reps = [boxes.astype(np.float32)]
    while sum(len(x) for x in reps) < target_n:
        reps.append(last.astype(np.float32))
    out = np.concatenate(reps, axis=0)[:target_n]
    return out.astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    policy = load_yaml(Path(cfg["policy_path"]))
    seq_map = {s["sequence_name"]: s for s in policy["sequences"]}
    if args.sequence not in seq_map:
        raise ValueError(f"sequence 不在policy里: {args.sequence}")
    seq_cfg = seq_map[args.sequence]

    paths = cfg["paths"]
    # 先转绝对路径，避免后续 chdir(sam2_dir) 导致相对输出写到错误目录
    project_root = Path.cwd().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    video_dir = (Path(paths["davis_jpeg_root"]).expanduser() / args.sequence).resolve()
    if not video_dir.exists():
        raise FileNotFoundError(f"序列目录不存在: {video_dir}")
    frames = list_frames(video_dir)
    if not frames:
        raise RuntimeError(f"未找到帧: {video_dir}")

    segmentor = cfg.get("segmentor", {})
    backend = str(segmentor.get("backend", "sam2")).lower()
    if backend not in {"sam2", "sam3"}:
        raise ValueError(f"不支持的segmentor.backend: {backend}")
    sam2_dir = Path(cfg["sam2"]["sam2_dir"]).expanduser().resolve()
    yolo_weight = cfg["detector"]["yolo_weight"]
    yolo_conf = float(cfg["detector"].get("yolo_conf", 0.3))
    detector_requested = cfg["detector"].get("primary", "gdino")
    gdino_box_thr = float(cfg["gdino"].get("box_threshold", 0.35))
    gdino_text_thr = float(cfg["gdino"].get("text_threshold", 0.25))
    stage2_stride = int(
        seq_cfg.get(
            "gdino_reanchor_stride",
            policy.get("defaults", {}).get("gdino_reanchor_stride", 10),
        )
    )
    stage2_cfg = cfg.get("stage2", {})
    stage2_smooth_alpha = float(stage2_cfg.get("smooth_alpha", 0.6))
    stage2_min_iou_for_update = float(stage2_cfg.get("min_iou_for_update", 0.0))

    prompt_decision = infer_prompt(cfg, seq_cfg, frames[0])
    prompt = prompt_decision.prompt
    prompt_source = prompt_decision.source
    yolo_classes = coco_classes_from_prompt(prompt)
    innovation = cfg.get("innovation", {})
    quality_gate_cfg = innovation.get("quality_gated_reanchor", {})
    quality_gate_enable = bool(quality_gate_cfg.get("enable", False))
    quality_gate_iou = float(quality_gate_cfg.get("drift_iou_threshold", 0.55))
    o2o_assoc_enable = bool(innovation.get("o2o_temporal_association", {}).get("enable", False))

    # setup detector
    import os
    import sys

    from ultralytics import YOLO

    yolo_model = YOLO(yolo_weight)
    gdino_obj = try_load_gdino(cfg) if detector_requested == "gdino" else None
    detector_actual = "gdino" if gdino_obj is not None else "yolo_fallback"

    def detect_at(frame_idx: int) -> np.ndarray:
        frame_path = frames[frame_idx]
        if gdino_obj is not None:
            boxes = detect_boxes_gdino(gdino_obj, frame_path, prompt, gdino_box_thr, gdino_text_thr)
        else:
            boxes = detect_boxes_yolo(yolo_model, frame_path, classes=yolo_classes, conf=yolo_conf)
        return boxes

    # stage1: anchor only first frame
    # stage2: first frame + every stride
    anchor_indices = [0]
    if args.stage == "stage2":
        anchor_indices = sorted(set([0] + list(range(stage2_stride, len(frames), stage2_stride))))

    prev_boxes = None
    total_boxes = 0
    fixed_num_objs = None
    anchor_boxes: Dict[int, np.ndarray] = {}
    for t, fid in enumerate(anchor_indices):
        boxes_raw = detect_at(fid)
        boxes = boxes_raw
        if t == 0:
            if len(boxes) == 0:
                # 极端回退：给一个空框附近的小框，避免完全无锚点导致流程中断
                boxes = np.array([[0.0, 0.0, 8.0, 8.0]], dtype=np.float32)
            fixed_num_objs = len(boxes)
        assert fixed_num_objs is not None
        boxes = match_box_count(boxes, fixed_num_objs)
        if args.stage == "stage2" and prev_boxes is not None:
            if quality_gate_enable:
                drift = mean_best_iou(prev_boxes, boxes)
                if drift >= quality_gate_iou:
                    boxes = prev_boxes.copy()
                else:
                    boxes = smooth_boxes(
                        prev_boxes,
                        boxes,
                        alpha=stage2_smooth_alpha,
                        min_iou_for_update=stage2_min_iou_for_update,
                        one_to_one=o2o_assoc_enable,
                    )
            else:
                boxes = smooth_boxes(
                    prev_boxes,
                    boxes,
                    alpha=stage2_smooth_alpha,
                    min_iou_for_update=stage2_min_iou_for_update,
                    one_to_one=o2o_assoc_enable,
                )
        if len(boxes) > 0:
            prev_boxes = boxes.copy()
        anchor_boxes[int(fid)] = boxes.copy()
        total_boxes += int(len(boxes))

    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "sam2":
        # setup sam2 video predictor
        os.chdir(str(sam2_dir))
        sys.path.insert(0, str(sam2_dir))
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            cfg["sam2"]["config"],
            cfg["sam2"]["checkpoint"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        inference_state = predictor.init_state(video_path=str(video_dir))
        for fid in anchor_indices:
            boxes = anchor_boxes[int(fid)]
            for i, b in enumerate(boxes):
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=int(fid),
                    obj_id=i + 1,  # 固定obj_id，后续锚点更新同一对象
                    box=b.astype(np.float32),
                )
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            combined = (mask_logits > 0).any(dim=0).squeeze().cpu().numpy()
            mask = (combined * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), mask)
    else:
        # sam3 backend: frame-wise visual prompting with anchor refresh
        from ultralytics import SAM

        sam3_cfg = cfg.get("sam3", {})
        sam3_ckpt = str(sam3_cfg.get("checkpoint", "")).strip()
        if not sam3_ckpt:
            raise ValueError("segmentor.backend=sam3 时必须提供 sam3.checkpoint")
        sam3_model = SAM(sam3_ckpt)
        cur_boxes = anchor_boxes[int(anchor_indices[0])].copy()
        anchor_ptr = 1
        fixed_num = int(cur_boxes.shape[0]) if len(cur_boxes) > 0 else 1
        for frame_idx, frame_path in enumerate(frames):
            if anchor_ptr < len(anchor_indices) and frame_idx == int(anchor_indices[anchor_ptr]):
                cur_boxes = anchor_boxes[int(anchor_indices[anchor_ptr])].copy()
                anchor_ptr += 1
            mask, new_boxes = sam3_segment_frame(sam3_model, frame_path, cur_boxes)
            cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), mask)
            if len(new_boxes) > 0:
                cur_boxes = match_box_count(new_boxes, fixed_num)
    png_count = len(list(out_dir.glob("*.png")))
    if png_count != len(frames):
        raise RuntimeError(
            f"mask落盘数量与帧数不一致: png={png_count}, frames={len(frames)}, out_dir={out_dir}"
        )

    meta = RunMeta(
        stage=args.stage,
        sequence_name=args.sequence,
        detector_requested=detector_requested,
        detector_actual=detector_actual,
        prompt=prompt,
        num_frames=len(frames),
        num_anchor_frames=len(anchor_indices),
        num_boxes_total=total_boxes,
        segmentor_backend=backend,
        prompt_source=prompt_source,
        innovation_quality_gate=quality_gate_enable,
        innovation_o2o_association=o2o_assoc_enable,
        innovation_real_vlm=bool(cfg.get("vlm", {}).get("real_prompt", {}).get("enable", False)),
        raw_caption=prompt_decision.raw_caption,
        normalized_prompt_tokens=prompt_decision.normalized_tokens,
        prompt_quality_guard_triggered=prompt_decision.quality_guard_triggered,
        prompt_fallback_reason=prompt_decision.fallback_reason,
    )
    meta_path = out_dir / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, ensure_ascii=False, indent=2)
    print(f"[ok] masks saved to: {out_dir}")
    print(f"[ok] meta saved to:  {meta_path}")
    print(f"[ok] png_count / frame_count: {png_count} / {len(frames)}")

    if not args.no_export_mp4:
        from export_gdino_vlm_mp4 import export_mask_overlay_bundle

        bundle_out = project_root / "part3" / "gdino_vlm" / "outputs" / backend / f"{args.sequence}_{args.stage}"
        inpaint_video = Path(args.inpaint_video).expanduser().resolve() if args.inpaint_video else None
        manifest = export_mask_overlay_bundle(
            masks_dir=out_dir,
            frames_dir=video_dir,
            output_dir=bundle_out,
            inpaint_video=inpaint_video,
            fps=24.0,
        )
        print("[ok] bundle exported:")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
