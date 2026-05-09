from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np


ROOT = Path("/data3/jli657/project3/part3")
DELIVERABLES = ROOT / "part3_deliverables"
EVAL_DIR = Path("/home/jli657/my_storage2_1T/project3/eval")
PART2_ROOT = Path("/data3/jli657/project3/part2")
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
WILD_FRAMES = Path("/data3/jli657/project3/wild_frames")


SEQUENCES: Dict[str, Dict[str, str]] = {
    "tennis": {
        "frame_dir": str(DAVIS_FRAMES / "tennis"),
        "difficulty": "人物和球拍都在快速运动，容易漏球拍或把背景误吞进去。",
    },
    "bmx-trees": {
        "frame_dir": str(DAVIS_FRAMES / "bmx-trees"),
        "difficulty": "树枝和骑手/自行车交错，目标细长且运动快，是目前最难的弱点序列。",
    },
    "blackswan": {
        "frame_dir": str(DAVIS_FRAMES / "blackswan"),
        "difficulty": "黑天鹅与水面反光容易混在一起，但目标整体比较稳定。",
    },
    "car-shadow": {
        "frame_dir": str(DAVIS_FRAMES / "car-shadow"),
        "difficulty": "不仅要去掉车，还要处理地面阴影；阴影是否算目标会直接影响分数。",
    },
    "horsejump-low": {
        "frame_dir": str(DAVIS_FRAMES / "horsejump-low"),
        "difficulty": "马和人高速运动，形变较大，容易出现跟踪漂移。",
    },
    "koala": {
        "frame_dir": str(DAVIS_FRAMES / "koala"),
        "difficulty": "目标遮挡较大，局部贴着树枝，修复难度也比较高。",
    },
    "wild_video-1person": {
        "frame_dir": str(WILD_FRAMES / "wild_video-1person"),
        "difficulty": "真实视频场景，人物、书包、阴影都会互相干扰，没有 DAVIS 那么干净。",
    },
    "bear": {
        "frame_dir": str(DAVIS_FRAMES / "bear"),
        "difficulty": "动物轮廓较大，但毛边和自然背景纹理容易混淆。",
    },
    "camel": {
        "frame_dir": str(DAVIS_FRAMES / "camel"),
        "difficulty": "主体较大，整体不算最难，但边界和背景颜色接近时仍可能抖动。",
    },
}


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json_load(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metric_book() -> Dict[str, Dict[str, Dict[str, Any]]]:
    book: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def put(method_key: str, seq: str, data: Dict[str, Any]) -> None:
        book.setdefault(method_key, {})[seq] = data

    # Part2 baseline masks
    for row in _csv_rows(EVAL_DIR / "results_davis_masks.csv"):
        put(
            "part2_mask_baseline",
            row["sequence_name"],
            {
                "JM": float(row["J_mean"]),
                "JR": float(row["JR_at_tau"]),
                "F": float(row["F_mean"]),
                "source_file": str(EVAL_DIR / "results_davis_masks.csv"),
            },
        )

    # Official SAM3
    for method_key, filename in [
        ("official_sam3_video_mask_only", "results_official_sam3.csv"),
        ("official_sam3_best_mask_only", "results_official_sam3_best.csv"),
        ("sam3_multiobj_mask_only", "results_sam3_multiobj_final.csv"),
        ("sam3_rebuild_v1_mask_only", "results_part3_sam3_rebuild_v1.csv"),
        ("vlm_on_stage1_mask_only", "results_vlm_on.csv"),
        ("vlm_off_stage1_mask_only", "results_vlm_off_stage1.csv"),
    ]:
        for row in _csv_rows(EVAL_DIR / filename):
            put(
                method_key,
                row["sequence_name"],
                {
                    "JM": float(row["J_mean"]),
                    "JR": float(row["JR_at_tau"]),
                    "F": float(row["F_mean"]),
                    "source_file": str(EVAL_DIR / filename),
                },
            )

    # GDINO SAM2 / SAM3 ablation
    for row in _csv_rows(ROOT / "gdino_vlm" / "gdino_ablation.csv"):
        backend = row["segmentor_backend"]
        stage = row["stage"]
        seq = row["sequence_name"]
        if row["method_tag"] == "part3_gdino" and backend in {"sam2", "sam3"}:
            method_key = f"gdino_{backend}_{stage}_mask_only"
            put(
                method_key,
                seq,
                {
                    "JM": float(row["iou_mean"]),
                    "F": float(row["f_mean"]),
                    "source_file": str(ROOT / "gdino_vlm" / "gdino_ablation.csv"),
                },
            )

    # GDINO innovation variants
    innov_files = {
        "gdino_sam3_o2o_mask_only": ROOT / "gdino_vlm" / "eval_sam3_innov_o2o.csv",
        "gdino_sam3_quality_gate_mask_only": ROOT / "gdino_vlm" / "eval_sam3_innov_quality_gate.csv",
        "gdino_sam3_real_vlm_mask_only": ROOT / "gdino_vlm" / "eval_sam3_innov_real_vlm.csv",
    }
    for method_key, path in innov_files.items():
        for row in _csv_rows(path):
            put(
                method_key,
                row["sequence_name"],
                {
                    "JM": float(row["J_mean"]),
                    "JR": float(row["JR_at_tau"]),
                    "F": float(row["F_mean"]),
                    "source_file": str(path),
                },
            )

    # Direction A fusion
    for row in _csv_rows(EVAL_DIR / "direction_a_fusion_results.csv"):
        put(
            "direction_a_mask_fusion_mask_only",
            row["sequence"],
            {
                "JM": float(row["fused_JM"]),
                "JR": float(row["fused_JR"]),
                "F": float(row["fused_F"]),
                "routing_strategy": row["strategy"],
                "source_file": str(EVAL_DIR / "direction_a_fusion_results.csv"),
            },
        )

    # Direction A shadow sweep
    shadow_rows = _csv_rows(EVAL_DIR / "direction_a_shadow_ablation.csv")
    if shadow_rows:
        best_row = max(shadow_rows, key=lambda r: float(r["JM"]))
        put(
            "direction_a_shadow_geom_scale_sweep_mask_only",
            "car-shadow",
            {
                "best_scale": float(best_row["shadow_scale"]),
                "best_JM": float(best_row["JM"]),
                "all_scales_source": str(EVAL_DIR / "direction_a_shadow_ablation.csv"),
                "source_file": str(EVAL_DIR / "direction_a_shadow_ablation.csv"),
            },
        )

    # Direction B
    for row in _csv_rows(EVAL_DIR / "direction_b_vggt4d_results.csv"):
        put(
            "vggt4d_vggt_mask_only",
            row["sequence"],
            {
                "JM": float(row["JM"]),
                "JR": float(row["JR"]),
                "F": float(row["F"]),
                "status": row["status"],
                "source_file": str(EVAL_DIR / "direction_b_vggt4d_results.csv"),
            },
        )
    for row in _csv_rows(EVAL_DIR / "direction_b_pi3_results.csv"):
        put(
            "pi3_transplant_v3_mask_only",
            row["sequence"],
            {
                "JM": float(row["JM"]),
                "JR": float(row["JR"]),
                "F": float(row["F"]),
                "status": row["status"],
                "source_file": str(EVAL_DIR / "direction_b_pi3_results.csv"),
            },
        )
    for method_key, filename in [
        ("vggt4d_sam3_refine_v2_mask_only", "direction_b_sam3_refined_v2_results.csv"),
        ("vggt4d_sam3_refine_v3_mask_only", "direction_b_sam3_refined_v3_results.csv"),
        ("vggt4d_sam3_refine_v4_mask_only", "direction_b_sam3_refined_v4_results.csv"),
        ("vggt4d_sam3_refine_v5_mask_only", "direction_b_sam3_refined_v5_results.csv"),
    ]:
        for row in _csv_rows(EVAL_DIR / filename):
            seq = row["sequence"]
            data: Dict[str, Any] = {
                "status": row.get("status", ""),
                "source": row.get("source", ""),
                "prompt_mode": row.get("prompt_mode", ""),
                "prompt_frame": row.get("prompt_frame", ""),
                "source_file": str(EVAL_DIR / filename),
            }
            for key in ("JM", "JR", "F", "rough_JM"):
                val = row.get(key, "")
                if val not in {"", None}:
                    try:
                        data[key] = float(val)
                    except ValueError:
                        data[key] = val
            put(method_key, seq, data)

    # Inpaint / full-pipeline summary
    eval_summary = {}
    for row in _csv_rows(ROOT / "results" / "evaluation_summary.csv"):
        eval_summary.setdefault(row["method"], {})[row["sequence"]] = {
            "PSNR_proxy": float(row["PSNR_proxy"]) if row["PSNR_proxy"] != "nan" else None,
            "PSNR_synthetic": float(row["PSNR_synthetic"]) if row["PSNR_synthetic"] != "nan" else None,
            "SSIM": float(row["SSIM"]),
            "n_frames": int(row["n_frames"]),
            "source_file": str(ROOT / "results" / "evaluation_summary.csv"),
        }
    method_map = {
        "part2_baseline_full_pipeline": "part2_baseline",
        "sam3_multiobj_propainter_full_pipeline": "part3_dir_a",
        "pure_propainter_fixed_mask": "part3_pure_pp",
        "sdxl_kf5_propainter_fixed_mask": "part3_sdxl_kf5",
        "lama_propainter_fixed_mask": "part3_lama",
    }
    for method_key, raw_method in method_map.items():
        for seq, data in eval_summary.get(raw_method, {}).items():
            put(method_key, seq, data)

    # GT-mask inpaint evaluation (from evaluation_summary_gtmask.csv)
    eval_summary_gtmask = {}
    for row in _csv_rows(ROOT / "results" / "evaluation_summary_gtmask.csv"):
        eval_summary_gtmask.setdefault(row["method"], {})[row["sequence"]] = {
            "PSNR_proxy": float(row["PSNR_proxy"]) if row.get("PSNR_proxy", "nan") != "nan" else None,
            "PSNR_synthetic": float(row["PSNR_synthetic"]) if row.get("PSNR_synthetic", "nan") != "nan" else None,
            "SSIM": float(row["SSIM"]) if row.get("SSIM", "nan") != "nan" else None,
            "n_frames": int(row["n_frames"]) if row.get("n_frames") else 0,
            "source_file": str(ROOT / "results" / "evaluation_summary_gtmask.csv"),
        }
    gtmask_method_map = {
        "pure_propainter_gtmask": "part3_pure_pp_gtmask",
        "sdxl_kf5_gtmask_propainter": "part3_sdxl_kf5_gtmask",
        "lama_gtmask_propainter": "part3_lama_gtmask",
    }
    for method_key, raw_method in gtmask_method_map.items():
        for seq, data in eval_summary_gtmask.get(raw_method, {}).items():
            put(method_key, seq, data)

    # Unified eval v2 for A+B best
    for row in _csv_rows(EVAL_DIR / "unified_eval_v2.csv"):
        method_name = row["method"]
        seq = row["sequence"]
        if method_name == "A+B Best Fusion+PP":
            put(
                "a_plus_b_best_full_pipeline",
                seq,
                {
                    "mask_JM": float(row["mask_JM"]),
                    "PSNR_proxy": float(row["psnr_proxy"]) if row["psnr_proxy"] != "nan" else None,
                    "PSNR_synthetic": float(row["psnr_synth"]) if row["psnr_synth"] != "nan" else None,
                    "SSIM": float(row["ssim"]) if row["ssim"] != "nan" else None,
                    "source_file": str(EVAL_DIR / "unified_eval_v2.csv"),
                },
            )

    # ControlNet ablation
    summary_5seq = _json_load(ROOT / "outputs" / "controlnet" / "ablation_5seq" / "summary_5seq.json") or []
    controlnet_method_map = {
        "pure_propainter": "controlnet_pure_propainter_fixed_mask",
        "hybrid": "controlnet_hybrid_propainter_fixed_mask",
        "hybrid_temporal_consistency": "controlnet_hybrid_tc_propainter_fixed_mask",
    }
    for item in summary_5seq:
        method_key = controlnet_method_map.get(item["method"])
        if not method_key:
            continue
        put(
            method_key,
            item["sequence"],
            {
                "JM": float(item["JM"]),
                "JR": float(item["JR"]),
                "PSNR_proxy": float(item["PSNR_proxy"]),
                "PSNR_synthetic": float(item["PSNR_synth"]),
                "SSIM_proxy": float(item["SSIM_proxy"]),
                "SSIM_synthetic": float(item["SSIM_synth"]),
                "recommended_method": item["recommended_method"],
                "source_file": str(ROOT / "outputs" / "controlnet" / "ablation_5seq" / "summary_5seq.json"),
            },
        )

    # Legacy SDXL interval10 two-seq metrics
    for seq, filename in [
        ("tennis", "metrics_part3_sdxl_tennis.json"),
        ("bmx-trees", "metrics_part3_sdxl_bmx.json"),
    ]:
        data = _json_load(EVAL_DIR / filename)
        if data:
            put(
                "sdxl_interval10_legacy_fixed_mask",
                seq,
                {
                    "JM": float(data["mask_quality"]["JM"]),
                    "JR": float(data["mask_quality"]["JR"]),
                    "PSNR_proxy": float(data["video_quality_proxy"]["PSNR"]),
                    "PSNR_synthetic": float(data["video_quality_synthetic"]["PSNR"]),
                    "SSIM_proxy": float(data["video_quality_proxy"]["SSIM"]),
                    "SSIM_synthetic": float(data["video_quality_synthetic"]["SSIM"]),
                    "source_file": str(EVAL_DIR / filename),
                },
            )
    for seq, filename in [
        ("tennis", "metrics_part3_propainter_tennis.json"),
        ("bmx-trees", "metrics_part3_propainter_bmx.json"),
    ]:
        data = _json_load(EVAL_DIR / filename)
        if data:
            put(
                "pure_propainter_legacy_fixed_mask",
                seq,
                {
                    "JM": float(data["mask_quality"]["JM"]),
                    "JR": float(data["mask_quality"]["JR"]),
                    "PSNR_proxy": float(data["video_quality_proxy"]["PSNR"]),
                    "PSNR_synthetic": float(data["video_quality_synthetic"]["PSNR"]),
                    "SSIM_proxy": float(data["video_quality_proxy"]["SSIM"]),
                    "SSIM_synthetic": float(data["video_quality_synthetic"]["SSIM"]),
                    "source_file": str(EVAL_DIR / filename),
                },
            )

    # Koala diffusion / VOID
    koala_summary = _json_load(ROOT / "outputs" / "koala_diffusion" / "koala_experiment_summary.json")
    if koala_summary:
        for item in koala_summary["results"]:
            key = None
            if item["method"] == "VOID pass1 (Netflix CogVideoX-Fun-5B)":
                key = "koala_void_full_pipeline"
            if key:
                put(
                    key,
                    "koala",
                    {
                        "PSNR_proxy": float(item["PSNR_proxy_dB"]),
                        "video": item["video"],
                        "masked_in": item["masked_in"],
                        "note": item.get("note", ""),
                        "source_file": str(ROOT / "outputs" / "koala_diffusion" / "koala_experiment_summary.json"),
                    },
                )

    return book


METRIC_BOOK = load_metric_book()


def metric_for(exp_id: str, seq: str) -> Dict[str, Any]:
    return METRIC_BOOK.get(exp_id, {}).get(seq, {})


def part2_mask_dir(seq: str) -> Optional[Path]:
    mapping = {
        "tennis": PART2_ROOT / "masks_cache" / "tennis_v3",
        "bmx-trees": PART2_ROOT / "masks_cache" / "bmx-trees_bbox_union",
        "blackswan": PART2_ROOT / "masks_cache" / "blackswan",
        "car-shadow": PART2_ROOT / "masks_cache" / "car-shadow",
        "horsejump-low": PART2_ROOT / "masks_cache" / "horsejump-low",
        "wild_video-1person": PART2_ROOT / "masks_cache" / "wild_video-1person",
    }
    return mapping.get(seq)


def part2_video_dir(seq: str) -> Optional[Path]:
    mapping = {
        "tennis": PART2_ROOT / "outputs" / "tennis_v3" / "tennis",
        "bmx-trees": PART2_ROOT / "outputs" / "bmx-trees_v2" / "bmx-trees",
        "wild_video-1person": PART2_ROOT / "outputs" / "wild_video-1person" / "wild_video-1person",
    }
    return mapping.get(seq)


def sam3_rebuild_mask_dir(seq: str) -> Path:
    if seq == "wild_video-1person":
        return ROOT / "outputs" / "sam3_rebuild_v1" / "masks" / "wild" / seq
    return ROOT / "outputs" / "sam3_rebuild_v1" / "masks" / "davis5" / seq


def sam3_rebuild_propainter_dir(seq: str) -> Path:
    if seq == "wild_video-1person":
        return ROOT / "outputs" / "sam3_rebuild_v1" / "propainter" / "wild" / seq
    return ROOT / "outputs" / "sam3_rebuild_v1" / "propainter" / "davis5" / seq


@dataclass
class ExperimentDef:
    exp_id: str
    readable_name: str
    family: str
    comparison_type: str
    audit_status: str
    sequences: List[str]
    script_path: Optional[Path]
    config_path: Optional[Path]
    command_template: str
    plain_explanation: str
    what_to_check: str
    current_takeaway: str
    path_builder: Callable[[str], Dict[str, Path]]
    # ── 新增 schema v2 字段（带默认值，向后兼容旧 ExperimentDef 实例）───────
    version: str = field(default="legacy")
    """版本标签，例如 v1 / v2 / v3 / legacy；持续优化实验必须显式递增。"""
    mask_protocol: str = field(default="")
    """mask 来源协议：davis_gt | sam3_mask | wild_existing_mask | controlnet_gt | 空字符串=未指定。"""
    baseline: str = field(default="")
    """本实验对比的基线 exp_id，例如 pure_propainter_gtmask；首版写最强可比基线。"""
    stage_gate: str = field(default="")
    """进入下一阶段的门槛描述；为空表示当前是终态或不需要门控。"""
    next_decision: str = field(default="")
    """当前结果产出后下一步该做什么（扩序列 / 调参 / 停止 / 进 full_pipeline）。"""
    failure_reason: str = field(default="")
    """仅当 audit_status 为 failed / partial_or_failed / superseded 时填写失败原因。"""


def build_defs() -> List[ExperimentDef]:
    davis5 = ["tennis", "bmx-trees", "blackswan", "car-shadow", "horsejump-low"]
    davis6 = davis5 + ["koala"]
    all7 = davis6 + ["wild_video-1person"]
    big_set = davis6 + ["bear", "camel"]
    return [
        ExperimentDef(
            "part2_mask_baseline",
            "Part2 YOLO+SAM2 基线 mask",
            "baseline",
            "mask_only",
            "reference",
            davis6 + ["wild_video-1person"],
            None,
            None,
            "",
            "这是 Part2 的历史基线，只看遮罩本身，不看后续修复。",
            "重点看它为什么在 tennis / bmx-trees / car-shadow 上曾经比一些 Part3 版本更稳。",
            "它是后面所有 Part3 路线的参照系，不是这次要主推的创新点。",
            lambda seq: {"mask_frames": part2_mask_dir(seq) or Path("")},
        ),
        ExperimentDef(
            "part2_baseline_full_pipeline",
            "Part2 YOLO+SAM2+ProPainter 基线完整流程",
            "baseline",
            "full_pipeline",
            "reference",
            ["tennis", "bmx-trees", "wild_video-1person"],
            None,
            None,
            "",
            "这是旧基线：YOLO+SAM2 先出 mask，再用 ProPainter 修复。",
            "先看 masked_in.mp4 判断旧 mask 是否稳，再看 inpaint_out.mp4 作为全流程参照。",
            "它不是最先进，但它很重要，因为很多新方法必须先超过它才有说服力。",
            lambda seq: {
                "mask_frames": part2_mask_dir(seq) or Path(""),
                "source_output_dir": part2_video_dir(seq) or Path(""),
                "inpaint_out": (part2_video_dir(seq) or Path("")) / "inpaint_out.mp4",
                "masked_in": (part2_video_dir(seq) or Path("")) / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "official_sam3_video_mask_only",
            "官方 SAM3 单 prompt 视频分割",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "direction_a/run_official_sam3_video.py",
            None,
            "python part3/direction_a/run_official_sam3_video.py --sequences {sequence} --output_root part3/outputs/official_sam3_video/masks",
            "这是最直接的官方 SAM3 文本 prompt 版本，一条 prompt 直接做视频分割。",
            "看它是不是因为 prompt 太单一，导致漏目标或框得不完整。",
            "它证明了官方 SAM3 能跑通，但单 prompt 版本不是最终最强方案。",
            lambda seq: {"mask_frames": ROOT / "outputs" / "official_sam3_video" / "masks" / seq},
        ),
        ExperimentDef(
            "official_sam3_best_mask_only",
            "官方 SAM3 best-prompt 版本",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "direction_a/run_official_sam3_video.py",
            None,
            "python part3/direction_a/run_official_sam3_video.py --sequences {sequence} --output_root part3/outputs/official_sam3_best/masks",
            "这是官方 SAM3 的调过 prompt 版本，用来回答“只换 prompt 能不能变好”。",
            "看它相对单 prompt 是否只在少数序列改善，还是整体都提升。",
            "它说明 prompt 确实重要，但仅靠 prompt 调整还不够。",
            lambda seq: {"mask_frames": ROOT / "outputs" / "official_sam3_best" / "masks" / seq},
        ),
        ExperimentDef(
            "sam3_multiobj_mask_only",
            "SAM3 multi-object 联合 prompt mask",
            "direction_a",
            "mask_only",
            "stable",
            all7,
            ROOT / "direction_a/run_sam3_multiobject.py",
            ROOT / "configs" / "prompt_scope.yaml",
            "python part3/direction_a/run_sam3_multiobject.py --scope_yaml part3/configs/prompt_scope.yaml --sequences {sequence}",
            "这是 Part3 Direction A 的主力版本：多个 prompt 一起提示，再把所有对象的 mask 取并集。",
            "看它是否解决了单 prompt 漏球拍、漏附属目标、漏阴影的问题。",
            "这是目前 Direction A 最有代表性的 SAM3 路线之一。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "sam3_multiobj" / "masks_final" / seq,
                "source_output_dir": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq,
                "inpaint_out": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sam3_multiobj_propainter_full_pipeline",
            "SAM3 multi-object + ProPainter 完整流程",
            "direction_a",
            "full_pipeline",
            "stable",
            all7,
            ROOT / "direction_a/run_sam3_multiobject.py",
            ROOT / "configs" / "prompt_scope.yaml",
            "python part3/direction_a/run_sam3_multiobject.py --scope_yaml part3/configs/prompt_scope.yaml --sequences {sequence}",
            "这是把 SAM3 multi-object mask 接到 ProPainter 后的完整视频结果。",
            "先看 masked_in.mp4 判断 mask，再看 inpaint_out.mp4 判断 ProPainter 是否能把洞补自然。",
            "它代表了 Direction A 在最终视频上的主要交付版本。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "sam3_multiobj" / "masks_final" / seq,
                "source_output_dir": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq,
                "inpaint_out": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sam3_rebuild_v1_mask_only",
            "SAM3 rebuild v1 mask",
            "direction_a",
            "mask_only",
            "exploratory",
            big_set + ["wild_video-1person"],
            ROOT / "direction_a/run_part3_sam3_rebuild.py",
            ROOT / "configs" / "sam3_rebuild_mainline_davis5.yaml",
            "python part3/direction_a/run_part3_sam3_rebuild.py --config part3/configs/sam3_rebuild_mainline_davis5.yaml",
            "这是后来重建的一条 SAM3 主线，和 multi-object 不是完全同一个产物来源。",
            "看它与 multi-object 比，是否在某些序列更稳，还是只是另一条中间路线。",
            "它是重要的重建实验，但不是所有序列最终都用它。",
            lambda seq: {
                "mask_frames": sam3_rebuild_mask_dir(seq),
                "source_output_dir": sam3_rebuild_propainter_dir(seq),
                "inpaint_out": sam3_rebuild_propainter_dir(seq) / "inpaint_out.mp4",
                "masked_in": sam3_rebuild_propainter_dir(seq) / "masked_in.mp4",
                "manifest": ROOT / "outputs" / "sam3_rebuild_v1" / "rebuild_manifest.json",
            },
        ),
        ExperimentDef(
            "sam3_rebuild_v1_propainter_full_pipeline",
            "SAM3 rebuild v1 + ProPainter",
            "direction_a",
            "full_pipeline",
            "exploratory",
            big_set + ["wild_video-1person"],
            ROOT / "direction_a/run_part3_sam3_rebuild.py",
            ROOT / "configs" / "sam3_rebuild_mainline_davis5.yaml",
            "python part3/direction_a/run_part3_sam3_rebuild.py --config part3/configs/sam3_rebuild_mainline_davis5.yaml",
            "这是 rebuild v1 接 ProPainter 的完整流程，用来和 multi-object 路线分开看。",
            "如果 masked_in.mp4 比 multi-object 更好，但最终视频并没有更好，问题可能出在修复而不是 mask。",
            "它更像一条对照和重建路线，不是目前最统一的最终版本。",
            lambda seq: {
                "mask_frames": sam3_rebuild_mask_dir(seq),
                "source_output_dir": sam3_rebuild_propainter_dir(seq),
                "inpaint_out": sam3_rebuild_propainter_dir(seq) / "inpaint_out.mp4",
                "masked_in": sam3_rebuild_propainter_dir(seq) / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "gdino_sam2_stage1_mask_only",
            "GDINO + SAM2 Stage1 mask",
            "direction_a",
            "mask_only",
            "stable",
            davis5,
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_mainline.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_mainline.yaml --sequence {sequence} --stage stage1 --output part3/gdino_vlm/masks/stage1/{sequence}",
            "GDINO 先根据文字找框，再交给 SAM2 做 mask 和传播。这是比较稳的历史主线。",
            "看第一帧检测框准不准，以及后续传播有没有漂移。",
            "在当前数据里，它依然是很强的参照，尤其 bmx-trees 上比 SAM3 更稳。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "stage1" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "stage1" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam2_stage2_mask_only",
            "GDINO + SAM2 Stage2 稀疏重锚 mask",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_mainline.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_mainline.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/stage2/{sequence}",
            "这条路线尝试在视频中间重新找框，想解决长视频漂移问题。",
            "重点看重锚后是不是反而引入更多抖动或目标切换。",
            "当前整体不如 Stage1 稳，说明重锚不是越多越好。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "stage2" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "stage2" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_stage1_mask_only",
            "GDINO + SAM3 Stage1 mask",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_mainline_sam3.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_mainline_sam3.yaml --sequence {sequence} --stage stage1 --output part3/gdino_vlm/masks/sam3/stage1/{sequence}",
            "GDINO 负责找框，SAM3 负责分割和传播，用来回答“把 SAM2 换成 SAM3 是否整体变强”。",
            "重点看是框的问题还是 SAM3 传播的问题，尤其 bmx-trees 和 tennis。",
            "它在部分序列有亮点，但整体还没有超过 GDINO+SAM2 Stage1。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage1" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage1" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_stage2_mask_only",
            "GDINO + SAM3 Stage2 稀疏重锚 mask",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_mainline_sam3_stage2.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_mainline_sam3_stage2.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/sam3/stage2/{sequence}",
            "这是 GDINO+SAM3 再加稀疏重锚的版本，原本想提高长时稳定性。",
            "看重锚是否真修复了漂移，还是把错误框继续传播了。",
            "当前它比 GDINO+SAM3 Stage1 还低，说明现有重锚设计不够稳。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage2" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage2" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_o2o_mask_only",
            "GDINO + SAM3 + O2O 时序关联",
            "direction_a",
            "mask_only",
            "promising",
            ["tennis", "bmx-trees"],
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_sam3_innov_o2o.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_sam3_innov_o2o.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/sam3/innovation/o2o_assoc/{sequence}",
            "这条创新是给相邻帧做更严格的一对一关联，减少 ID 混乱和错配。",
            "重点看高速运动时 mask 是否更连续、是否少了断裂和目标切换。",
            "这是目前 GDINO+SAM3 创新里最有正向信号的一条。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "o2o_assoc" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "o2o_assoc" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_quality_gate_mask_only",
            "GDINO + SAM3 + QualityGate",
            "direction_a",
            "mask_only",
            "exploratory",
            ["tennis", "bmx-trees"],
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_sam3_innov_quality_gate.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_sam3_innov_quality_gate.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/sam3/innovation/quality_gate/{sequence}",
            "QualityGate 想做的是：质量太差时，不要瞎更新。",
            "重点看门控到底有没有真的触发，而不是只看名字觉得它很高级。",
            "现有阈值下几乎没带来增益，更像一次未充分触发的尝试。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "quality_gate" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "quality_gate" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_real_vlm_mask_only",
            "GDINO + SAM3 + RealVLM prompt",
            "direction_a",
            "mask_only",
            "exploratory",
            ["tennis", "bmx-trees"],
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_sam3_innov_real_vlm.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_sam3_innov_real_vlm.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/sam3/innovation/real_vlm/{sequence}",
            "这条路线用真实 VLM caption 来生成 prompt，而不是完全手写规则。",
            "看它是不是只是在‘说法不同但效果相同’，还是确实更懂图像内容。",
            "当前已验证能跑通，但已测序列上没有额外数值收益。",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "real_vlm" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "real_vlm" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "direction_a_mask_fusion_mask_only",
            "Direction A 三路/两路 mask 融合",
            "direction_a",
            "mask_only",
            "stable",
            davis6,
            ROOT / "mask_fusion.py",
            None,
            "python part3/mask_fusion.py --sequences {sequence} --output_root /data3/jli657/project3/part3/outputs/direction_a/mask_fusion",
            "这是把不同 mask 路线按序列或逐帧策略做融合的版本。",
            "重点看它是不是在弱点序列上补洞，而不是把背景也一起吞掉。",
            "这是 Direction A 里很关键的一步，因为它承认不同序列最优路线并不一样。",
            lambda seq: {"mask_frames": ROOT / "outputs" / "direction_a" / "mask_fusion" / seq},
        ),
        ExperimentDef(
            "direction_a_shadow_geom_scale_sweep_mask_only",
            "Direction A car-shadow 阴影几何先验扫参",
            "direction_a",
            "mask_only",
            "failed_or_exploratory",
            ["car-shadow"],
            ROOT / "extend_shadow_mask.py",
            None,
            "",
            "这是专门给 car-shadow 做的阴影扩展先验，不是通用方法。",
            "重点看它是不是把阴影补进来了，同时又没有把路面误吞太多。",
            "当前扫参结果整体不如已有主线，说明这个先验还没有调到能正向贡献的程度。",
            lambda seq: {
                "source_output_dir": ROOT / "outputs" / "direction_a" / "shadow_geom" / seq,
                "log": ROOT / "outputs" / "shadow_prior.log",
            },
        ),
        ExperimentDef(
            "vggt4d_vggt_mask_only",
            "VGGT4D 原始动态发现 mask",
            "direction_b",
            "mask_only",
            "exploratory",
            davis6,
            ROOT / "direction_b/run_direction_b_vggt4d.py",
            None,
            "python part3/direction_b/run_direction_b_vggt4d.py --sequences {sequence} --output_root /data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt",
            "这是 Direction B 的原始 baseline：不靠文字 prompt，直接从 3D foundation model 的时序动态线索里找运动物体。",
            "先看 rough mask 是否覆盖到了真正的动态区域，再看边缘是不是太粗。",
            "它覆盖面有启发，但原始边缘比较粗，单独用还不够强。",
            lambda seq: {"mask_frames": ROOT / "outputs" / "direction_b" / "vggt4d_vggt" / seq},
        ),
        ExperimentDef(
            "pi3_transplant_v3_mask_only",
            "Pi3 transplant v3 mask",
            "direction_b",
            "mask_only",
            "failed",
            davis6,
            ROOT / "direction_b/run_direction_b_pi3_transplant.py",
            None,
            "python part3/direction_b/run_direction_b_pi3_transplant.py --sequences {sequence}",
            "这是把 VGGT4D 的思路移植到 Pi3 上的尝试，理论上想用更强 backbone 换更好 mask。",
            "重点看是不是完全没抓到目标，或者 mask 位置明显错了。",
            "当前结果基本失败，后续更适合作为失败分析或附录，而不是主贡献。",
            lambda seq: {"mask_frames": ROOT / "outputs" / "direction_b" / "pi3_transplant_v3" / seq},
        ),
        ExperimentDef(
            "vggt4d_sam3_refine_v2_mask_only",
            "VGGT4D + SAM3 refine v2",
            "direction_b",
            "mask_only",
            "superseded",
            davis6,
            ROOT / "direction_b/run_direction_b_sam3_refine.py",
            None,
            "conda run -n sam3_official_env python3 part3/direction_b/run_direction_b_sam3_refine.py --source_method vggt4d --sequences {sequence}",
            "这条路线是把 VGGT4D 粗 mask 当作 SAM3 的空间提示来精修边界。",
            "重点看相对原始 VGGT4D，边缘是不是明显干净了，且没有丢主体。",
            "v2 是早期版本，后面还有 v3/v4/v5，整理时会保留但不作为最终主推。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "direction_b" / "sam3_refined_v2" / "vggt4d" / seq,
                "rough_mask_frames": ROOT / "outputs" / "direction_b" / "vggt4d_vggt" / seq,
            },
        ),
        ExperimentDef(
            "vggt4d_sam3_refine_v3_mask_only",
            "VGGT4D + SAM3 refine v3",
            "direction_b",
            "mask_only",
            "partial_or_failed",
            davis6,
            ROOT / "direction_b/run_direction_b_sam3_refine.py",
            None,
            "conda run -n sam3_official_env python3 part3/direction_b/run_direction_b_sam3_refine.py --source_method vggt4d --sequences {sequence}",
            "这是 refine 的中间版本，部分序列仍有 prompt 失败或不稳定问题。",
            "重点看是否出现明显空 mask、提示失败或传播断掉。",
            "它是中间版本，应该保留在台账里，但不能当成最终结论。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "direction_b" / "sam3_refined_v3" / "vggt4d" / seq,
                "rough_mask_frames": ROOT / "outputs" / "direction_b" / "vggt4d_vggt" / seq,
            },
        ),
        ExperimentDef(
            "vggt4d_sam3_refine_v4_mask_only",
            "VGGT4D + SAM3 refine v4",
            "direction_b",
            "mask_only",
            "superseded",
            davis6,
            ROOT / "direction_b/run_direction_b_sam3_refine.py",
            None,
            "conda run -n sam3_official_env python3 part3/direction_b/run_direction_b_sam3_refine.py --source_method vggt4d --sequences {sequence}",
            "这是 refine 的较成熟版本，已经比早期版本稳很多。",
            "重点看它相对 v5 差在哪里，是召回不够还是边界还不够稳。",
            "它接近可用，但最终汇总主要还是用 v5。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "direction_b" / "sam3_refined_v4" / "vggt4d" / seq,
                "rough_mask_frames": ROOT / "outputs" / "direction_b" / "vggt4d_vggt" / seq,
            },
        ),
        ExperimentDef(
            "vggt4d_sam3_refine_v5_mask_only",
            "VGGT4D + SAM3 refine v5",
            "direction_b",
            "mask_only",
            "stable",
            davis6,
            ROOT / "direction_b/run_direction_b_sam3_refine.py",
            None,
            "conda run -n sam3_official_env python3 part3/direction_b/run_direction_b_sam3_refine.py --source_method vggt4d --sequences {sequence}",
            "这是当前 Direction B 最终保留的 refine 版本：先自动发现动态区域，再用 SAM3 把边缘精修。",
            "重点看它是否成功把粗糙的无监督发现，变成了真正可用的高质量 mask。",
            "这是 Direction B 当前最有说服力的正向结果。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "direction_b" / "sam3_refined_v5" / "vggt4d" / seq,
                "rough_mask_frames": ROOT / "outputs" / "direction_b" / "vggt4d_vggt" / seq,
            },
        ),
        # ── inpaint_only: DAVIS GT mask 公平比较组 ──────────────────────────────
        ExperimentDef(
            "pure_propainter_gtmask",
            "纯 ProPainter（DAVIS GT mask 统一口径）",
            "direction_c",
            "inpaint_only",
            "reference",
            ["tennis", "bmx-trees", "blackswan", "koala", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_propainter_gtmask.py",
            None,
            "conda run -n propainter_env python3 part3/inpainting/run_propainter_gtmask.py --seqs {sequence}",
            "所有 DAVIS 序列统一使用 DAVIS annotation / GT mask 作为输入，只比较 ProPainter 自己的修复能力。这是公平 inpaint-only 对比的基线。",
            "重点看在 GT mask 固定的前提下，ProPainter 能否稳定补背景、不闪烁。",
            "这是 GT mask 协议下 inpaint-only 的基准。",
            lambda seq: {
                "mask_frames": DAVIS_GT_MASKS / seq,
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "pure_propainter_gtmask",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "pure_propainter_gtmask" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "pure_propainter_gtmask" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sdxl_kf5_gtmask_propainter",
            "SDXL kf5 + ProPainter（DAVIS GT mask 统一口径）",
            "direction_c",
            "inpaint_only",
            "stable",
            ["tennis", "bmx-trees", "blackswan", "koala", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_phase2_sdxl_all7.py",
            None,
            "conda run -n controlnet_env python3 part3/inpainting/run_phase2_sdxl_all7.py --seqs {sequence}",
            "DAVIS 序列统一使用 DAVIS GT mask；先用 SDXL 修关键帧，再让 ProPainter 传播到全视频。与 pure_propainter_gtmask 做公平对比。",
            "重点看关键帧是否更漂亮，传播后有没有闪烁或风格不一致。",
            "GT mask 协议下 SDXL 的表现，可直接与 pure/LaMa 做工具层面公平对比。",
            lambda seq: {
                "mask_frames": DAVIS_GT_MASKS / seq,
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_gtmask_propainter",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_gtmask_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_gtmask_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "lama_gtmask_propainter",
            "LaMa + ProPainter（DAVIS GT mask 统一口径）",
            "direction_c",
            "inpaint_only",
            "stable",
            ["tennis", "bmx-trees", "blackswan", "koala", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_phase3_lama_all7.py",
            None,
            "conda run -n controlnet_env python3 part3/inpainting/run_phase3_lama_all7.py --seqs {sequence}",
            "DAVIS 序列统一使用 DAVIS GT mask；先用 LaMa 修关键区域，再交给 ProPainter 做时序传播。与 pure_propainter_gtmask 做公平对比。",
            "重点看大遮挡区域是不是比纯 ProPainter 更自然，以及有没有模糊感。",
            "GT mask 协议下 LaMa 的表现，尤其适合和 SDXL 做工具层面公平对比。",
            lambda seq: {
                "mask_frames": DAVIS_GT_MASKS / seq,
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter",
                "keyframes": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "lama_keyframes",
                "repair_frames": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "pp_input" / "frames",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "masked_in.mp4",
            },
        ),
        # ── inpaint_only: 旧版 SAM3 mask 历史记录（legacy，保留用于对比追溯）──
        ExperimentDef(
            "pure_propainter_fixed_mask",
            "纯 ProPainter（旧版 SAM3 mask，legacy）",
            "direction_c",
            "inpaint_only",
            "stable",
            all7,
            ROOT / "eval/evaluate_all.py",
            None,
            "",
            "这里固定的是同一套 mask，只比较 ProPainter 自己的修复能力。",
            "重点看在 mask 已经固定时，ProPainter 是否能稳定补背景、不闪烁。",
            "它是 inpaint-only 的强基线，经常是最难被超过的一条。",
            lambda seq: {
                "mask_frames": ROOT / "results" / seq / "mask_frames",
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "pure_propainter",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "pure_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "pure_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sdxl_kf5_propainter_fixed_mask",
            "SDXL kf5 + ProPainter（旧版 SAM3 mask，legacy）",
            "direction_c",
            "inpaint_only",
            "stable",
            all7,
            ROOT / "inpainting/run_phase2_sdxl_all7.py",
            None,
            "conda run -n controlnet_env python3 part3/inpainting/run_phase2_sdxl_all7.py --seqs {sequence}",
            "这是先用 SDXL 修关键帧，再让 ProPainter 传播到全视频的路线。",
            "重点看关键帧是否更漂亮，但传播后有没有闪烁或风格不一致。",
            "它在部分序列有视觉提升，但不是每个序列都比纯 ProPainter 稳。",
            lambda seq: {
                "mask_frames": ROOT / "results" / seq / "mask_frames",
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_propainter",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "lama_propainter_fixed_mask",
            "LaMa + ProPainter（旧版 SAM3 mask，legacy）",
            "direction_c",
            "inpaint_only",
            "stable",
            all7,
            ROOT / "inpainting/run_phase3_lama_all7.py",
            None,
            "python part3/inpainting/run_phase3_lama_all7.py --seqs {sequence}",
            "这是先用 LaMa 修关键区域，再交给 ProPainter 做时序传播。",
            "重点看大遮挡区域是不是比纯 ProPainter 更自然，以及有没有模糊感。",
            "它是很好的大遮挡对照路线，尤其适合和 SDXL 做工具层面对比。",
            lambda seq: {
                "mask_frames": ROOT / "results" / seq / "mask_frames",
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "lama_propainter",
                "keyframes": ROOT / "results" / seq / "direction_c" / "lama_propainter" / "lama_keyframes",
                "repair_frames": ROOT / "results" / seq / "direction_c" / "lama_propainter" / "pp_input" / "frames",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "lama_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "lama_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "controlnet_pure_propainter_fixed_mask",
            "ControlNet 消融：纯 ProPainter",
            "direction_c",
            "inpaint_only",
            "stable",
            ["tennis", "bmx-trees", "koala", "bear", "camel"],
            ROOT / "inpainting/run_controlnet_ablation_5seq.py",
            None,
            "python part3/inpainting/run_controlnet_ablation_5seq.py --sequences {sequence}",
            "这是在 ControlNet 消融里保留原始帧、直接走 ProPainter 的对照组。",
            "重点看它为什么经常仍然最稳，用来防止我们对生成式方法过度乐观。",
            "它是 ControlNet 消融里的基线方法，经常被推荐为最终选择。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "local_masks",
                "source_output_dir": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_pure" / seq,
                "inpaint_out": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_pure" / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_pure" / seq / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "controlnet_hybrid_propainter_fixed_mask",
            "ControlNet 消融：hybrid",
            "direction_c",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "koala", "bear", "camel"],
            ROOT / "inpainting/run_controlnet_ablation_5seq.py",
            None,
            "python part3/inpainting/run_controlnet_ablation_5seq.py --sequences {sequence}",
            "这是先做关键帧生成式修复，再与原视频拼成 hybrid frames 后交给 ProPainter。",
            "重点看生成式关键帧是不是带来了更好细节，还是引入了风格不一致。",
            "它能展示思路，但当前大多数情况下不如纯 ProPainter 稳。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "local_masks",
                "repair_frames": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "hybrid_frames",
                "source_output_dir": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_hybrid" / "hybrid_frames",
                "inpaint_out": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_hybrid" / "hybrid_frames" / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_hybrid" / "hybrid_frames" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "controlnet_hybrid_tc_propainter_fixed_mask",
            "ControlNet 消融：hybrid + temporal consistency",
            "direction_c",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "koala", "bear", "camel"],
            ROOT / "inpainting/run_controlnet_ablation_5seq.py",
            None,
            "python part3/inpainting/run_controlnet_ablation_5seq.py --sequences {sequence}",
            "这是在 hybrid 的基础上，再加一层更保守的时序一致性处理。",
            "重点看它是否真减少闪烁，还是只是看起来更稳但细节没有更好。",
            "这是个合理尝试，但目前仍然很难整体超过纯 ProPainter。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "local_masks_hybrid_tc",
                "repair_frames": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "hybrid_tc_frames",
                "source_output_dir": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_hybrid_tc" / "hybrid_tc_frames",
                "inpaint_out": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_hybrid_tc" / "hybrid_tc_frames" / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_hybrid_tc" / "hybrid_tc_frames" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sdxl_interval10_legacy_fixed_mask",
            "旧版 SDXL interval10 + ProPainter",
            "direction_c",
            "inpaint_only",
            "legacy",
            ["tennis", "bmx-trees"],
            ROOT / "inpainting/run_keyframe_sdxl_propainter.py",
            None,
            "",
            "这是更早一版的 SDXL 关键帧路线，间隔更大，只覆盖两条序列。",
            "重点看它和后来的 kf5 版本比，是不是关键帧太稀导致传播难跟上。",
            "它应该保留在台账里，但属于旧版对照，不是现在主推配置。",
            lambda seq: {
                "mask_frames": ROOT / "results" / seq / "mask_frames",
                "source_output_dir": ROOT / "outputs" / "keyframe_sdxl" / seq / "interval10" / "propainter_output" / "frames",
                "keyframes": ROOT / "outputs" / "keyframe_sdxl" / seq / "interval10" / "inpainted_keyframes",
                "inpaint_out": ROOT / "outputs" / "keyframe_sdxl" / seq / "interval10" / "propainter_output" / "frames" / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "keyframe_sdxl" / seq / "interval10" / "propainter_output" / "frames" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "koala_void_full_pipeline",
            "Koala VOID 生成式视频修复",
            "direction_c",
            "full_pipeline",
            "exploratory",
            ["koala"],
            ROOT / "inpainting/run_void_koala.py",
            None,
            "",
            "这是生成式视频修复方向的探索，不再是传统的先 mask 再补洞那种保守路线。",
            "重点看它是不是整体风格变化太大，虽然能生成，但是否还保留了原视频结构。",
            "它可以作为创新方向保留，但当前更像探索性结果，不适合直接当稳定主线。",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "sam3_multiobj" / "masks_final" / seq,
                "source_output_dir": ROOT / "outputs" / "koala_diffusion" / "void" / "final_output",
                "inpaint_out": ROOT / "outputs" / "koala_diffusion" / "void" / "final_output" / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "koala_diffusion" / "void" / "final_output" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "a_plus_b_best_full_pipeline",
            "A+B Best 按序列选优完整流程",
            "fusion",
            "full_pipeline",
            "stable",
            davis6,
            ROOT / "pipeline/run_a3_best_pipeline.py",
            None,
            "python part3/pipeline/run_a3_best_pipeline.py --sequences {sequence}",
            "这是当前最强的汇总版本：每个序列选当下最好的 mask 来源，再接 ProPainter。",
            "重点看它到底是真正稳定最优，还是只是‘每个序列挑最好看起来分数高’。",
            "它适合作为当前最优结果，但报告里必须讲清楚它是按序列路由选优，不是单一方法无脑统治全部序列。",
            lambda seq: {
                "source_output_dir": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter",
                "inpaint_out": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / seq / "masked_in.mp4",
                "masks_dilated": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / "masks_dilated",
                "log": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / f"propainter_{seq}.log",
            },
        ),
        # ── Direction C: DiffuEraser GT mask v1 (首轮 inpaint_only 最小闭环) ──
        ExperimentDef(
            "diffueraser_gtmask_v1",
            "DiffuEraser GT mask v1（inpaint_only）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py "
                "--seq {sequence} --version v1\n"
                "conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py "
                "--seq {sequence} --version v1"
            ),
            (
                "以 DAVIS GT mask 固定输入，比较 DiffuEraser（视频 diffusion inpainting）与 "
                "pure ProPainter 的修复质量。DiffuEraser 基于 BrushNet+AnimateDiff 架构，"
                "内置 ProPainter 做 prior 初始化，再用 diffusion 精修，目标是更强生成能力和时间一致性。"
                "首轮只在 tennis（70帧）验证，通过门槛后再扩序列。"
            ),
            (
                "1. masked_in.mp4：GT mask 是否完整覆盖运动员，无背景误伤\n"
                "2. inpaint_out.mp4：与 pure_propainter_gtmask 对比——是否更自然？是否闪烁？"
                "背景是否有 hallucination？非 mask 区域有无污染？\n"
                "3. PSNR_proxy / SSIM：不应明显低于 pure_propainter_gtmask"
            ),
            (
                "v1 最终状态（三版停止决策后）：PSNR_proxy=31.42 vs 基线 34.98（-3.56dB），"
                "SSIM=0.897 vs 0.930（-0.033）。v2/v3 均未改善。"
                "根本原因：DiffuEraser soft-blending 污染非 mask 区域。"
                "决策：停止扩序列，不进 full_pipeline，归档 exploratory。"
                "PSNR_synthetic (mask 区域) ≈ ProPainter，mask 内修复质量相当但无提升。"
            ),
            lambda seq: {
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "masked_in.mp4",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "inpaint_out.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "run_manifest.json",
                "log":         ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "run.log",
            },
            version="v1",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            stage_gate=(
                "工程稳定通过 + 视觉不明显劣于 pure_propainter_gtmask + "
                "PSNR_proxy 或 SSIM 至少一项接近或优于基线，另一项不显著退化"
            ),
            next_decision=(
                "三版停止条件已触发（v1/v2/v3 均低于基线 3.5-4.5 dB）。"
                "DiffuEraser 不扩序列，不进 full_pipeline。归档为 exploratory。"
                "后续可探索修改 soft-blending 策略或使用更大 neighbor_length。"
            ),
        ),
        # ── Direction C: DiffuEraser GT mask v3 (max_img_size=640, dilation=4) ──
        ExperimentDef(
            "diffueraser_gtmask_v3",
            "DiffuEraser GT mask v3（max_img_size=640, dilation=4）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "cd /data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser && "
                "conda run -n diffueraser_env python3 run_diffueraser.py "
                "--input_video .../diffueraser_gtmask_v3/input_video.mp4 "
                "--input_mask .../diffueraser_gtmask_v3/input_mask.mp4 "
                "--video_length 70 --save_path .../diffueraser_gtmask_v3 "
                "--mask_dilation_iter 4 --max_img_size 640"
            ),
            (
                "v3 最终测试：降分辨率（max_img_size=640，约 640x360）+ 中等 dilation=4。"
                "v1/v2 结果显示 dilation 不是根本问题（两版相差<0.1dB）。"
                "v3 测试分辨率是否影响背景污染。结果：更差（-4.56dB vs 基线）。"
                "三版均未改善，确认为停止条件，归档为 exploratory。"
            ),
            (
                "验证三版是否一致：v1/v2/v3 PSNR_proxy 均约 30-31dB，系统性低于基线 34.98dB。"
                "问题根因：DiffuEraser soft-blending 在边界处 feathering 污染非 mask 区域。"
                "此为架构固有问题，不建议继续调参。"
            ),
            (
                "DiffuEraser tennis 三版验证均未过门槛（v1: -3.56dB, v2: -3.66dB, v3: -4.56dB）。"
                "根本原因：soft-blended diffusion 输出影响非 mask 区域。"
                "决策：停止扩序列，不进 full_pipeline，归档为 exploratory 供参考。"
            ),
            lambda seq: {
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "masked_in.mp4",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "inpaint_out.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "run_manifest.json",
                "log":         ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v3" / "run.log",
            },
            version="v3",
            mask_protocol="davis_gt",
            baseline="diffueraser_gtmask_v2",
            stage_gate="N/A — 已触发三版停止条件",
            next_decision=(
                "三版停止条件已触发。DiffuEraser 不扩序列，不进 full_pipeline。"
                "归档为 exploratory。后续可探索修改 blending 策略或更大 neighbor_length。"
            ),
            failure_reason=(
                "v3 (dilation=4, max_img_size=640): PSNR_proxy=30.42 (-4.56dB vs 基线 34.98)，"
                "SSIM=0.871 (vs 0.930)。三版一致低于基线，确认停止。"
            ),
        ),
        # ── Direction C: DiffuEraser v4 (硬贴回，消除白色虚影) ──
        ExperimentDef(
            "diffueraser_gtmask_v4",
            "DiffuEraser GT mask v4（hard blend，消除软融合泄漏）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/apply_hard_blend.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/apply_hard_blend.py "
                "--base_version v1 --out_version v4 --sequence tennis"
            ),
            (
                "基于 v1 inpaint_out 做后处理硬贴回：mask 内保留 DiffuEraser 修复结果，"
                "mask 外强制使用 DAVIS 原始 JPEG 帧（不经 MP4 重编码）。"
                "目的：消除 soft-blending 泄漏导致的白色虚影问题。"
                "此版不重跑 DiffuEraser 推理，是对 v1 的纯后处理。"
            ),
            (
                "1. inpaint_out.mp4：非 mask 区域是否完全等同于原始视频，虚影是否消除\n"
                "2. PSNR_proxy：应显著提升（理论上 mask 外像素完全一致）\n"
                "3. PSNR_synthetic：应与 v1 相近（mask 内像素未变）\n"
                "4. 视觉质量：mask 边界是否有硬切割感"
            ),
            (
                "v4 结果（硬贴回，从 DAVIS JPEG 直接取非 mask 像素）："
                "PSNR_proxy=36.28（+1.40dB vs 基线 34.88），通过 PSNR_proxy 门槛！"
                "PSNR_synthetic=32.88（mask 区域修复质量高，与 ProPainter prior 一致）。"
                "白色虚影问题已通过硬贴回解决。决策：v4 通过门槛，可扩到其他序列。"
            ),
            lambda seq: {
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v4" / "inpaint_out.mp4",
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v4" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v4" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v4" / "masked_in.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v4" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v4" / "run_manifest.json",
            },
            version="v4",
            mask_protocol="davis_gt",
            baseline="diffueraser_gtmask_v1",
            stage_gate="PSNR_proxy>=34.88 AND SSIM>=0.92",
            next_decision=(
                "v4 通过 PSNR_proxy 门槛（36.28 ≥ 34.88）。"
                "下一步：扩到 bmx-trees 和 car-shadow 验证泛化性。"
            ),
            failure_reason="",
        ),
        # ── Direction C: DiffuEraser v5 (大 neighbor window) ──
        ExperimentDef(
            "diffueraser_gtmask_v5",
            "DiffuEraser GT mask v5（neighbor_length=20, subvideo=70）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "cd /data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser && "
                "conda run -n diffueraser_env python3 run_diffueraser.py "
                "--input_video .../diffueraser_gtmask_v5/input_video.mp4 "
                "--input_mask .../diffueraser_gtmask_v5/input_mask.mp4 "
                "--video_length 70 --save_path .../diffueraser_gtmask_v5 "
                "--mask_dilation_iter 0 --neighbor_length 20 --subvideo_length 70"
            ),
            (
                "扩大 ProPainter prior 的时序上下文窗口：neighbor_length=20（原 10），"
                "subvideo_length=70（原 50）。更长上下文有助于减少边界 feathering 和 hallucination。"
                "假设：更好的 prior 可以减少 diffusion 在不一致区域的 soft-blending 影响。"
            ),
            (
                "1. PSNR_proxy vs v1：是否因 prior 改善而减少非 mask 污染\n"
                "2. PSNR_synthetic：mask 内修复质量是否提升\n"
                "3. 视觉对比：边界区域是否更清晰"
            ),
            (
                "v5 结果：PSNR_proxy=31.32（与 v1 31.32 相近），PSNR_synthetic=8.42。"
                "扩大上下文窗口未显著改善 PSNR_proxy，根本原因仍是 soft-blending 泄漏。"
                "决策：归档 exploratory，不扩序列。"
            ),
            lambda seq: {
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "inpaint_out.mp4",
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "masked_in.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "run_manifest.json",
                "log":         ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v5" / "run.log",
            },
            version="v5",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            stage_gate="PSNR_proxy>=34.88 AND SSIM>=0.92",
            next_decision="未过门槛，归档 exploratory。soft-blending 根本问题需 v4 类硬贴回解决。",
            failure_reason=(
                "PSNR_proxy=31.32（-3.56dB vs 基线 34.88）。"
                "扩大 neighbor_length/subvideo 未改善非 mask 区域污染。"
            ),
        ),
        # ── Direction C: DiffuEraser v6 (高分辨率 max_img_size=1280) ──
        ExperimentDef(
            "diffueraser_gtmask_v6",
            "DiffuEraser GT mask v6（max_img_size=1280，原分辨率推理）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "cd /data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser && "
                "conda run -n diffueraser_env python3 run_diffueraser.py "
                "--input_video .../diffueraser_gtmask_v6/input_video.mp4 "
                "--input_mask .../diffueraser_gtmask_v6/input_mask.mp4 "
                "--video_length 70 --save_path .../diffueraser_gtmask_v6 "
                "--mask_dilation_iter 0 --max_img_size 1280"
            ),
            (
                "提高 max_img_size=1280（原 960），tennis 原始 854x480 不会被 downsample，"
                "在原分辨率下做 diffusion 推理。v3 降低分辨率（640）反而更差，"
                "假设原分辨率推理可以保留更多细节，减少上采样引入的 artifacts。"
            ),
            (
                "1. PSNR_proxy vs v1：原分辨率是否减少非 mask 污染\n"
                "2. PSNR_synthetic：mask 内生成质量是否更好\n"
                "3. 显存使用：A6000 49GB 是否能承受 1280px 全分辨率"
            ),
            (
                "v6 结果：PSNR_proxy=31.32（与 v1 相近），PSNR_synthetic=8.43。"
                "原分辨率推理未改善 PSNR_proxy，确认根本问题是 soft-blending 架构，与分辨率无关。"
                "决策：归档 exploratory。"
            ),
            lambda seq: {
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "inpaint_out.mp4",
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "masked_in.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "run_manifest.json",
                "log":         ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v6" / "run.log",
            },
            version="v6",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            stage_gate="PSNR_proxy>=34.88 AND SSIM>=0.92",
            next_decision="未过门槛，归档 exploratory。soft-blending 根本问题与分辨率无关。",
            failure_reason=(
                "PSNR_proxy=31.32（-3.56dB vs 基线 34.88）。"
                "max_img_size=1280 未改善非 mask 区域污染。"
            ),
        ),
        # ── Direction C: DiffuEraser v7 (最紧 mask, dilate_px=0) ──
        ExperimentDef(
            "diffueraser_gtmask_v7",
            "DiffuEraser GT mask v7（最紧 mask, dilate_px=0, dilation_iter=0）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py "
                "--seq tennis --version v7 --dilate_px 0\n"
                "cd /data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser && "
                "conda run -n diffueraser_env python3 run_diffueraser.py "
                "--input_video .../diffueraser_gtmask_v7/input_video.mp4 "
                "--input_mask .../diffueraser_gtmask_v7/input_mask.mp4 "
                "--video_length 70 --save_path .../diffueraser_gtmask_v7 "
                "--mask_dilation_iter 0"
            ),
            (
                "最紧 mask 实验：prepare 阶段 dilate_px=0（不外扩 GT mask），"
                "DiffuEraser 内部 mask_dilation_iter=0，输入完全对齐 GT mask 边界。"
                "假设：更小 mask 减少 soft-blending 影响的非 mask 区域面积，提升 PSNR_proxy。"
            ),
            (
                "1. PSNR_proxy vs v1（dilate=8px）：更小 mask 是否提升非 mask 区域 PSNR\n"
                "2. PSNR_synthetic：mask 内修复是否受影响（边界更紧可能漏目标）\n"
                "3. 视觉质量：目标边缘是否有遗漏（GT mask 本身是否完整覆盖目标）"
            ),
            (
                "v7 结果：PSNR_proxy=32.46（-2.43dB vs 基线 34.88），PSNR_synthetic=13.10。"
                "最紧 mask 有所改善（vs v1 -3.56dB），但仍未过门槛。"
                "PSNR_synthetic 13.10 比 v1 的 8.42 高，说明更紧 mask 使 mask 内更多区域保留了原始像素。"
                "决策：有改善趋势，但仍归档 exploratory。"
            ),
            lambda seq: {
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "inpaint_out.mp4",
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "masked_in.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "run_manifest.json",
                "log":         ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v7" / "run.log",
            },
            version="v7",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            stage_gate="PSNR_proxy>=34.88 AND SSIM>=0.92",
            next_decision=(
                "未过门槛（32.46 < 34.88），但有改善趋势。"
                "归档 exploratory。若需进一步探索可结合 v4 硬贴回思路。"
            ),
            failure_reason=(
                "PSNR_proxy=32.46（-2.43dB vs 基线 34.88）。"
                "最紧 mask 有改善但仍未过门槛。PSNR_synthetic=13.10 异常高（更多原始像素被计入 mask 内）。"
            ),
        ),
        # ── Direction C: DiffuEraser v8 (密参考帧 ref_stride=5) ──
        ExperimentDef(
            "diffueraser_gtmask_v8",
            "DiffuEraser GT mask v8（ref_stride=5，密参考帧）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "cd /data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser && "
                "conda run -n diffueraser_env python3 run_diffueraser.py "
                "--input_video .../diffueraser_gtmask_v8/input_video.mp4 "
                "--input_mask .../diffueraser_gtmask_v8/input_mask.mp4 "
                "--video_length 70 --save_path .../diffueraser_gtmask_v8 "
                "--mask_dilation_iter 0 --ref_stride 5"
            ),
            (
                "密参考帧实验：ref_stride=5（原 10），每 5 帧取一个参考帧。"
                "更密的参考帧让 ProPainter prior 有更连贯的时序信息，"
                "有望减少 diffusion 在不一致 prior 上的 hallucination 和边界 blending 问题。"
            ),
            (
                "1. PSNR_proxy vs v1：更密参考帧是否减少非 mask 污染\n"
                "2. PSNR_synthetic：时序连贯性是否改善\n"
                "3. 运行时间：ref_stride=5 比 10 慢约 2x，需记录耗时"
            ),
            (
                "v8 结果：PSNR_proxy=31.31（与 v1 相近），PSNR_synthetic=8.41。"
                "密参考帧未改善 PSNR_proxy，soft-blending 根本问题不受 ref_stride 影响。"
                "决策：归档 exploratory。8 版调参方向已全部探索完毕。"
            ),
            lambda seq: {
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "inpaint_out.mp4",
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "masked_in.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "run_manifest.json",
                "log":         ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v8" / "run.log",
            },
            version="v8",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            stage_gate="PSNR_proxy>=34.88 AND SSIM>=0.92",
            next_decision=(
                "未过门槛，归档 exploratory。8 版 DiffuEraser 调参探索完毕。"
                "结论：仅 v4（硬贴回）通过门槛，其余版本均因 soft-blending 架构问题失败。"
            ),
            failure_reason=(
                "PSNR_proxy=31.31（-3.57dB vs 基线 34.88）。"
                "ref_stride=5 未改善非 mask 区域污染，与 soft-blending 根本原因无关。"
            ),
        ),
        # ── Direction C: DiffuEraser smoke test (工程验证，不进最终对比表) ──
        ExperimentDef(
            "diffueraser_smoke_v1",
            "DiffuEraser 环境 smoke test（工程验证）",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py "
                "--seq tennis --version v1 --smoke_test"
            ),
            "工程 smoke test：确认 DiffuEraser 环境能正确导入，输入视频格式兼容。不产生最终 inpaint_out.mp4。",
            "只看导入是否成功，不评价视觉质量。",
            "待 smoke test 运行后更新",
            lambda seq: {
                "manifest": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "run_manifest.json",
            },
            version="v1",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            next_decision="smoke 通过后运行完整 v1 推理",
        ),
    ]


EXPERIMENTS = build_defs()


def path_exists(path: Optional[Path]) -> bool:
    return bool(path) and str(path) and path.exists()


def safe_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            return
        dst.unlink()
    os.symlink(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_mask_preview(mask_dir: Path, frame_dir: Path, output_mp4: Path) -> bool:
    if output_mp4.exists() or not mask_dir.exists() or not frame_dir.exists():
        return output_mp4.exists()
    frame_paths = sorted([p for p in frame_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda p: p.stem)
    mask_paths = sorted(mask_dir.glob("*.png"), key=lambda p: p.stem)
    if not frame_paths or not mask_paths:
        return False
    ref = cv2.imread(str(frame_paths[0]))
    if ref is None:
        return False
    h, w = ref.shape[:2]
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    n = min(len(frame_paths), len(mask_paths))
    for i in range(n):
        frame = cv2.imread(str(frame_paths[i]))
        mask = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        elif mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask > 127).astype(np.uint8)
        green = np.zeros_like(frame)
        green[:, :, 1] = 255
        overlay = frame.copy().astype(np.float32)
        pos = mask_bin > 0
        overlay[pos] = 0.45 * overlay[pos] + 0.55 * green[pos]
        writer.write(overlay.astype(np.uint8))
    writer.release()
    return output_mp4.exists()


def family_label(exp: ExperimentDef) -> str:
    mapping = {
        "baseline": "基线/历史参照",
        "direction_a": "Direction A（Mask Upgrade）",
        "direction_b": "Direction B（Better Mask）",
        "direction_c": "Direction C（Inpainting）",
        "fusion": "A+B 融合",
    }
    return mapping.get(exp.family, exp.family)


def build_root_readme(registry_rows: List[Dict[str, Any]]) -> str:
    stable = sum(1 for r in registry_rows if r["audit_status"] in {"stable", "reference", "promising"})
    failed = sum(1 for r in registry_rows if "failed" in r["audit_status"])
    return f"""# Part3 Deliverables

这个目录是新的可审计交付入口，不改旧实验本体，只做统一入口和说明。

## 先看什么

1. 先进某个 sequence 目录，比如 `tennis/`
2. 看 `00_readme.md`
3. 对每个方法先看 `masked_in.mp4`
4. 再看 `inpaint_out.mp4`
5. 最后看 `experiment_card.md` 和 `metrics.json`

## 三类实验

- `mask_only`: 只比较 mask
- `inpaint_only`: 固定 mask，只比较修复工具
- `full_pipeline`: mask + inpaint 一起比较

## 审计状态说明

- `reference`: 历史参照，不一定是最终最强，但必须保留
- `stable`: 当前可直接用于汇报的稳定结果
- `promising`: 有正向信号，值得继续精进
- `exploratory`: 探索性结果，能说明问题，但不宜直接当最终结论
- `legacy`: 旧版路线，保留做对照
- `superseded`: 被后续版本替代，但仍然保留痕迹
- `partial_or_failed` / `failed`: 有明显失败或不完整问题

## 这次整理包含的实验卡片

- 总行数: `{len(registry_rows)}`
- 可直接重点看的稳定/参照项: `{stable}`
- 明确失败或需要谨慎看的项: `{failed}`

中文大白话总说明见：`../DELIVERABLES_GUIDE_CN.md`
"""


def build_sequence_readme(seq: str, rows: List[Dict[str, Any]]) -> str:
    meta = SEQUENCES.get(seq, {})
    lines = [
        f"# {seq}",
        "",
        f"- 这个素材难点：{meta.get('difficulty', '待补充')}",
        "- 先看 `masked_in.mp4` 判断 mask，再看 `inpaint_out.mp4` 判断修复。",
        "- `mask_only` 看遮罩质量；`inpaint_only` 看固定 mask 下修复工具；`full_pipeline` 看最终交付视频。",
        "",
        "## 当前已整理的方法",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['method_id']}` | {row['comparison_type']} | {row['audit_status']} | {row['readable_name']}")
        lines.append(f"  这条方法一句话：{row['plain_explanation']}")
    return "\n".join(lines) + "\n"


def build_card(seq: str, exp: ExperimentDef, metrics: Dict[str, Any], existing: Dict[str, str]) -> str:
    evidence_lines = []
    for key, value in existing.items():
        evidence_lines.append(f"- `{key}`: `{value}`")
    if not evidence_lines:
        evidence_lines.append("- 当前没有对上的实体路径，需要复查。")
    metrics_lines = []
    for key, value in metrics.items():
        if key == "source_file":
            continue
        metrics_lines.append(f"- `{key}`: `{value}`")
    if not metrics_lines:
        metrics_lines.append("- 当前未在统一指标表里找到对应数值。")

    # ── 固定四问（schema v2） ─────────────────────────────────────────────
    mask_src_answer = exp.mask_protocol if exp.mask_protocol else "未显式指定，请补充 mask_protocol 字段"
    four_q = f"""### Q1  这条实验到底在比什么？
{exp.plain_explanation}

### Q2  这个实验用的 mask 是哪来的？
- **mask_protocol**: `{mask_src_answer}`
- DAVIS 默认：`DAVIS annotation / GT mask`，路径 `/home/jli657/shared_data/project3/DAVIS/Annotations/480p/<seq>`
- 非 DAVIS 或 wild video 必须在 `mask_protocol` 字段中说明来源。

### Q3  修复视频是哪个工具做的？
- 修复工具见 `family`：`{exp.family}`
- 典型工具：ProPainter（PP）、LaMa、SDXL、DiffuEraser、VOID、ControlNet

### Q4  结果不好时，是 mask 的锅，还是 inpainting 的锅？
- 先看 `masked_in.mp4`：覆盖目标是否完整？有无误伤背景？时间是否抖动？
- 再看 `inpaint_out.mp4`：修复区域是否自然？是否闪烁 / hallucination？
- 只有两个视频都合理，`PSNR / SSIM / JM / JR / F` 指标才能作为结论证据。"""

    # ── Version History（schema v2） ─────────────────────────────────────
    baseline_note = exp.baseline if exp.baseline else "（首版或无可比基线）"
    failure_note = f"\n- **failure_reason**: {exp.failure_reason}" if exp.failure_reason else ""
    next_note = f"\n- **next_decision**: {exp.next_decision}" if exp.next_decision else ""
    stage_note = f"\n- **stage_gate**: {exp.stage_gate}" if exp.stage_gate else ""
    version_history = f"""| 字段 | 值 |
|---|---|
| version | `{exp.version}` |
| based_on | `{baseline_note}` |
| changes_from_previous | 见下方"当前结论"与 `exp_id` 命名增量 |
| motivation | {exp.what_to_check} |
| expected_gain | 见 `plain_explanation` |
| actual_result | {exp.current_takeaway}{failure_note} |
| next_decision | {exp.next_decision if exp.next_decision else "待实验结果更新"} |{stage_note}"""

    return f"""# {exp.readable_name}

## 这是什么

{exp.plain_explanation}

## 这条实验属于哪一类

- `{exp.comparison_type}`
- 家族：{family_label(exp)}
- 审计状态：`{exp.audit_status}`
- 版本：`{exp.version}`
- mask 协议：`{exp.mask_protocol if exp.mask_protocol else "未指定"}`
- 对比基线：`{exp.baseline if exp.baseline else "无"}`

## 固定四问

{four_q}

## 你看这个实验时重点看什么

{exp.what_to_check}

## 当前结论

{exp.current_takeaway}

{(f"> ⚠ 失败 / 停用原因：{exp.failure_reason}" + chr(10) + chr(10)) if exp.failure_reason else ""}## Version History / 版本演化

{version_history}

## 关键路径

{chr(10).join(evidence_lines)}

## 指标摘录

{chr(10).join(metrics_lines)}

---
*generated by build_part3_deliverables.py — schema v2*
"""


def build_exp_from_manifest(manifest_path: Path) -> Optional[ExperimentDef]:
    """
    Build an ExperimentDef on-the-fly from a run_manifest.json.

    This allows experiments to auto-register without requiring a static
    ExperimentDef entry in build_defs(). The manifest must satisfy the
    schema defined in pipeline/run_manifest_schema.py.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = Path(data.get("output_dir", manifest_path.parent))

    def _path_builder(seq: str) -> Dict[str, Path]:
        pb: Dict[str, Path] = {"manifest": manifest_path}
        for key in ("inpaint_out", "masked_in", "mask_frames_dir", "log_path"):
            val = data.get(key)
            if val:
                pb[key.replace("_dir", "_frames")] = Path(val)
        if out_dir.exists():
            for candidate in ("inpaint_out.mp4", "masked_in.mp4"):
                p = out_dir / candidate
                if p.exists():
                    pb.setdefault(candidate.replace(".mp4", ""), p)
            mask_dir = out_dir / "mask_frames"
            if mask_dir.exists():
                pb.setdefault("mask_frames", mask_dir)
        return pb

    return ExperimentDef(
        exp_id=data["exp_id"],
        readable_name=data.get("readable_name", data["exp_id"]),
        family=data.get("family", "unknown"),
        comparison_type=data.get("comparison_type", "inpaint_only"),
        audit_status=data.get("audit_status", "exploratory"),
        sequences=[data["sequence"]],
        script_path=Path(data["script_path"]) if data.get("script_path") else None,
        config_path=Path(data["config_path"]) if data.get("config_path") else None,
        command_template=data.get("command", "# see run_manifest.json"),
        plain_explanation=data.get("plain_explanation", ""),
        what_to_check=data.get("what_to_check", ""),
        current_takeaway=data.get("current_takeaway", "待实验结果填写"),
        path_builder=_path_builder,
        version=data.get("version", "v1"),
        mask_protocol=data.get("mask_protocol", ""),
        baseline=data.get("baseline", ""),
        stage_gate=data.get("stage_gate", ""),
        next_decision=data.get("next_decision", ""),
        failure_reason=data.get("failure_reason", ""),
    )


def collect_manifests_from_results() -> List[ExperimentDef]:
    """
    Scan results/ directory for run_manifest.json files not already in EXPERIMENTS.
    Returns ExperimentDef instances for each discovered manifest.
    """
    existing_ids = {e.exp_id for e in EXPERIMENTS}
    discovered: List[ExperimentDef] = []
    results_root = Path("/data3/jli657/project3/part3/results")
    if not results_root.exists():
        return discovered
    for manifest_path in sorted(results_root.glob("**/run_manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            exp_id = data.get("exp_id", "")
            if exp_id and exp_id not in existing_ids:
                exp = build_exp_from_manifest(manifest_path)
                if exp:
                    discovered.append(exp)
                    existing_ids.add(exp_id)
        except Exception as e:
            print(f"[warn] failed to parse manifest {manifest_path}: {e}")
    return discovered


def main() -> None:
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Build part3_deliverables")
    parser.add_argument(
        "--manifest", default=None,
        help="Path to a single run_manifest.json to ingest (triggers targeted rebuild)"
    )
    args = parser.parse_args()

    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    # If a single manifest is passed, ingest it and exit early
    if args.manifest:
        manifest_path = Path(args.manifest)
        exp = build_exp_from_manifest(manifest_path)
        if exp is None:
            print(f"[error] failed to parse manifest: {manifest_path}")
            return
        seq = exp.sequences[0]
        if seq not in SEQUENCES:
            print(f"[warn] sequence '{seq}' not in SEQUENCES config; adding with default frame dir")
        print(f"[manifest] ingesting {exp.exp_id} / {seq}")
        method_dir = DELIVERABLES / seq / exp.exp_id
        method_dir.mkdir(parents=True, exist_ok=True)
        paths = exp.path_builder(seq)
        existing_paths: Dict[str, str] = {}
        for key, path in paths.items():
            if isinstance(path, Path) and path.exists():
                existing_paths[key] = str(path)
                if path.is_dir():
                    safe_symlink(path, method_dir / key)
                else:
                    dst = method_dir / (path.name if key in {"run_meta", "manifest", "log"} else path.name)
                    safe_symlink(path, dst)
        if exp.script_path and exp.script_path.exists():
            existing_paths["script_path"] = str(exp.script_path)
            safe_symlink(exp.script_path, method_dir / exp.script_path.name)
        metrics = metric_for(exp.exp_id, seq)
        metrics_json = {
            "sequence": seq, "method_id": exp.exp_id,
            "readable_name": exp.readable_name, "family": exp.family,
            "comparison_type": exp.comparison_type, "audit_status": exp.audit_status,
            "version": exp.version, "mask_protocol": exp.mask_protocol,
            "baseline": exp.baseline, "stage_gate": exp.stage_gate,
            "next_decision": exp.next_decision, "failure_reason": exp.failure_reason,
            "metrics": metrics, "evidence_paths": existing_paths,
        }
        write_text(method_dir / "metrics.json", json.dumps(metrics_json, indent=2, ensure_ascii=False))
        write_text(method_dir / "command.sh", exp.command_template.strip() + "\n")
        write_text(method_dir / "experiment_card.md", build_card(seq, exp, metrics, existing_paths))
        print(f"[done] {method_dir}")
        print("Run without --manifest to rebuild the full deliverables + registry.")
        return

    registry_rows: List[Dict[str, Any]] = []
    per_seq_rows: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SEQUENCES}

    # Merge static defs + auto-discovered manifests
    all_exps = list(EXPERIMENTS) + collect_manifests_from_results()

    for exp in all_exps:
        for seq in exp.sequences:
            if seq not in SEQUENCES:
                continue
            method_dir = DELIVERABLES / seq / exp.exp_id
            method_dir.mkdir(parents=True, exist_ok=True)
            paths = exp.path_builder(seq)
            existing_paths: Dict[str, str] = {}

            # Link core source files
            for key, path in paths.items():
                if not isinstance(path, Path) or not str(path):
                    continue
                if path.exists():
                    existing_paths[key] = str(path)
                    name = key
                    if key in {"script", "config"}:
                        safe_symlink(path, method_dir / path.name)
                    elif path.is_dir():
                        safe_symlink(path, method_dir / key)
                    else:
                        dst_name = path.name if key in {"run_meta", "manifest", "log"} else key if "." in key else path.name
                        safe_symlink(path, method_dir / dst_name)

            # Script / config links
            if exp.script_path and exp.script_path.exists():
                existing_paths["script_path"] = str(exp.script_path)
                safe_symlink(exp.script_path, method_dir / exp.script_path.name)
            if exp.config_path and exp.config_path.exists():
                existing_paths["config_path"] = str(exp.config_path)
                safe_symlink(exp.config_path, method_dir / exp.config_path.name)

            # If masked preview missing but mask dir exists, generate one for quick inspection
            frame_dir = Path(SEQUENCES[seq]["frame_dir"])
            mask_dir = paths.get("mask_frames")
            masked_out = method_dir / "masked_in.mp4"
            if not masked_out.exists() and isinstance(mask_dir, Path) and mask_dir.exists() and frame_dir.exists():
                if make_mask_preview(mask_dir, frame_dir, masked_out):
                    existing_paths["generated_masked_in"] = str(masked_out)

            metrics = metric_for(exp.exp_id, seq)
            metrics_json = {
                "sequence": seq,
                "method_id": exp.exp_id,
                "readable_name": exp.readable_name,
                "family": exp.family,
                "comparison_type": exp.comparison_type,
                "audit_status": exp.audit_status,
                # schema v2 fields
                "version": exp.version,
                "mask_protocol": exp.mask_protocol,
                "baseline": exp.baseline,
                "stage_gate": exp.stage_gate,
                "next_decision": exp.next_decision,
                "failure_reason": exp.failure_reason,
                "metrics": metrics,
                "evidence_paths": existing_paths,
            }
            write_text(method_dir / "metrics.json", json.dumps(metrics_json, indent=2, ensure_ascii=False))
            write_text(method_dir / "command.sh", exp.command_template.format(sequence=seq).strip() + "\n")
            write_text(method_dir / "experiment_card.md", build_card(seq, exp, metrics, existing_paths))

            row = {
                "sequence": seq,
                "method_id": exp.exp_id,
                "readable_name": exp.readable_name,
                "family": exp.family,
                "comparison_type": exp.comparison_type,
                "audit_status": exp.audit_status,
                # schema v2 fields
                "version": exp.version,
                "mask_protocol": exp.mask_protocol,
                "baseline": exp.baseline,
                "stage_gate": exp.stage_gate,
                "next_decision": exp.next_decision,
                "failure_reason": exp.failure_reason,
                "script_path": str(exp.script_path) if exp.script_path else "",
                "config_path": str(exp.config_path) if exp.config_path else "",
                "deliverable_dir": str(method_dir),
                "plain_explanation": exp.plain_explanation,
                "current_takeaway": exp.current_takeaway,
                "metrics_source": metrics.get("source_file", "") if metrics else "",
            }
            registry_rows.append(row)
            per_seq_rows[seq].append(row)

    # Root README: skip overwrite if the strong-constraint version is already in place
    _root_readme = DELIVERABLES / "README.md"
    _skip_root_readme = (
        _root_readme.exists()
        and "实验完成定义（强约束）" in _root_readme.read_text(encoding="utf-8")
    )
    if not _skip_root_readme:
        write_text(_root_readme, build_root_readme(registry_rows))
    else:
        print(f"[skip] {_root_readme} — 强约束版已存在，不覆盖")
    for seq, rows in per_seq_rows.items():
        write_text(DELIVERABLES / seq / "00_readme.md", build_sequence_readme(seq, rows))

    # Registry CSV / JSON
    registry_json = DELIVERABLES / "experiment_registry.json"
    registry_csv = DELIVERABLES / "experiment_registry.csv"
    write_text(registry_json, json.dumps(registry_rows, indent=2, ensure_ascii=False))
    with registry_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "sequence",
            "method_id",
            "readable_name",
            "family",
            "comparison_type",
            "audit_status",
            "version",
            "mask_protocol",
            "baseline",
            "stage_gate",
            "next_decision",
            "failure_reason",
            "script_path",
            "config_path",
            "deliverable_dir",
            "metrics_source",
            "plain_explanation",
            "current_takeaway",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(registry_rows)

    print(f"[done] deliverables root: {DELIVERABLES}")
    print(f"[done] registry json: {registry_json}")
    print(f"[done] registry csv: {registry_csv}")


if __name__ == "__main__":
    main()
