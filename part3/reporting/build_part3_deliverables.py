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
        "difficulty": "",
    },
    "bmx-trees": {
        "frame_dir": str(DAVIS_FRAMES / "bmx-trees"),
        "difficulty": "/",
    },
    "blackswan": {
        "frame_dir": str(DAVIS_FRAMES / "blackswan"),
        "difficulty": "",
    },
    "car-shadow": {
        "frame_dir": str(DAVIS_FRAMES / "car-shadow"),
        "difficulty": "",
    },
    "horsejump-low": {
        "frame_dir": str(DAVIS_FRAMES / "horsejump-low"),
        "difficulty": "",
    },
    "koala": {
        "frame_dir": str(DAVIS_FRAMES / "koala"),
        "difficulty": "",
    },
    "wild_video-1person": {
        "frame_dir": str(WILD_FRAMES / "wild_video-1person"),
        "difficulty": " DAVIS ",
    },
    "bear": {
        "frame_dir": str(DAVIS_FRAMES / "bear"),
        "difficulty": "",
    },
    "camel": {
        "frame_dir": str(DAVIS_FRAMES / "camel"),
        "difficulty": "",
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
        # DiffuEraser versions
        "diffueraser_gtmask_v1": "part3_diffueraser_gtmask_v1",
        "diffueraser_gtmask_v2": "part3_diffueraser_gtmask_v2",
        "diffueraser_gtmask_v3": "part3_diffueraser_gtmask_v3",
        "diffueraser_gtmask_v4": "part3_diffueraser_gtmask_v4",
        "diffueraser_gtmask_v5": "part3_diffueraser_gtmask_v5",
        "diffueraser_gtmask_v6": "part3_diffueraser_gtmask_v6",
        "diffueraser_gtmask_v7": "part3_diffueraser_gtmask_v7",
        "diffueraser_gtmask_v8": "part3_diffueraser_gtmask_v8",
        "diffueraser_gtmask_v9": "part3_diffueraser_gtmask_v9",
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
    # ──  schema v2  ExperimentDef ───────
    version: str = field(default="legacy")
    """ v1 / v2 / v3 / legacy"""
    mask_protocol: str = field(default="")
    """mask davis_gt | sam3_mask | wild_existing_mask | controlnet_gt | ="""
    baseline: str = field(default="")
    """ exp_id pure_propainter_gtmask"""
    stage_gate: str = field(default="")
    """"""
    next_decision: str = field(default="")
    """ /  /  /  full_pipeline"""
    failure_reason: str = field(default="")
    """ audit_status  failed / partial_or_failed / superseded """


def build_defs() -> List[ExperimentDef]:
    davis5 = ["tennis", "bmx-trees", "blackswan", "car-shadow", "horsejump-low"]
    davis6 = davis5 + ["koala"]
    all7 = davis6 + ["wild_video-1person"]
    big_set = davis6 + ["bear", "camel"]
    return [
        ExperimentDef(
            "part2_mask_baseline",
            "Part2 YOLO+SAM2  mask",
            "baseline",
            "mask_only",
            "reference",
            davis6 + ["wild_video-1person"],
            None,
            None,
            "",
            " Part2 ",
            " tennis / bmx-trees / car-shadow  Part3 ",
            " Part3 ",
            lambda seq: {"mask_frames": part2_mask_dir(seq) or Path("")},
        ),
        ExperimentDef(
            "part2_baseline_full_pipeline",
            "Part2 YOLO+SAM2+ProPainter ",
            "baseline",
            "full_pipeline",
            "reference",
            ["tennis", "bmx-trees", "wild_video-1person"],
            None,
            None,
            "",
            "YOLO+SAM2  mask ProPainter ",
            " masked_in.mp4  mask  inpaint_out.mp4 ",
            "",
            lambda seq: {
                "mask_frames": part2_mask_dir(seq) or Path(""),
                "source_output_dir": part2_video_dir(seq) or Path(""),
                "inpaint_out": (part2_video_dir(seq) or Path("")) / "inpaint_out.mp4",
                "masked_in": (part2_video_dir(seq) or Path("")) / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "official_sam3_video_mask_only",
            " SAM3  prompt ",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "direction_a/run_official_sam3_video.py",
            None,
            "python part3/direction_a/run_official_sam3_video.py --sequences {sequence} --output_root part3/outputs/official_sam3_video/masks",
            " SAM3  prompt  prompt ",
            " prompt ",
            " SAM3  prompt ",
            lambda seq: {"mask_frames": ROOT / "outputs" / "official_sam3_video" / "masks" / seq},
        ),
        ExperimentDef(
            "official_sam3_best_mask_only",
            " SAM3 best-prompt ",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "direction_a/run_official_sam3_video.py",
            None,
            "python part3/direction_a/run_official_sam3_video.py --sequences {sequence} --output_root part3/outputs/official_sam3_best/masks",
            " SAM3  prompt  prompt ",
            " prompt ",
            " prompt  prompt ",
            lambda seq: {"mask_frames": ROOT / "outputs" / "official_sam3_best" / "masks" / seq},
        ),
        ExperimentDef(
            "sam3_multiobj_mask_only",
            "SAM3 multi-object  prompt mask",
            "direction_a",
            "mask_only",
            "stable",
            all7,
            ROOT / "direction_a/run_sam3_multiobject.py",
            ROOT / "configs" / "prompt_scope.yaml",
            "python part3/direction_a/run_sam3_multiobject.py --scope_yaml part3/configs/prompt_scope.yaml --sequences {sequence}",
            " Part3 Direction A  prompt  mask ",
            " prompt ",
            " Direction A  SAM3 ",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "sam3_multiobj" / "masks_final" / seq,
                "source_output_dir": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq,
                "inpaint_out": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "sam3_multiobj" / "propainter" / seq / seq / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sam3_multiobj_propainter_full_pipeline",
            "SAM3 multi-object + ProPainter ",
            "direction_a",
            "full_pipeline",
            "stable",
            all7,
            ROOT / "direction_a/run_sam3_multiobject.py",
            ROOT / "configs" / "prompt_scope.yaml",
            "python part3/direction_a/run_sam3_multiobject.py --scope_yaml part3/configs/prompt_scope.yaml --sequences {sequence}",
            " SAM3 multi-object mask  ProPainter ",
            " masked_in.mp4  mask inpaint_out.mp4  ProPainter ",
            " Direction A ",
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
            " SAM3  multi-object ",
            " multi-object ",
            "",
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
            " rebuild v1  ProPainter  multi-object ",
            " masked_in.mp4  multi-object  mask",
            "",
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
            "GDINO  SAM2  mask ",
            "",
            " bmx-trees  SAM3 ",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "stage1" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "stage1" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam2_stage2_mask_only",
            "GDINO + SAM2 Stage2  mask",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_mainline.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_mainline.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/stage2/{sequence}",
            "",
            "",
            " Stage1 ",
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
            "GDINO SAM3  SAM2  SAM3 ",
            " SAM3  bmx-trees  tennis",
            " GDINO+SAM2 Stage1",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage1" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage1" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_stage2_mask_only",
            "GDINO + SAM3 Stage2  mask",
            "direction_a",
            "mask_only",
            "exploratory",
            davis5,
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_mainline_sam3_stage2.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_mainline_sam3_stage2.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/sam3/stage2/{sequence}",
            " GDINO+SAM3 ",
            "",
            " GDINO+SAM3 Stage1 ",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage2" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "stage2" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "gdino_sam3_o2o_mask_only",
            "GDINO + SAM3 + O2O ",
            "direction_a",
            "mask_only",
            "promising",
            ["tennis", "bmx-trees"],
            ROOT / "gdino_vlm" / "run_gdino_mainline.py",
            ROOT / "configs" / "gdino_vlm_sam3_innov_o2o.yaml",
            "python part3/gdino_vlm/run_gdino_mainline.py --config part3/configs/gdino_vlm_sam3_innov_o2o.yaml --sequence {sequence} --stage stage2 --output part3/gdino_vlm/masks/sam3/innovation/o2o_assoc/{sequence}",
            " ID ",
            " mask ",
            " GDINO+SAM3 ",
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
            "QualityGate ",
            "",
            "",
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
            " VLM caption  prompt",
            "",
            "",
            lambda seq: {
                "mask_frames": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "real_vlm" / seq,
                "run_meta": ROOT / "gdino_vlm" / "masks" / "sam3" / "innovation" / "real_vlm" / seq / "run_meta.json",
            },
        ),
        ExperimentDef(
            "direction_a_mask_fusion_mask_only",
            "Direction A / mask ",
            "direction_a",
            "mask_only",
            "stable",
            davis6,
            ROOT / "mask_fusion.py",
            None,
            "python part3/mask_fusion.py --sequences {sequence} --output_root /data3/jli657/project3/part3/outputs/direction_a/mask_fusion",
            " mask ",
            "",
            " Direction A ",
            lambda seq: {"mask_frames": ROOT / "outputs" / "direction_a" / "mask_fusion" / seq},
        ),
        ExperimentDef(
            "direction_a_shadow_geom_scale_sweep_mask_only",
            "Direction A car-shadow ",
            "direction_a",
            "mask_only",
            "failed_or_exploratory",
            ["car-shadow"],
            ROOT / "extend_shadow_mask.py",
            None,
            "",
            " car-shadow ",
            "",
            "",
            lambda seq: {
                "source_output_dir": ROOT / "outputs" / "direction_a" / "shadow_geom" / seq,
                "log": ROOT / "outputs" / "shadow_prior.log",
            },
        ),
        ExperimentDef(
            "vggt4d_vggt_mask_only",
            "VGGT4D  mask",
            "direction_b",
            "mask_only",
            "exploratory",
            davis6,
            ROOT / "direction_b/run_direction_b_vggt4d.py",
            None,
            "python part3/direction_b/run_direction_b_vggt4d.py --sequences {sequence} --output_root /data3/jli657/project3/part3/outputs/direction_b/vggt4d_vggt",
            " Direction B  baseline prompt 3D foundation model ",
            " rough mask ",
            "",
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
            " VGGT4D  Pi3  backbone  mask",
            " mask ",
            "",
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
            " VGGT4D  mask  SAM3 ",
            " VGGT4D",
            "v2  v3/v4/v5",
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
            " refine  prompt ",
            " mask",
            "",
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
            " refine ",
            " v5 ",
            " v5",
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
            " Direction B  refine  SAM3 ",
            " mask",
            " Direction B ",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "direction_b" / "sam3_refined_v5" / "vggt4d" / seq,
                "rough_mask_frames": ROOT / "outputs" / "direction_b" / "vggt4d_vggt" / seq,
            },
        ),
        # ── inpaint_only: DAVIS GT mask  ──────────────────────────────
        ExperimentDef(
            "pure_propainter_gtmask",
            " ProPainterDAVIS GT mask ",
            "direction_c",
            "inpaint_only",
            "reference",
            ["tennis", "bmx-trees", "blackswan", "koala", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_propainter_gtmask.py",
            None,
            "conda run -n propainter_env python3 part3/inpainting/run_propainter_gtmask.py --seqs {sequence}",
            " DAVIS  DAVIS annotation / GT mask  ProPainter  inpaint-only ",
            " GT mask ProPainter ",
            " GT mask  inpaint-only ",
            lambda seq: {
                "mask_frames": DAVIS_GT_MASKS / seq,
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "pure_propainter_gtmask",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "pure_propainter_gtmask" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "pure_propainter_gtmask" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sdxl_kf5_gtmask_propainter",
            "SDXL kf5 + ProPainterDAVIS GT mask ",
            "direction_c",
            "inpaint_only",
            "stable",
            ["tennis", "bmx-trees", "blackswan", "koala", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_phase2_sdxl_all7.py",
            None,
            "conda run -n controlnet_env python3 part3/inpainting/run_phase2_sdxl_all7.py --seqs {sequence}",
            "DAVIS  DAVIS GT mask SDXL  ProPainter  pure_propainter_gtmask ",
            "",
            "GT mask  SDXL  pure/LaMa ",
            lambda seq: {
                "mask_frames": DAVIS_GT_MASKS / seq,
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_gtmask_propainter",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_gtmask_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_gtmask_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "lama_gtmask_propainter",
            "LaMa + ProPainterDAVIS GT mask ",
            "direction_c",
            "inpaint_only",
            "stable",
            ["tennis", "bmx-trees", "blackswan", "koala", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_phase3_lama_all7.py",
            None,
            "conda run -n controlnet_env python3 part3/inpainting/run_phase3_lama_all7.py --seqs {sequence}",
            "DAVIS  DAVIS GT mask LaMa  ProPainter  pure_propainter_gtmask ",
            " ProPainter ",
            "GT mask  LaMa  SDXL ",
            lambda seq: {
                "mask_frames": DAVIS_GT_MASKS / seq,
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter",
                "keyframes": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "lama_keyframes",
                "repair_frames": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "pp_input" / "frames",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "lama_gtmask_propainter" / "masked_in.mp4",
            },
        ),
        # ── inpaint_only:  SAM3 mask legacy──
        ExperimentDef(
            "pure_propainter_fixed_mask",
            " ProPainter SAM3 masklegacy",
            "direction_c",
            "inpaint_only",
            "stable",
            all7,
            ROOT / "eval/evaluate_all.py",
            None,
            "",
            " mask ProPainter ",
            " mask ProPainter ",
            " inpaint-only ",
            lambda seq: {
                "mask_frames": ROOT / "results" / seq / "mask_frames",
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "pure_propainter",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "pure_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "pure_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "sdxl_kf5_propainter_fixed_mask",
            "SDXL kf5 + ProPainter SAM3 masklegacy",
            "direction_c",
            "inpaint_only",
            "stable",
            all7,
            ROOT / "inpainting/run_phase2_sdxl_all7.py",
            None,
            "conda run -n controlnet_env python3 part3/inpainting/run_phase2_sdxl_all7.py --seqs {sequence}",
            " SDXL  ProPainter ",
            "",
            " ProPainter ",
            lambda seq: {
                "mask_frames": ROOT / "results" / seq / "mask_frames",
                "source_output_dir": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_propainter",
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_propainter" / "inpaint_out.mp4",
                "masked_in": ROOT / "results" / seq / "direction_c" / "sdxl_kf5_propainter" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "lama_propainter_fixed_mask",
            "LaMa + ProPainter SAM3 masklegacy",
            "direction_c",
            "inpaint_only",
            "stable",
            all7,
            ROOT / "inpainting/run_phase3_lama_all7.py",
            None,
            "python part3/inpainting/run_phase3_lama_all7.py --seqs {sequence}",
            " LaMa  ProPainter ",
            " ProPainter ",
            " SDXL ",
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
            "ControlNet  ProPainter",
            "direction_c",
            "inpaint_only",
            "stable",
            ["tennis", "bmx-trees", "koala", "bear", "camel"],
            ROOT / "inpainting/run_controlnet_ablation_5seq.py",
            None,
            "python part3/inpainting/run_controlnet_ablation_5seq.py --sequences {sequence}",
            " ControlNet  ProPainter ",
            "",
            " ControlNet ",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "local_masks",
                "source_output_dir": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_pure" / seq,
                "inpaint_out": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_pure" / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "controlnet" / "ablation_5seq" / seq / "propainter_pure" / seq / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "controlnet_hybrid_propainter_fixed_mask",
            "ControlNet hybrid",
            "direction_c",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "koala", "bear", "camel"],
            ROOT / "inpainting/run_controlnet_ablation_5seq.py",
            None,
            "python part3/inpainting/run_controlnet_ablation_5seq.py --sequences {sequence}",
            " hybrid frames  ProPainter",
            "",
            " ProPainter ",
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
            "ControlNet hybrid + temporal consistency",
            "direction_c",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "koala", "bear", "camel"],
            ROOT / "inpainting/run_controlnet_ablation_5seq.py",
            None,
            "python part3/inpainting/run_controlnet_ablation_5seq.py --sequences {sequence}",
            " hybrid ",
            "",
            " ProPainter",
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
            " SDXL interval10 + ProPainter",
            "direction_c",
            "inpaint_only",
            "legacy",
            ["tennis", "bmx-trees"],
            ROOT / "inpainting/run_keyframe_sdxl_propainter.py",
            None,
            "",
            " SDXL ",
            " kf5 ",
            "",
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
            "Koala VOID ",
            "direction_c",
            "full_pipeline",
            "exploratory",
            ["koala"],
            ROOT / "inpainting/run_void_koala.py",
            None,
            "",
            " mask ",
            "",
            "",
            lambda seq: {
                "mask_frames": ROOT / "outputs" / "sam3_multiobj" / "masks_final" / seq,
                "source_output_dir": ROOT / "outputs" / "koala_diffusion" / "void" / "final_output",
                "inpaint_out": ROOT / "outputs" / "koala_diffusion" / "void" / "final_output" / "inpaint_out.mp4",
                "masked_in": ROOT / "outputs" / "koala_diffusion" / "void" / "final_output" / "masked_in.mp4",
            },
        ),
        ExperimentDef(
            "a_plus_b_best_full_pipeline",
            "A+B Best ",
            "fusion",
            "full_pipeline",
            "stable",
            davis6,
            ROOT / "pipeline/run_a3_best_pipeline.py",
            None,
            "python part3/pipeline/run_a3_best_pipeline.py --sequences {sequence}",
            " mask  ProPainter",
            "",
            "",
            lambda seq: {
                "source_output_dir": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter",
                "inpaint_out": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / seq / "inpaint_out.mp4",
                "masked_in": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / seq / "masked_in.mp4",
                "masks_dilated": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / "masks_dilated",
                "log": ROOT / "results_v2" / seq / "a_plus_b_best" / "propainter" / f"propainter_{seq}.log",
            },
        ),
        # ── Direction C: DiffuEraser GT mask v1 ( inpaint_only ) ──
        ExperimentDef(
            "diffueraser_gtmask_v1",
            "DiffuEraser GT mask v1inpaint_only",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "car-shadow", "blackswan", "horsejump-low"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/prepare_diffueraser_inputs.py "
                "--seq {sequence} --version v1\n"
                "conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py "
                "--seq {sequence} --version v1"
            ),
            (
                " DAVIS GT mask  DiffuEraser diffusion inpainting "
                "pure ProPainter DiffuEraser  BrushNet+AnimateDiff "
                " ProPainter  prior  diffusion "
                " tennis70"
            ),
            (
                "1. masked_in.mp4GT mask \n"
                "2. inpaint_out.mp4 pure_propainter_gtmask ——"
                " hallucination mask \n"
                "3. PSNR_proxy / SSIM pure_propainter_gtmask"
            ),
            (
                "v1 PSNR_proxy=31.42 vs  34.98-3.56dB"
                "SSIM=0.897 vs 0.930-0.033v2/v3 "
                "DiffuEraser soft-blending  mask "
                " full_pipeline exploratory"
                "PSNR_synthetic (mask ) ≈ ProPaintermask "
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
                " +  pure_propainter_gtmask + "
                "PSNR_proxy  SSIM "
            ),
            next_decision=(
                "v1/v2/v3  3.5-4.5 dB"
                "DiffuEraser  full_pipeline exploratory"
                " soft-blending  neighbor_length"
            ),
        ),
        # ── Direction C: DiffuEraser GT mask v3 (max_img_size=640, dilation=4) ──
        ExperimentDef(
            "diffueraser_gtmask_v3",
            "DiffuEraser GT mask v3max_img_size=640, dilation=4",
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
                "v3 max_img_size=640 640x360+  dilation=4"
                "v1/v2  dilation <0.1dB"
                "v3 -4.56dB vs "
                " exploratory"
            ),
            (
                "v1/v2/v3 PSNR_proxy  30-31dB 34.98dB"
                "DiffuEraser soft-blending  feathering  mask "
                ""
            ),
            (
                "DiffuEraser tennis v1: -3.56dB, v2: -3.66dB, v3: -4.56dB"
                "soft-blended diffusion  mask "
                " full_pipeline exploratory "
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
            stage_gate="N/A — ",
            next_decision=(
                "DiffuEraser  full_pipeline"
                " exploratory blending  neighbor_length"
            ),
            failure_reason=(
                "v3 (dilation=4, max_img_size=640): PSNR_proxy=30.42 (-4.56dB vs  34.98)"
                "SSIM=0.871 (vs 0.930)"
            ),
        ),
        # ── Direction C: DiffuEraser v4 () ──
        ExperimentDef(
            "diffueraser_gtmask_v4",
            "DiffuEraser GT mask v4hard blend",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "car-shadow", "blackswan", "horsejump-low"],
            ROOT / "inpainting/apply_hard_blend.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/apply_hard_blend.py "
                "--base_version v1 --out_version v4 --sequence tennis"
            ),
            (
                " v1 inpaint_out mask  DiffuEraser "
                "mask  DAVIS  JPEG  MP4 "
                " soft-blending "
                " DiffuEraser  v1 "
            ),
            (
                "1. inpaint_out.mp4 mask \n"
                "2. PSNR_proxy mask \n"
                "3. PSNR_synthetic v1 mask \n"
                "4. mask "
            ),
            (
                "v4  DAVIS JPEG  mask "
                "PSNR_proxy=36.28+1.40dB vs  34.88 PSNR_proxy "
                "PSNR_synthetic=32.88mask  ProPainter prior "
                "v4 "
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
                "v4  PSNR_proxy 36.28 ≥ 34.88"
                " bmx-trees  car-shadow "
            ),
            failure_reason="",
        ),
        # ── Direction C: DiffuEraser v5 ( neighbor window) ──
        ExperimentDef(
            "diffueraser_gtmask_v5",
            "DiffuEraser GT mask v5neighbor_length=20, subvideo=70",
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
                " ProPainter prior neighbor_length=20 10"
                "subvideo_length=70 50 feathering  hallucination"
                " prior  diffusion  soft-blending "
            ),
            (
                "1. PSNR_proxy vs v1 prior  mask \n"
                "2. PSNR_syntheticmask \n"
                "3. "
            ),
            (
                "v5 PSNR_proxy=31.32 v1 31.32 PSNR_synthetic=8.42"
                " PSNR_proxy soft-blending "
                " exploratory"
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
            next_decision=" exploratorysoft-blending  v4 ",
            failure_reason=(
                "PSNR_proxy=31.32-3.56dB vs  34.88"
                " neighbor_length/subvideo  mask "
            ),
        ),
        # ── Direction C: DiffuEraser v6 ( max_img_size=1280) ──
        ExperimentDef(
            "diffueraser_gtmask_v6",
            "DiffuEraser GT mask v6max_img_size=1280",
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
                " max_img_size=1280 960tennis  854x480  downsample"
                " diffusion v3 640"
                " artifacts"
            ),
            (
                "1. PSNR_proxy vs v1 mask \n"
                "2. PSNR_syntheticmask \n"
                "3. A6000 49GB  1280px "
            ),
            (
                "v6 PSNR_proxy=31.32 v1 PSNR_synthetic=8.43"
                " PSNR_proxy soft-blending "
                " exploratory"
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
            next_decision=" exploratorysoft-blending ",
            failure_reason=(
                "PSNR_proxy=31.32-3.56dB vs  34.88"
                "max_img_size=1280  mask "
            ),
        ),
        # ── Direction C: DiffuEraser v7 ( mask, dilate_px=0) ──
        ExperimentDef(
            "diffueraser_gtmask_v7",
            "DiffuEraser GT mask v7 mask, dilate_px=0, dilation_iter=0",
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
                " mask prepare  dilate_px=0 GT mask"
                "DiffuEraser  mask_dilation_iter=0 GT mask "
                " mask  soft-blending  mask  PSNR_proxy"
            ),
            (
                "1. PSNR_proxy vs v1dilate=8px mask  mask  PSNR\n"
                "2. PSNR_syntheticmask \n"
                "3. GT mask "
            ),
            (
                "v7 PSNR_proxy=32.46-2.43dB vs  34.88PSNR_synthetic=13.10"
                " mask vs v1 -3.56dB"
                "PSNR_synthetic 13.10  v1  8.42  mask  mask "
                " exploratory"
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
                "32.46 < 34.88"
                " exploratory v4 "
            ),
            failure_reason=(
                "PSNR_proxy=32.46-2.43dB vs  34.88"
                " mask PSNR_synthetic=13.10  mask "
            ),
        ),
        # ── Direction C: DiffuEraser v8 ( ref_stride=5) ──
        ExperimentDef(
            "diffueraser_gtmask_v8",
            "DiffuEraser GT mask v8ref_stride=5",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "blackswan", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/run_diffueraser_gtmask.py",
            None,
            (
                "cd /data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser && "
                "conda run -n diffueraser_env python3 run_diffueraser.py "
                "--input_video .../diffueraser_gtmask_v8/input_video.mp4 "
                "--input_mask .../diffueraser_gtmask_v8/input_mask.mp4 "
                "--video_length <N> --save_path .../diffueraser_gtmask_v8 "
                "--mask_dilation_iter 0 --ref_stride 5 --max_img_size 960"
            ),
            (
                "ref_stride=5 10 5 "
                " ProPainter prior "
                " diffusion  prior  hallucination  blending "
            ),
            (
                "1. PSNR_proxy vs v1 mask \n"
                "2. PSNR_synthetic\n"
                "3. ref_stride=5  10  2x"
            ),
            (
                "v8 PSNR_proxy=31.31 v1 PSNR_synthetic=8.41"
                " PSNR_proxysoft-blending  ref_stride "
                " exploratory8 "
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
                " exploratory8  DiffuEraser "
                " v4 soft-blending "
            ),
            failure_reason=(
                "PSNR_proxy=31.31-3.57dB vs  34.88"
                "ref_stride=5  mask  soft-blending "
            ),
        ),
        # ── Direction C: DiffuEraser v9 (v8 + correct hard-blend, ) ──
        ExperimentDef(
            "diffueraser_gtmask_v9",
            "DiffuEraser GT mask v9v8 +  hard-blend",
            "DiffuEraser",
            "inpaint_only",
            "exploratory",
            ["tennis", "bmx-trees", "blackswan", "horsejump-low", "car-shadow"],
            ROOT / "inpainting/apply_hard_blend.py",
            None,
            (
                "conda run -n diffueraser_env python3 part3/inpainting/apply_hard_blend.py "
                "--base_version v8 --out_version v9 "
                "--results_root .../results/<seq>/direction_c "
                "--sequence <seq> --feather 0"
            ),
            (
                "v9 = v8 +  hard-blendapply_hard_blend.py  >127  >0"
                " mask_frames/ 0/255 binary"
                " v4  mask  bug v4  >127  DAVIS 38/75"
                "v9  mask  DAVIS  JPEG  soft-blending "
            ),
            (
                "1. PSNR_proxy v8mask \n"
                "2. PSNR_synth~8–16 remove\n"
                "3.  soft-blending "
            ),
            (
                "v9 DAVIS5 meanPSNR_proxy=35.35 > pure ProPainter 34.95+0.40 dB"
                "PSNR_synth  ≤ 17 remove Direction C "
            ),
            lambda seq: {
                "inpaint_out": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v9" / "inpaint_out.mp4",
                "input_video": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v9" / "input_video.mp4",
                "input_mask":  ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v9" / "input_mask.mp4",
                "masked_in":   ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v9" / "masked_in.mp4",
                "mask_frames": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v9" / "mask_frames",
                "manifest":    ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v9" / "run_manifest.json",
            },
            version="v9",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            stage_gate="PSNR_proxy>=34.88 AND PSNR_synth<=20remove ",
            next_decision=(
                "✅ v9 DAVIS5 mean PSNR_proxy=35.35 ProPainter remove "
                "Direction C  v9"
            ),
            failure_reason="",
        ),
        # ── Direction C: DiffuEraser smoke test () ──
        ExperimentDef(
            "diffueraser_smoke_v1",
            "DiffuEraser  smoke test",
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
            " smoke test DiffuEraser  inpaint_out.mp4",
            "",
            " smoke test ",
            lambda seq: {
                "manifest": ROOT / "results" / seq / "direction_c" / "diffueraser_gtmask_v1" / "run_manifest.json",
            },
            version="v1",
            mask_protocol="davis_gt",
            baseline="pure_propainter_gtmask",
            next_decision="smoke  v1 ",
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
        "baseline": "/",
        "direction_a": "Direction AMask Upgrade",
        "direction_b": "Direction BBetter Mask",
        "direction_c": "Direction CInpainting",
        "fusion": "A+B ",
    }
    return mapping.get(exp.family, exp.family)


def build_root_readme(registry_rows: List[Dict[str, Any]]) -> str:
    stable = sum(1 for r in registry_rows if r["audit_status"] in {"stable", "reference", "promising"})
    failed = sum(1 for r in registry_rows if "failed" in r["audit_status"])
    return f"""# Part3 Deliverables



##

1.  sequence  `tennis/`
2.  `00_readme.md`
3.  `masked_in.mp4`
4.  `inpaint_out.mp4`
5.  `experiment_card.md`  `metrics.json`

##

- `mask_only`:  mask
- `inpaint_only`:  mask
- `full_pipeline`: mask + inpaint

##

- `reference`:
- `stable`:
- `promising`:
- `exploratory`:
- `legacy`:
- `superseded`:
- `partial_or_failed` / `failed`:

##

- : `{len(registry_rows)}`
- /: `{stable}`
- : `{failed}`

`../DELIVERABLES_GUIDE_CN.md`
"""


def build_sequence_readme(seq: str, rows: List[Dict[str, Any]]) -> str:
    meta = SEQUENCES.get(seq, {})
    lines = [
        f"# {seq}",
        "",
        f"- {meta.get('difficulty', '')}",
        "-  `masked_in.mp4`  mask `inpaint_out.mp4` ",
        "- `mask_only` `inpaint_only`  mask `full_pipeline` ",
        "",
        "## ",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['method_id']}` | {row['comparison_type']} | {row['audit_status']} | {row['readable_name']}")
        lines.append(f"  {row['plain_explanation']}")
    return "\n".join(lines) + "\n"


def build_card(seq: str, exp: ExperimentDef, metrics: Dict[str, Any], existing: Dict[str, str]) -> str:
    evidence_lines = []
    for key, value in existing.items():
        evidence_lines.append(f"- `{key}`: `{value}`")
    if not evidence_lines:
        evidence_lines.append("- ")
    metrics_lines = []
    for key, value in metrics.items():
        if key == "source_file":
            continue
        metrics_lines.append(f"- `{key}`: `{value}`")
    if not metrics_lines:
        metrics_lines.append("- ")

    # ── schema v2 ─────────────────────────────────────────────
    mask_src_answer = exp.mask_protocol if exp.mask_protocol else " mask_protocol "
    four_q = f"""### Q1
{exp.plain_explanation}

### Q2   mask
- **mask_protocol**: `{mask_src_answer}`
- DAVIS `DAVIS annotation / GT mask` `/home/jli657/shared_data/project3/DAVIS/Annotations/480p/<seq>`
-  DAVIS  wild video  `mask_protocol`

### Q3
-  `family``{exp.family}`
- ProPainterPPLaMaSDXLDiffuEraserVOIDControlNet

### Q4   mask  inpainting
-  `masked_in.mp4`
-  `inpaint_out.mp4` / hallucination
- `PSNR / SSIM / JM / JR / F` """

    # ── Version Historyschema v2 ─────────────────────────────────────
    baseline_note = exp.baseline if exp.baseline else ""
    failure_note = f"\n- **failure_reason**: {exp.failure_reason}" if exp.failure_reason else ""
    next_note = f"\n- **next_decision**: {exp.next_decision}" if exp.next_decision else ""
    stage_note = f"\n- **stage_gate**: {exp.stage_gate}" if exp.stage_gate else ""
    version_history = f"""|  |  |
|---|---|
| version | `{exp.version}` |
| based_on | `{baseline_note}` |
| changes_from_previous | "" `exp_id`  |
| motivation | {exp.what_to_check} |
| expected_gain |  `plain_explanation` |
| actual_result | {exp.current_takeaway}{failure_note} |
| next_decision | {exp.next_decision if exp.next_decision else ""} |{stage_note}"""

    return f"""# {exp.readable_name}

##

{exp.plain_explanation}

##

- `{exp.comparison_type}`
- {family_label(exp)}
- `{exp.audit_status}`
- `{exp.version}`
- mask `{exp.mask_protocol if exp.mask_protocol else ""}`
- `{exp.baseline if exp.baseline else ""}`

##

{four_q}

##

{exp.what_to_check}

##

{exp.current_takeaway}

{(f"> ⚠  / {exp.failure_reason}" + chr(10) + chr(10)) if exp.failure_reason else ""}## Version History /

{version_history}

##

{chr(10).join(evidence_lines)}

##

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
        current_takeaway=data.get("current_takeaway", ""),
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
        and "" in _root_readme.read_text(encoding="utf-8")
    )
    if not _skip_root_readme:
        write_text(_root_readme, build_root_readme(registry_rows))
    else:
        print(f"[skip] {_root_readme} — ")
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
