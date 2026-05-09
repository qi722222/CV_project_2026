from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class SequenceResult:
    sequence: str
    status: str
    masks_dir: str
    propainter_output_dir: str
    inpaint_video: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part3 rebuild: GDINO+VLM -> SAM3 -> ProPainter")
    p.add_argument(
        "--mode",
        choices=["davis5", "wild", "all"],
        default="all",
        help="Which subset to run",
    )
    p.add_argument(
        "--davis_config",
        default="part3/configs/sam3_rebuild_mainline_davis5.yaml",
        help="Config for DAVIS5 mainline",
    )
    p.add_argument(
        "--wild_config",
        default="part3/configs/sam3_rebuild_mainline_wild.yaml",
        help="Config for wild video mainline",
    )
    p.add_argument(
        "--gdino_python",
        default="/data2/jli657/envs/sam3_env/bin/python",
        help="Python for gdino/sam3 mainline",
    )
    p.add_argument(
        "--propainter_python",
        default="/data2/jli657/envs/propainter_env/bin/python",
        help="Python for ProPainter stage",
    )
    p.add_argument(
        "--propainter_dir",
        default="/home/jli657/my_storage2_1T/ProPainter",
        help="ProPainter repository",
    )
    p.add_argument(
        "--output_root",
        default="part3/outputs/sam3_rebuild",
        help="Root output directory",
    )
    p.add_argument(
        "--sequences",
        default="",
        help="Optional comma-separated sequence filter",
    )
    p.add_argument(
        "--stage",
        choices=["stage1", "stage2"],
        default="stage2",
        help="Mainline stage to run for mask generation",
    )
    p.add_argument("--skip_gdino", action="store_true", help="Skip mask generation stage")
    p.add_argument("--skip_propainter", action="store_true", help="Skip ProPainter stage")
    p.add_argument("--neighbor_length", type=int, default=10, help="ProPainter neighbor_length")
    p.add_argument("--ref_stride", type=int, default=10, help="ProPainter ref_stride")
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print("[cmd]", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(cwd), env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"command failed ({ret.returncode}): {' '.join(cmd)}")


def read_policy_sequences(config_path: Path) -> List[str]:
    cfg = load_yaml(config_path)
    pol = load_yaml(Path(cfg["policy_path"]))
    return [str(x["sequence_name"]) for x in pol.get("sequences", [])]


def pick_video_root(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    return Path(cfg["paths"]["davis_jpeg_root"]).expanduser().resolve()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    env_common = os.environ.copy()
    env_common["HF_ENDPOINT"] = "https://hf-mirror.com"
    env_common["HF_HOME"] = "/data3/jli657/hf_cache"
    env_common["HUGGINGFACE_HUB_CACHE"] = "/data3/jli657/hf_cache"

    jobs: List[Dict[str, str]] = []
    if args.mode in {"davis5", "all"}:
        cfg = Path(args.davis_config).resolve()
        video_root = pick_video_root(cfg)
        for seq in read_policy_sequences(cfg):
            jobs.append({"group": "davis5", "sequence": seq, "config": str(cfg), "video_root": str(video_root)})
    if args.mode in {"wild", "all"}:
        cfg = Path(args.wild_config).resolve()
        video_root = pick_video_root(cfg)
        for seq in read_policy_sequences(cfg):
            jobs.append({"group": "wild", "sequence": seq, "config": str(cfg), "video_root": str(video_root)})

    if not jobs:
        raise RuntimeError("no sequences selected")
    seq_filter = {s.strip() for s in args.sequences.split(",") if s.strip()}
    if seq_filter:
        jobs = [j for j in jobs if j["sequence"] in seq_filter]
        if not jobs:
            raise RuntimeError(f"no sequence matched filter: {sorted(seq_filter)}")

    results: List[SequenceResult] = []
    for job in jobs:
        seq = job["sequence"]
        group = job["group"]
        cfg_path = Path(job["config"])
        video_dir = Path(job["video_root"]) / seq

        masks_dir = output_root / "masks" / group / seq
        propainter_root = output_root / "propainter" / group / seq
        inpaint_video = propainter_root / "inpaint_out.mp4"
        result = SequenceResult(
            sequence=seq,
            status="ok",
            masks_dir=str(masks_dir),
            propainter_output_dir=str(propainter_root),
            inpaint_video=str(inpaint_video),
        )

        try:
            if not video_dir.exists():
                raise FileNotFoundError(f"video frames dir missing: {video_dir}")

            if not args.skip_gdino:
                run_cmd(
                    [
                        args.gdino_python,
                        "part3/gdino_vlm/run_gdino_mainline.py",
                        "--config",
                        str(cfg_path),
                        "--sequence",
                        seq,
                        "--stage",
                        args.stage,
                        "--output",
                        str(masks_dir),
                    ],
                    cwd=project_root,
                    env=env_common,
                )

            if not args.skip_propainter:
                run_cmd(
                    [
                        args.propainter_python,
                        "part2/run_propainter.py",
                        "--video",
                        str(video_dir),
                        "--masks",
                        str(masks_dir),
                        "--output",
                        str(propainter_root.parent),
                        "--propainter_dir",
                        str(Path(args.propainter_dir).resolve()),
                        "--dilate_kernel",
                        "9",
                        "--resize_ratio",
                        "1.0",
                        "--neighbor_length",
                        str(args.neighbor_length),
                        "--ref_stride",
                        str(args.ref_stride),
                    ],
                    cwd=project_root,
                    env=env_common,
                )
        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
        results.append(result)

    manifest = {
        "mode": args.mode,
        "stage": args.stage,
        "output_root": str(output_root),
        "num_sequences": len(results),
        "num_failed": sum(1 for r in results if r.status != "ok"),
        "results": [asdict(r) for r in results],
    }
    manifest_path = output_root / "rebuild_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[save] {manifest_path}")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print("[warn] some sequences failed:")
        for r in failed:
            print(f"  - {r.sequence}: {r.error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
