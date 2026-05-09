"""
run_diffueraser_gtmask.py — DiffuEraser inpaint_only 运行脚本（DAVIS GT mask）

工作流程：
  1. 读取 prepare_diffueraser_inputs.py 生成的 input_video.mp4 + input_mask.mp4
  2. 调用 DiffuEraser 推理，输出 inpaint_out.mp4
  3. 写出 run_manifest.json（供 build_part3_deliverables.py 自动注册）
  4. 触发 PSNR/SSIM 评估

实验命名规范：
  - 首版：diffueraser_gtmask_v1
  - 调参版：diffueraser_gtmask_v2 / v3（必须新建目录，不覆盖旧版）

用法：
  conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py \\
      --seq tennis --version v1

  # Smoke test（仅检查能否运行，不要求完美结果）
  conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py \\
      --seq tennis --version v1 --smoke_test

DiffuEraser 关键参数（通过 --de_args 传递）：
  --guidance_scale    (default: 2.5)  弱条件强度，值越低越少保留原始内容
  --n_timesteps       (default: 2)    DDIM 步数，步数少则快
  --mask_dilation     (default: 8)    内部 mask 膨胀（已在 prepare 中做了外部膨胀）
  --fps               (default: 2)    DiffuEraser 处理帧率（建议 2-4，太高显存不足）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RESULTS_ROOT    = Path("/data3/jli657/project3/part3/results")
DIFFUERASER_DIR = Path("/data3/jli657/project3/part3/DiffuEraser_workspace/DiffuEraser")
DAVIS_FRAMES    = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
PART3_ROOT      = Path("/home/jli657/my_storage2_1T/project3/part3")
MANIFEST_SCHEMA = PART3_ROOT / "pipeline" / "run_manifest_schema.py"

BASELINE = "pure_propainter_gtmask"


def run_cmd(cmd: list[str], cwd: Path | None = None, env_name: str = "diffueraser_env") -> int:
    full_cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + cmd
    print(f"[run] {' '.join(str(c) for c in full_cmd)}")
    result = subprocess.run(full_cmd, cwd=str(cwd) if cwd else None)
    return result.returncode


def write_manifest(out_dir: Path, seq: str, version: str,
                   returncode: int, elapsed: float) -> None:
    status = "exploratory" if returncode == 0 else "partial_or_failed"
    manifest = {
        "exp_id": f"diffueraser_gtmask_{version}",
        "readable_name": f"DiffuEraser GT mask {version} (inpaint_only)",
        "sequence": seq,
        "family": "DiffuEraser",
        "comparison_type": "inpaint_only",
        "audit_status": status,
        "version": version,
        "mask_protocol": "davis_gt",
        "baseline": BASELINE,
        "stage_gate": (
            "工程稳定、视觉不明显劣于 pure_propainter_gtmask、"
            "PSNR_proxy 或 SSIM 至少一项接近或优于基线"
        ),
        "next_decision": (
            "过门槛 → 扩到 bmx-trees 和 car-shadow；"
            "未过 → 调参形成 v2（调 guidance_scale / n_timesteps）"
        ),
        "failure_reason": "" if returncode == 0 else f"inference returned code {returncode}",
        "script_path": str(PART3_ROOT / "inpainting" / "run_diffueraser_gtmask.py"),
        "command": (
            f"conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py "
            f"--seq {seq} --version {version}"
        ),
        "output_dir": str(out_dir),
        "inpaint_out": str(out_dir / "inpaint_out.mp4"),
        "masked_in": str(out_dir / "masked_in.mp4"),
        "mask_frames_dir": str(out_dir / "mask_frames"),
        "log_path": str(out_dir / "run.log"),
        "plain_explanation": (
            "以 DAVIS GT mask 固定 mask 输入，比较 DiffuEraser（视频 diffusion inpainting）"
            "与 pure ProPainter 的修复质量。首轮验证 DiffuEraser 的时间一致性和生成质量。"
        ),
        "what_to_check": (
            "1. masked_in.mp4：GT mask 覆盖是否正确（应完整覆盖运动员/目标）\n"
            "2. inpaint_out.mp4：修复区域是否自然、有无闪烁、背景 hallucination、"
            "非 mask 区域污染\n"
            "3. PSNR_proxy / SSIM：与 pure_propainter_gtmask 对比"
        ),
        "current_takeaway": "待推理完成后更新",
        "elapsed_sec": round(elapsed, 1),
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] written → {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="tennis")
    parser.add_argument("--version", default="v1",
                        help="Version tag; new versions must use new directory")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Only verify inputs exist and DiffuEraser can import; skip full inference")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--n_timesteps", type=int, default=2)
    parser.add_argument("--fps", type=int, default=2,
                        help="Temporal fps for DiffuEraser sliding window")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Clip length override (0 = use all frames in input_video)")
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / args.seq / "direction_c" / f"diffueraser_gtmask_{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    input_video = out_dir / "input_video.mp4"
    input_mask  = out_dir / "input_mask.mp4"

    # Check inputs
    if not input_video.exists() or not input_mask.exists():
        print(f"[ERROR] Missing inputs in {out_dir}")
        print("  Run first: conda run -n diffueraser_env python3 "
              "part3/inpainting/prepare_diffueraser_inputs.py "
              f"--seq {args.seq} --version {args.version}")
        sys.exit(1)

    if not DIFFUERASER_DIR.exists():
        print(f"[ERROR] DiffuEraser repo not found at {DIFFUERASER_DIR}")
        print("  Run first: bash part3/inpainting/setup_diffueraser.sh")
        sys.exit(1)

    print(f"[diffueraser] seq={args.seq}  version={args.version}")
    print(f"  input_video : {input_video}")
    print(f"  input_mask  : {input_mask}")
    print(f"  output_dir  : {out_dir}")

    if args.smoke_test:
        # Smoke test: just verify DiffuEraser can be imported
        rc = run_cmd([
            "python3", "-c",
            "import sys; sys.path.insert(0, str('"
            + str(DIFFUERASER_DIR) + "'));"
            "from pipeline_diffueraser import DiffuEraserPipeline; "
            "print('[smoke] DiffuEraser import OK')"
        ])
        print(f"[smoke] import test exit code: {rc}")
        write_manifest(out_dir, args.seq, args.version, rc, 0.0)
        if rc == 0:
            print("[smoke] PASS — environment is ready")
        else:
            print("[smoke] FAIL — check environment; see setup_diffueraser.sh")
        return

    # DiffuEraser uses run_diffueraser.py
    # Output goes to --save_path dir as "diffueraser_result.mp4"
    infer_script = DIFFUERASER_DIR / "run_diffueraser.py"
    if not infer_script.exists():
        print(f"[ERROR] Cannot find run_diffueraser.py in {DIFFUERASER_DIR}")
        sys.exit(1)

    log_path = out_dir / "run.log"
    inpaint_result = out_dir / "diffueraser_result.mp4"
    inpaint_out    = out_dir / "inpaint_out.mp4"  # final standardized name

    weights_root = DIFFUERASER_DIR / "weights"
    video_length = args.max_frames if args.max_frames > 0 else 70  # tennis has 70 frames

    cmd = [
        "python3", str(infer_script),
        "--input_video",       str(input_video),
        "--input_mask",        str(input_mask),
        "--video_length",      str(video_length),
        "--save_path",         str(out_dir),
        "--base_model_path",   str(weights_root / "stable-diffusion-v1-5"),
        "--vae_path",          str(weights_root / "sd-vae-ft-mse"),
        "--diffueraser_path",  str(weights_root / "diffuEraser"),
        "--propainter_model_dir", str(weights_root / "propainter"),
        "--mask_dilation_iter", "0",  # already dilated externally
    ]

    t0 = time.time()
    print(f"[diffueraser] starting inference... (log → {log_path})")
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            ["conda", "run", "-n", "diffueraser_env", "--no-capture-output"] + cmd,
            cwd=str(DIFFUERASER_DIR),
            stdout=log_f, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    rc = proc.returncode
    print(f"[diffueraser] done in {elapsed:.1f}s  exit_code={rc}")

    write_manifest(out_dir, args.seq, args.version, rc, elapsed)

    if rc != 0:
        print(f"[ERROR] Inference failed. Check log: {log_path}")
        print("  Fallback options:")
        print("  1. Try --max_frames 20  (shorter clip for debugging)")
        print("  2. Reduce --max_img_size, e.g. add to cmd: --max_img_size 480")
        print("  3. Check GPU memory with nvidia-smi")
        sys.exit(rc)

    # Rename DiffuEraser's output to standardized inpaint_out.mp4
    if inpaint_result.exists() and not inpaint_out.exists():
        inpaint_result.rename(inpaint_out)
        print(f"[rename] diffueraser_result.mp4 → inpaint_out.mp4")
    elif inpaint_result.exists():
        import shutil
        shutil.copy2(inpaint_result, inpaint_out)

    print(f"\n[OK] DiffuEraser inference complete")
    print(f"  inpaint_out.mp4 → {inpaint_out}")
    print(f"  priori.mp4      → {out_dir / 'priori.mp4'}  (ProPainter prior step)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\nNext steps:")
    print(f"  1. Watch masked_in.mp4 and inpaint_out.mp4")
    print(f"  2. Run evaluation:")
    print(f"     conda run -n controlnet_env python3 part3/eval/evaluate_all.py "
          f"--seqs {args.seq}")
    print(f"  3. Register to deliverables:")
    print(f"     python3 part3/reporting/build_part3_deliverables.py "
          f"--manifest {out_dir}/run_manifest.json")


if __name__ == "__main__":
    main()
