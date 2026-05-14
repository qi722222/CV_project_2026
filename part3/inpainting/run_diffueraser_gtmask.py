"""
run_diffueraser_gtmask.py — DiffuEraser inpaint_only DAVIS GT mask


  1.  prepare_diffueraser_inputs.py  input_video.mp4 + input_mask.mp4
  2.  DiffuEraser  inpaint_out.mp4
  3.  run_manifest.json build_part3_deliverables.py
  4.  PSNR/SSIM


  - diffueraser_gtmask_v1
  - diffueraser_gtmask_v2 / v3


  conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py \\
      --seq tennis --version v1

  # Smoke test
  conda run -n diffueraser_env python3 part3/inpainting/run_diffueraser_gtmask.py \\
      --seq tennis --version v1 --smoke_test

DiffuEraser  --de_args
  --guidance_scale    (default: 2.5)
  --n_timesteps       (default: 2)    DDIM
  --mask_dilation     (default: 8)     mask  prepare
  --fps               (default: 2)    DiffuEraser  2-4
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
            " pure_propainter_gtmask"
            "PSNR_proxy  SSIM "
        ),
        "next_decision": (
            " →  bmx-trees  car-shadow"
            " →  v2 guidance_scale / n_timesteps"
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
            " DAVIS GT mask  mask  DiffuEraser diffusion inpainting"
            " pure ProPainter  DiffuEraser "
        ),
        "what_to_check": (
            "1. masked_in.mp4GT mask /\n"
            "2. inpaint_out.mp4 hallucination"
            " mask \n"
            "3. PSNR_proxy / SSIM pure_propainter_gtmask "
        ),
        "current_takeaway": "",
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
    parser.add_argument("--mask_dilation_iter", type=int, default=0,
                        help="DiffuEraser internal mask dilation iterations (0=rely on external dilate_px)")
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
    # Auto-detect frame count from mask_frames dir
    mask_frames_dir = out_dir / "mask_frames"
    if args.max_frames > 0:
        video_length = args.max_frames
    elif mask_frames_dir.exists():
        n_mask = len([p for p in mask_frames_dir.iterdir() if p.suffix == ".png"])
        video_length = n_mask if n_mask > 0 else 70
    else:
        video_length = 70

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
        "--mask_dilation_iter", str(args.mask_dilation_iter),  # configurable; default=0 (already dilated externally)
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
