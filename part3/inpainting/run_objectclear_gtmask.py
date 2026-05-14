"""
run_objectclear_gtmask.py — ObjectClear inpaint_only DAVIS GT mask


  1.  prepare_objectclear_inputs.py  imgs/ + masks/
  2.  ObjectClearPipeline  image-level object removal
  3.  raw ObjectClear  repair_frames/
  4.  hard-blend mask  ObjectClear mask  DAVIS
      inpaint-only
  5.  hard-blend  inpaint_out.mp4
  6.  run_manifest.json build_part3_deliverables.py


  - v1raw  hard-blend
  - v2+ guidance_scale=2.5, num_inference_steps=50 + hard-blend


  conda run -n objectclear_env python3 part3/inpainting/run_objectclear_gtmask.py \\
      --seq tennis --version v2


  --guidance_scale  float  (default 2.5)
  --num_steps       int    (default 50) DDIM
  --hard_blend              Truemask  DAVIS
  --no_hard_blend           hard-blend v1
  --smoke_test              2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

OBJECTCLEAR_CODE = Path("/data3/jli657/project3/part3/ObjectClear_workspace/ObjectClear_space")
RESULTS_ROOT     = Path("/data3/jli657/project3/part3/results")
DAVIS_FRAMES     = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_MASKS      = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
PART3_ROOT       = Path("/home/jli657/my_storage2_1T/project3/part3")
HF_CACHE         = Path("/data3/jli657/hf_cache")
BASELINE         = "pure_propainter_gtmask"

# Make sure ObjectClear space code is importable
if str(OBJECTCLEAR_CODE) not in sys.path:
    sys.path.insert(0, str(OBJECTCLEAR_CODE))


def load_sorted(d: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts], key=lambda p: p.stem)


def write_video(out_path: Path, frames: list[np.ndarray], fps: float = 24.0) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()


def resize_by_short_side(img: Image.Image, short_side: int, resample=Image.BICUBIC) -> Image.Image:
    """Resize so that the short side equals short_side, then round both dims to multiples of 64.
    SDXL attention maps require resolution divisible by 64 for safe reshaping."""
    w, h = img.size
    if w < h:
        new_w = short_side
        new_h = int(h * short_side / w)
    else:
        new_h = short_side
        new_w = int(w * short_side / h)
    # Round to multiples of 64 (SDXL attention map requirement)
    new_w = max(64, (new_w // 64) * 64)
    new_h = max(64, (new_h // 64) * 64)
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), resample)


def load_pipeline(device: torch.device, args) -> object:
    from pipeline_objectclear import ObjectClearPipeline

    print(f"[objectclear] Loading model from HF (cache: {HF_CACHE}) ...")
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=str(HF_CACHE),
        apply_attention_guided_fusion=True,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[objectclear] Model loaded OK")
    return pipe


def infer_frame(
    pipe,
    img_pil: Image.Image,
    mask_pil: Image.Image,
    guidance_scale: float,
    num_steps: int,
    generator: torch.Generator,
) -> Image.Image:
    orig_w, orig_h = img_pil.size
    img_resized  = resize_by_short_side(img_pil, 512, resample=Image.BICUBIC)
    mask_resized = resize_by_short_side(mask_pil, 512, resample=Image.NEAREST)
    w, h = img_resized.size
    result = pipe(
        prompt="remove the instance of object",
        image=img_resized.convert("RGB"),
        mask_image=mask_resized.convert("RGB"),
        generator=generator,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=h,
        width=w,
    )
    out_pil = result.images[0]
    # Restore original resolution
    out_pil = out_pil.resize((orig_w, orig_h), Image.BICUBIC)
    return out_pil


def apply_hard_blend(
    repair_frames: list[np.ndarray],
    orig_paths: list[Path],
    mask_paths: list[Path],
) -> list[np.ndarray]:
    """Replace pixels outside the binary mask with the original DAVIS frame.
    This ensures inpaint_onlymask
    """
    blended = []
    for i, (rep, op, mp) in enumerate(zip(repair_frames, orig_paths, mask_paths)):
        orig = cv2.imread(str(op))
        mask_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if orig is None or mask_raw is None:
            blended.append(rep)
            continue
        if mask_raw.ndim == 3:
            mask_raw = mask_raw.max(axis=2)  # handle palette PNGs (any channel may carry annotation)
        binary = (mask_raw > 0).astype(np.uint8)  # 1=inpaint region

        # Align sizes to original
        h, w = orig.shape[:2]
        rep_r = cv2.resize(rep, (w, h), interpolation=cv2.INTER_LINEAR) if rep.shape[:2] != (h, w) else rep
        m = binary[:, :, np.newaxis].astype(np.float32)
        result = (m * rep_r.astype(np.float32) + (1.0 - m) * orig.astype(np.float32)).clip(0, 255).astype(np.uint8)
        blended.append(result)
    return blended


def write_manifest(
    out_dir: Path, seq: str, version: str,
    n_frames: int, elapsed: float, rc: int,
    guidance_scale: float, num_steps: int,
    hard_blend: bool = True,
) -> None:
    status = "exploratory" if rc == 0 else "partial_or_failed"
    manifest = {
        "exp_id": f"objectclear_gtmask_{version}",
        "readable_name": f"ObjectClear GT-mask {version} (guidance={guidance_scale}, steps={num_steps})",
        "sequence": seq,
        "family": "ObjectClear",
        "comparison_type": "inpaint_only",
        "direction": "C",
        "version": version,
        "mask_protocol": "davis_gt",
        "baseline": BASELINE,
        "audit_status": status,
        "stage_gate": "PSNR_proxy >= ProPainter",
        "next_decision": (
            " → "
            " →  v2guidance_scale/num_steps"
            " 2-3  →  exploratory/failed"
        ),
        "failure_reason": "" if rc == 0 else f"inference returned code {rc}",
        "parameters": {
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_steps,
            "model": "jixin0101/ObjectClear (fp16)",
            "inference_mode": "per-frame image inpainting",
        },
        "script": str(PART3_ROOT / "inpainting" / "run_objectclear_gtmask.py"),
        "command": (
            f"conda run -n objectclear_env python3 part3/inpainting/run_objectclear_gtmask.py "
            f"--seq {seq} --version {version} "
            f"--guidance_scale {guidance_scale} --num_steps {num_steps}"
        ),
        "output_dir": str(out_dir),
        "inpaint_out": str(out_dir / "inpaint_out.mp4"),
        "masked_in": str(out_dir / "masked_in.mp4"),
        "mask_frames_dir": str(out_dir / "mask_frames"),
        "repair_frames_dir": str(out_dir / "repair_frames"),
        "n_frames": n_frames,
        "elapsed_sec": round(elapsed, 1),
        "hard_blend": hard_blend,
        "plain_explanation": (
            f" ObjectClearCVPR 2026 image-level  +  remove"
            f"guidance_scale={guidance_scale}num_steps={num_steps}"
            + ("hard-blend mask  DAVIS  inpaint-only " if hard_blend
               else "raw  hard-blend v1 ")
        ),
        "what_to_check": (
            "1. masked_in.mp4 mask \n"
            "2. inpaint_out.mp4\n"
            "3. repair_frames/raw ObjectClear  hard-blend\n"
            "4. PSNR_proxy vs pure_propainter_gtmask \n"
            "5. PSNR_synth = remove "
        ),
        "current_takeaway": "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="tennis")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard_blend", action="store_true", default=True,
                        help="Replace mask-outside pixels with DAVIS original frames (default: on)")
    parser.add_argument("--no_hard_blend", dest="hard_blend", action="store_false",
                        help="Disable hard-blend (raw ObjectClear output, v1 legacy behavior)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run on first 2 frames only to verify the pipeline works")
    args = parser.parse_args()

    out_dir       = RESULTS_ROOT / args.seq / "direction_c" / f"objectclear_gtmask_{args.version}"
    imgs_dir      = out_dir / "imgs"
    masks_dir     = out_dir / "masks"
    repair_dir    = out_dir / "repair_frames"
    repair_dir.mkdir(parents=True, exist_ok=True)

    # Sanity checks
    if not imgs_dir.exists() or not masks_dir.exists():
        print(f"[ERROR] Input dirs not found. Run prepare first:")
        print(f"  conda run -n objectclear_env python3 part3/inpainting/prepare_objectclear_inputs.py "
              f"--seq {args.seq} --version {args.version}")
        sys.exit(1)

    img_paths  = load_sorted(imgs_dir)
    mask_paths = load_sorted(masks_dir)
    if not img_paths:
        print(f"[ERROR] No images in {imgs_dir}")
        sys.exit(1)

    n_frames = len(img_paths)
    if args.smoke_test:
        print(f"[smoke] Running on first 2 frames only")
        img_paths  = img_paths[:2]
        mask_paths = mask_paths[:2]
        n_frames   = 2

    print(f"[objectclear] seq={args.seq}  version={args.version}")
    print(f"  frames:         {n_frames}")
    print(f"  guidance_scale: {args.guidance_scale}")
    print(f"  num_steps:      {args.num_steps}")
    print(f"  output:         {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    pipe = load_pipeline(device, args)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    t0 = time.time()
    repaired: list[np.ndarray] = []
    orig_frames: list[np.ndarray] = []

    for i, (ip, mp) in enumerate(zip(img_paths, mask_paths)):
        # Load
        img_bgr  = cv2.imread(str(ip))
        mask_bin = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask_bin is None:
            print(f"[WARN] Cannot read {ip} or {mp}, skipping")
            continue

        img_pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_bin)

        # Inference
        out_pil = infer_frame(pipe, img_pil, mask_pil,
                              args.guidance_scale, args.num_steps, generator)

        out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
        orig_frames.append(img_bgr)
        repaired.append(out_bgr)

        # Save repair frame
        cv2.imwrite(str(repair_dir / ip.name.replace(".jpg", ".png")), out_bgr)

        if (i + 1) % 10 == 0 or (i + 1) == n_frames:
            elapsed_so_far = time.time() - t0
            fps_so_far = (i + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
            eta = (n_frames - i - 1) / fps_so_far if fps_so_far > 0 else 0
            print(f"  [{i+1}/{n_frames}]  {elapsed_so_far:.0f}s elapsed  "
                  f"{fps_so_far:.2f} fps  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[objectclear] done in {elapsed:.1f}s  ({elapsed/n_frames:.1f}s/frame)")

    inpaint_out = out_dir / "inpaint_out.mp4"

    # Hard-blend: replace mask-outside pixels with original DAVIS frames
    if args.hard_blend:
        orig_paths_all = sorted(
            [p for p in (DAVIS_FRAMES / args.seq).iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            key=lambda p: p.stem
        )[:n_frames]
        gt_mask_paths = sorted(
            [p for p in (DAVIS_MASKS / args.seq).iterdir() if p.suffix.lower() == ".png"],
            key=lambda p: p.stem
        )[:n_frames]
        print(f"[hard_blend] applying mask-outside replacement ({len(orig_paths_all)} frames)...")
        repaired = apply_hard_blend(repaired, orig_paths_all, gt_mask_paths)
        print("[hard_blend] done")

    # Assemble inpaint_out.mp4 (after hard-blend if enabled)
    write_video(inpaint_out, repaired, fps=args.fps)
    print(f"[video] inpaint_out.mp4 → {inpaint_out}")

    # Write manifest
    write_manifest(out_dir, args.seq, args.version, n_frames, elapsed,
                   rc=0, guidance_scale=args.guidance_scale, num_steps=args.num_steps,
                   hard_blend=args.hard_blend)

    if args.smoke_test:
        print("\n[smoke] PASS — pipeline works, 2 frames written")
        print(f"  repair_frames/ → {repair_dir}")
        print(f"  inpaint_out.mp4 → {inpaint_out}")
        return

    blend_tag = "hard-blend (mask-outside=DAVIS original)" if args.hard_blend else "raw full-frame output"
    print(f"\n[OK] ObjectClear inference complete ({blend_tag})")
    print(f"  inpaint_out.mp4 → {inpaint_out}")
    print(f"  repair_frames/  → {repair_dir}  (raw ObjectClear per-frame)")
    print(f"\nNext steps:")
    print(f"  1. Watch masked_in.mp4 and inpaint_out.mp4")
    print(f"  2. Run evaluation:")
    print(f"     conda run -n controlnet_env python3 part3/eval/evaluate_all.py --seqs {args.seq}")
    print(f"  3. Build deliverable:")
    print(f"     python3 part3/reporting/build_part3_deliverables.py "
          f"--manifest {out_dir}/run_manifest.json")


if __name__ == "__main__":
    main()
