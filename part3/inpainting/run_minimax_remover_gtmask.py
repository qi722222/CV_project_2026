"""
run_minimax_remover_gtmask.py — MiniMax-Remover (GT mask )

:
  1.  DAVIS 480p  + GT annotation masks ( shadow-aware masks)
  2. Resize  480x832 (MiniMax )
  3. : < 81  padding; > 81  chunking + 10  overlap stitching
  4.  MiniMax-Remover (12 steps, iterations=6 mask dilation)
  5.  resize  DAVIS  (hard-blend )
  6. Hard-blend: mask  DAVIS  (,  ON)
  7.  inpaint_out.mp4 + run_manifest.json

:
  # GT mask (horsejump-low, 60 ,  pad  81)
  conda run -p /data3/jli657/envs/minimax_env python3 \\
    /home/jli657/my_storage2_1T/project3/part3/inpainting/run_minimax_remover_gtmask.py \\
    --seq horsejump-low --version v1

  # GT mask (car-shadow, 40 ,  pad  81)
  conda run -p /data3/jli657/envs/minimax_env python3 \\
    /home/jli657/my_storage2_1T/project3/part3/inpainting/run_minimax_remover_gtmask.py \\
    --seq car-shadow --version v1

  # Shadow-aware mask (car-shadow, step 2 use)
  conda run -p /data3/jli657/envs/minimax_env python3 \\
    /home/jli657/my_storage2_1T/project3/part3/inpainting/run_minimax_remover_gtmask.py \\
    --seq car-shadow --version v2 \\
    --mask_dir /data3/jli657/project3/part3/outputs/sam3_multiobj/masks_shadow_v3/car-shadow

: /data3/jli657/project3/part3/MiniMax_Remover_workspace/
  (vae/, transformer/, scheduler/, pipeline_minimax_remover.py, transformer_minimax_remover.py)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Paths ───────────────────────────────────────────────────────────────────
MINIMAX_WORKSPACE = Path("/data3/jli657/project3/part3/MiniMax_Remover_workspace")
RESULTS_ROOT      = Path("/data3/jli657/project3/part3/results")
DAVIS_FRAMES      = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT_MASKS    = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
PART3_ROOT        = Path("/home/jli657/my_storage2_1T/project3/part3")
DELIVERABLES_ROOT = PART3_ROOT / "part3_deliverables"
BASELINE          = "pure_propainter_gtmask"

# MiniMax-Remover fixed resolution and frame count
MINIMAX_H        = 480
MINIMAX_W        = 832
MINIMAX_N_FRAMES = 81


# ── File helpers ─────────────────────────────────────────────────────────────

def load_sorted(d: Path, exts: set[str] | None = None) -> list[Path]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        [p for p in d.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.stem,
    )


def load_frames_as_tensor(frame_paths: list[Path], h: int, w: int) -> torch.Tensor:
    """Load RGB frames, resize to (h, w), normalize to [-1, 1].

    Returns float32 tensor [N, h, w, 3].
    """
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Cannot read frame: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(img)
    arr = np.stack(frames, axis=0).astype(np.float32)  # [N, h, w, 3]
    arr = arr / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr)


def load_masks_as_tensor(mask_paths: list[Path], h: int, w: int) -> torch.Tensor:
    """Load binary masks, resize to (h, w).

    Returns float32 tensor [N, h, w, 1] with values in {0.0, 1.0}.
    """
    masks = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            masks.append(np.zeros((h, w, 1), np.float32))
            continue
        if m.ndim == 3:
            m = m.max(axis=2)
        m = (m > 0).astype(np.float32)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m[:, :, np.newaxis])
    arr = np.stack(masks, axis=0)  # [N, h, w, 1]
    return torch.from_numpy(arr)


# ── Padding / chunking ───────────────────────────────────────────────────────

def pad_to_n(
    frames: torch.Tensor, masks: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Repeat the last frame/mask until length reaches n. Returns (padded_f, padded_m, orig_len)."""
    orig_len = frames.shape[0]
    if orig_len >= n:
        return frames[:n], masks[:n], orig_len
    pad_n  = n - orig_len
    last_f = frames[-1:].expand(pad_n, -1, -1, -1)
    last_m = masks[-1:].expand(pad_n, -1, -1, -1)
    return (
        torch.cat([frames, last_f.clone()], dim=0),
        torch.cat([masks,  last_m.clone()], dim=0),
        orig_len,
    )


# ── Inference ────────────────────────────────────────────────────────────────

def run_minimax_chunk(
    pipe,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    seed: int,
    iterations: int,
    num_steps: int,
) -> np.ndarray:
    """Run one MiniMax-Remover inference chunk (exactly MINIMAX_N_FRAMES frames).

    Returns uint8 numpy array of shape [MINIMAX_N_FRAMES, H, W, 3] (RGB).
    """
    result = pipe(
        images=images,
        masks=masks,
        num_frames=MINIMAX_N_FRAMES,
        height=MINIMAX_H,
        width=MINIMAX_W,
        num_inference_steps=num_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
        iterations=iterations,
        output_type="np",
    )
    return normalize_output_frames(result.frames[0])  # [N_frames, H, W, C] uint8 RGB


def normalize_output_frames(frames: np.ndarray) -> np.ndarray:
    """Convert diffusers video output to uint8 RGB.

    diffusers returns numpy video frames as float32 in [0, 1] for output_type="np".
    The first MiniMax v1 run mistakenly treated those frames as [0, 255], causing
    the inpainted mask region to become black after hard-blend.
    """
    arr = np.asarray(frames)
    raw_min = float(np.nanmin(arr))
    raw_max = float(np.nanmax(arr))
    print(f"[minimax] raw output range: min={raw_min:.4f}, max={raw_max:.4f}, dtype={arr.dtype}")
    arr = np.nan_to_num(arr)
    if arr.dtype == np.uint8:
        return arr
    if raw_max <= 1.5:
        arr = arr * 255.0
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


# ── Post-processing ──────────────────────────────────────────────────────────

def apply_hard_blend(
    repair_frames_bgr: list[np.ndarray],
    orig_frame_paths: list[Path],
    blend_mask_paths: list[Path],
) -> list[np.ndarray]:
    """Replace pixels outside the binary mask with the original DAVIS frames.

    blend_mask_paths: can be GT masks or custom shadow masks — whichever defines
    the inpainted region boundary for fair-protocol comparison.
    """
    blended = []
    for rep, op, mp in zip(repair_frames_bgr, orig_frame_paths, blend_mask_paths):
        orig     = cv2.imread(str(op))
        mask_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if orig is None or mask_raw is None:
            blended.append(rep)
            continue
        if mask_raw.ndim == 3:
            mask_raw = mask_raw.max(axis=2)
        binary = (mask_raw > 0).astype(np.uint8)
        h, w   = orig.shape[:2]
        rep_r  = (
            cv2.resize(rep, (w, h), interpolation=cv2.INTER_LINEAR)
            if rep.shape[:2] != (h, w)
            else rep
        )
        m      = binary[:, :, np.newaxis].astype(np.float32)
        result = (
            m * rep_r.astype(np.float32) + (1.0 - m) * orig.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        blended.append(result)
    return blended


def apply_soft_blend(
    repair_frames_bgr: list[np.ndarray],
    orig_frame_paths: list[Path],
    feather_mask_paths: list[Path],
) -> list[np.ndarray]:
    """Alpha-blend MiniMax output with original frames using a pre-computed feather mask.

    Inside the mask region (alpha ≈ 1) the inpainted result is used; outside
    (alpha ≈ 0) the original frame is used; at the boundary the two are linearly
    blended according to the Gaussian-blurred feather alpha.

    feather_mask_paths: float PNG saved as 8-bit 0..255 by prepare_minimax_masks.py
    (feather_mask_frames/).  If a feather mask for a frame is missing, falls back
    to hard binary blend.
    """
    blended = []
    for rep, op, fp in zip(repair_frames_bgr, orig_frame_paths, feather_mask_paths):
        orig = cv2.imread(str(op))
        if orig is None:
            blended.append(rep)
            continue
        h, w = orig.shape[:2]
        rep_r = (
            cv2.resize(rep, (w, h), interpolation=cv2.INTER_LINEAR)
            if rep.shape[:2] != (h, w)
            else rep
        )
        if fp is not None and fp.exists():
            alpha_raw = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if alpha_raw is None:
                alpha = np.ones((h, w), np.float32)
            else:
                if alpha_raw.ndim == 3:
                    alpha_raw = alpha_raw.max(axis=2)
                alpha_raw = alpha_raw.astype(np.float32) / 255.0
                if alpha_raw.shape[:2] != (h, w):
                    alpha_raw = cv2.resize(alpha_raw, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = np.clip(alpha_raw, 0.0, 1.0)
        else:
            alpha = np.ones((h, w), np.float32)

        alpha3 = alpha[:, :, np.newaxis]
        result = (
            alpha3 * rep_r.astype(np.float32)
            + (1.0 - alpha3) * orig.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        blended.append(result)
    return blended


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
        writer.write(f)
    writer.release()


def write_frames(out_dir: Path, frames: list[np.ndarray], stem_paths: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame, src_path in zip(frames, stem_paths):
        cv2.imwrite(str(out_dir / f"{src_path.stem}.png"), frame)


def write_mask_frames(out_dir: Path, mask_paths: list[Path], stem_paths: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for mask_path, src_path in zip(mask_paths, stem_paths):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        mask = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{src_path.stem}.png"), mask)


def generate_masked_preview(
    frames_dir: Path,
    mask_paths: list[Path],
    frame_paths: list[Path],
    output_mp4: Path,
    fps: float,
) -> None:
    preview_frames: list[np.ndarray] = []
    for frame_path, mask_path in zip(frame_paths, mask_paths):
        frame = cv2.imread(str(frames_dir / frame_path.name))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if frame is None or mask is None:
            continue
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        h, w = frame.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (mask > 0).astype(np.float32)[:, :, None]
        green = np.zeros_like(frame, dtype=np.float32)
        green[:, :, 1] = 255.0
        overlay = binary * (0.4 * frame.astype(np.float32) + 0.6 * green) + (1.0 - binary) * frame.astype(np.float32)
        preview_frames.append(overlay.clip(0, 255).astype(np.uint8))
    if preview_frames:
        write_video(output_mp4, preview_frames, fps=fps)


# ── Manifest ─────────────────────────────────────────────────────────────────

def write_manifest(
    out_dir: Path,
    seq: str,
    version: str,
    n_frames: int,
    elapsed: float,
    num_steps: int,
    iterations: int,
    mask_protocol: str,
    hard_blend: bool,
    mask_dir_used: str,
) -> None:
    manifest = {
        "exp_id": f"minimax_remover_{mask_protocol}_{version}",
        "readable_name": (
            f"MiniMax-Remover {mask_protocol} {version} "
            f"(steps={num_steps}, iter={iterations})"
        ),
        "sequence": seq,
        "family": "MiniMax-Remover",
        "comparison_type": "inpaint_only",
        "direction": "C",
        "version": version,
        "mask_protocol": mask_protocol,
        "mask_dir": mask_dir_used,
        "baseline": BASELINE,
        "audit_status": "exploratory",
        "stage_gate": "PSNR_proxy >= ProPainter",
        "parameters": {
            "num_inference_steps": num_steps,
            "iterations": iterations,
            "height": MINIMAX_H,
            "width": MINIMAX_W,
            "num_frames_per_chunk": MINIMAX_N_FRAMES,
            "model": "zibojia/minimax-remover",
            "inference_mode": "native video DiT (temporal joint modeling)",
        },
        "script": str(PART3_ROOT / "inpainting" / "run_minimax_remover_gtmask.py"),
        "command": (
            f"conda run -p /data3/jli657/envs/minimax_env python3 "
            f"part3/inpainting/run_minimax_remover_gtmask.py "
            f"--seq {seq} --version {version}"
            + (f" --mask_dir {mask_dir_used}" if mask_protocol != "davis_gt" else "")
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
            f" MiniMax-Remover (video DiT) "
            f"steps={num_steps}, mask_dilation_iterations={iterations}"
            + ("hard-blend: mask  DAVIS  inpaint-only "
               if hard_blend else "raw output, no hard-blend")
        ),
        "what_to_check": (
            "1. inpaint_out.mp4/\n"
            "2.  pure_propainter_gtmask/inpaint_out.mp4\n"
            "3. car-shadow\n"
            "4. horsejump-low/streak"
        ),
        "current_takeaway": "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[manifest] → {path}")


def write_metrics(out_dir: Path, seq: str, version: str, n_frames: int, elapsed: float) -> None:
    metrics = {
        "sequence": seq,
        "method_id": f"minimax_remover_gtmask_{version}",
        "audit_status": "needs_review",
        "n_frames": n_frames,
        "elapsed_sec": round(elapsed, 1),
        "note": "No quantitative evaluation run yet; use visual review first.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def write_experiment_card(
    out_dir: Path,
    seq: str,
    version: str,
    mask_protocol: str,
    n_frames: int,
    elapsed: float,
    hard_blend: bool,
) -> None:
    card = f"""# MiniMax-Remover GT Mask {version}

- sequence: `{seq}`
- method_id: `minimax_remover_gtmask_{version}`
- family: `MiniMax-Remover`
- audit_status: `needs_review`
- mask_protocol: `{mask_protocol}`
- frames: `{n_frames}`
- elapsed_sec: `{elapsed:.1f}`
- hard_blend: `{hard_blend}`

## Purpose

Re-run MiniMax-Remover after fixing the output scaling bug in v1. The v1 output treated diffusers `[0, 1]` float frames as uint8, which made the inpainted region appear black.

## Files

- `mask_frames/`: binary masks used for inpainting
- `repair_frames/`: raw MiniMax repaired frames before hard-blend
- `masked_in.mp4`: mask overlay preview
- `inpaint_out.mp4`: final hard-blended comparison video
- `run_manifest.json`: executable metadata
- `metrics.json`: review status and runtime placeholder

## Visual Check

Compare `inpaint_out.mp4` against `pure_propainter_gtmask/inpaint_out.mp4`. This experiment is only promising if the masked region is actually synthesized and temporally coherent.
"""
    (out_dir / "experiment_card.md").write_text(card, encoding="utf-8")


def sync_deliverable(out_dir: Path, seq: str, exp_name: str) -> None:
    deliverable_dir = DELIVERABLES_ROOT / seq / exp_name
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    for name in ["inpaint_out.mp4", "masked_in.mp4", "run_manifest.json", "metrics.json", "experiment_card.md"]:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, deliverable_dir / name)
    for dirname in ["mask_frames", "repair_frames"]:
        src_dir = out_dir / dirname
        dst_dir = deliverable_dir / dirname
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
    print(f"[deliverable] → {deliverable_dir}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline(device: torch.device):
    """Load MiniMax-Remover pipeline from local workspace clone."""
    workspace = str(MINIMAX_WORKSPACE)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)

    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import UniPCMultistepScheduler
    from pipeline_minimax_remover import Minimax_Remover_Pipeline
    from transformer_minimax_remover import Transformer3DModel

    print("[minimax] Loading VAE ...")
    vae = AutoencoderKLWan.from_pretrained(
        str(MINIMAX_WORKSPACE / "vae"), torch_dtype=torch.float16
    )
    print("[minimax] Loading Transformer ...")
    transformer = Transformer3DModel.from_pretrained(
        str(MINIMAX_WORKSPACE / "transformer"), torch_dtype=torch.float16
    )
    print("[minimax] Loading Scheduler ...")
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(MINIMAX_WORKSPACE / "scheduler")
    )
    pipe = Minimax_Remover_Pipeline(
        transformer=transformer, vae=vae, scheduler=scheduler
    )
    pipe.to(device)
    print("[minimax] Model loaded OK")
    return pipe


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniMax-Remover with DAVIS GT masks (fair inpaint-only protocol)"
    )
    parser.add_argument("--seq",        default="horsejump-low")
    parser.add_argument("--version",    default="v1")
    parser.add_argument(
        "--frames_dir", type=str, default=None,
        help="Custom frame directory. Default: DAVIS JPEGImages/480p/<seq>.",
    )
    parser.add_argument(
        "--mask_dir", type=str, default=None,
        help="Custom mask directory (e.g. shadow-aware masks). "
             "Default: DAVIS GT annotation masks.",
    )
    parser.add_argument("--num_steps",  type=int,   default=12,
                        help="Diffusion steps (default: 12)")
    parser.add_argument("--iterations", type=int,   default=6,
                        help="Mask dilation iterations (default: 6)")
    parser.add_argument("--fps",        type=float, default=24.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--hard_blend", action="store_true",  default=True,
                        help="Replace mask-outside with DAVIS original (default: on)")
    parser.add_argument("--no_hard_blend", dest="hard_blend", action="store_false",
                        help="Disable hard-blend (raw output)")
    parser.add_argument("--soft_blend", action="store_true", default=False,
                        help="Use Gaussian-feathered alpha blend instead of hard binary blend. "
                             "Requires --feather_mask_dir or feather_mask_frames/ alongside mask_dir.")
    parser.add_argument("--feather_mask_dir", type=str, default=None,
                        help="Directory of 8-bit feather alpha PNGs (0..255) produced by "
                             "prepare_minimax_masks.py. Used only when --soft_blend is set.")
    parser.add_argument("--gpu",        type=int,   default=0,
                        help="CUDA device index (default: 0)")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[minimax] device: {device}  (CUDA_VISIBLE_DEVICES={args.gpu})")

    # ── Resolve paths ──────────────────────────────────────────────────────
    frames_dir = Path(args.frames_dir) if args.frames_dir else (DAVIS_FRAMES / args.seq)
    if args.mask_dir:
        masks_dir     = Path(args.mask_dir)
        mask_protocol = "shadow_mask"
    else:
        masks_dir     = DAVIS_GT_MASKS / args.seq
        mask_protocol = "davis_gt"

    if not frames_dir.exists():
        print(f"[ERROR] frames dir not found: {frames_dir}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"[ERROR] masks dir not found: {masks_dir}")
        sys.exit(1)

    # Resolve feather mask dir for soft-blend
    feather_mask_dir: Path | None = None
    if args.soft_blend:
        if args.feather_mask_dir:
            feather_mask_dir = Path(args.feather_mask_dir)
        elif args.mask_dir:
            # Conventional sibling directory produced by prepare_minimax_masks.py
            candidate = Path(args.mask_dir).parent / "feather_mask_frames"
            if candidate.exists():
                feather_mask_dir = candidate
                print(f"[soft_blend] auto-detected feather masks at {feather_mask_dir}")
        if feather_mask_dir is None or not feather_mask_dir.exists():
            print("[WARN] --soft_blend requested but no feather_mask_dir found; "
                  "falling back to hard-blend")
            args.soft_blend = False

    # Output directory
    if mask_protocol == "davis_gt":
        exp_name = f"minimax_remover_gtmask_{args.version}"
    else:
        # version already describes the mask variant (e.g. shadow_edge_v1)
        exp_name = f"minimax_remover_{args.version}"
    out_dir = RESULTS_ROOT / args.seq / "direction_c" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    inpaint_out = out_dir / "inpaint_out.mp4"
    if inpaint_out.exists():
        print(f"[skip] {exp_name}/inpaint_out.mp4 already exists")
        return

    # ── Load input files ───────────────────────────────────────────────────
    frame_paths = load_sorted(frames_dir)
    mask_paths  = load_sorted(masks_dir, {".png"})

    n_orig  = len(frame_paths)
    n_masks = len(mask_paths)
    print(f"[minimax] seq={args.seq}  frames={n_orig}  masks={n_masks}")
    print(f"  mask_protocol: {mask_protocol}  ({masks_dir})")
    print(f"  output: {out_dir}")

    if n_orig == 0:
        print(f"[ERROR] No frames found in {frames_dir}")
        sys.exit(1)

    # Align mask count to frame count
    if n_masks < n_orig:
        print(f"[WARN] Fewer masks ({n_masks}) than frames ({n_orig}), padding with last mask")
        mask_paths = mask_paths + [mask_paths[-1]] * (n_orig - n_masks)
    elif n_masks > n_orig:
        mask_paths = mask_paths[:n_orig]

    # ── Build input tensors ────────────────────────────────────────────────
    print(f"[minimax] Loading frames (resize → {MINIMAX_H}×{MINIMAX_W}) ...")
    frames_t = load_frames_as_tensor(frame_paths, MINIMAX_H, MINIMAX_W)  # [N, H, W, 3]
    print(f"[minimax] Loading masks ...")
    masks_t  = load_masks_as_tensor(mask_paths, MINIMAX_H, MINIMAX_W)    # [N, H, W, 1]

    # ── Load pipeline ──────────────────────────────────────────────────────
    pipe = load_pipeline(device)

    # ── Inference ──────────────────────────────────────────────────────────
    t0 = time.time()
    repair_frames_rgb: list[np.ndarray] = []

    if n_orig <= MINIMAX_N_FRAMES:
        # Pad to exactly 81 frames
        frames_padded, masks_padded, orig_len = pad_to_n(
            frames_t, masks_t, MINIMAX_N_FRAMES
        )
        print(f"[minimax] Running inference (pad {n_orig}→{MINIMAX_N_FRAMES} frames) ...")
        out = run_minimax_chunk(
            pipe, frames_padded, masks_padded, device,
            args.seed, args.iterations, args.num_steps,
        )  # [81, H, W, 3] uint8 RGB
        for frame in out[:orig_len]:
            repair_frames_rgb.append(frame)
    else:
        # Chunk processing with 10-frame overlap for seamless stitching
        stride = MINIMAX_N_FRAMES - 10
        starts = list(range(0, n_orig, stride))
        print(f"[minimax] {n_orig} > {MINIMAX_N_FRAMES} frames → chunk mode "
              f"({len(starts)} chunks, stride={stride}) ...")
        for chunk_idx, start in enumerate(starts):
            end       = min(start + MINIMAX_N_FRAMES, n_orig)
            chunk_f   = frames_t[start:end]
            chunk_m   = masks_t[start:end]
            chunk_len = chunk_f.shape[0]
            cf_pad, cm_pad, _ = pad_to_n(chunk_f, chunk_m, MINIMAX_N_FRAMES)
            print(f"  chunk {chunk_idx+1}/{len(starts)}: "
                  f"frames [{start}:{end}] ({chunk_len} frames) ...")
            out = run_minimax_chunk(
                pipe, cf_pad, cm_pad, device,
                args.seed, args.iterations, args.num_steps,
            )
            # Keep the first chunk in full, then skip the 10-frame overlap on
            # later chunks. This preserves all frames in long custom videos.
            keep_start = 0 if chunk_idx == 0 else 10
            keep_end   = chunk_len if chunk_idx == 0 else min(keep_start + stride, chunk_len)
            for frame in out[keep_start:keep_end]:
                repair_frames_rgb.append(frame)
            if end >= n_orig:
                break

    elapsed = time.time() - t0
    print(f"[minimax] done in {elapsed:.1f}s  ({elapsed/max(1,n_orig):.1f}s/frame)")
    print(f"  collected {len(repair_frames_rgb)} repair frames (expected {n_orig})")

    # Trim/pad collected frames to exactly n_orig
    if len(repair_frames_rgb) > n_orig:
        repair_frames_rgb = repair_frames_rgb[:n_orig]
    elif len(repair_frames_rgb) < n_orig:
        print(f"[WARN] Got fewer frames than expected; padding with last")
        while len(repair_frames_rgb) < n_orig:
            repair_frames_rgb.append(repair_frames_rgb[-1])

    # ── Convert RGB → BGR for OpenCV ───────────────────────────────────────
    repair_frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in repair_frames_rgb]

    # Save standardized intermediate outputs before hard-blend.
    write_frames(out_dir / "repair_frames", repair_frames_bgr, frame_paths[:n_orig])
    write_mask_frames(out_dir / "mask_frames", mask_paths[:n_orig], frame_paths[:n_orig])
    generate_masked_preview(
        frames_dir, mask_paths[:n_orig], frame_paths[:n_orig],
        out_dir / "masked_in.mp4", fps=args.fps,
    )

    # ── Blend ──────────────────────────────────────────────────────────────
    if args.soft_blend and feather_mask_dir is not None:
        feather_paths = load_sorted(feather_mask_dir, {".png"})
        # Align feather paths to frame paths by stem name
        feather_by_stem = {p.stem: p for p in feather_paths}
        aligned_feather = [
            feather_by_stem.get(fp.stem, None) for fp in frame_paths[:n_orig]
        ]
        print(f"[soft_blend] applying Gaussian alpha-blend ({len(repair_frames_bgr)} frames) ...")
        repair_frames_bgr = apply_soft_blend(
            repair_frames_bgr, frame_paths[:n_orig], aligned_feather
        )
        print("[soft_blend] done")
    elif args.hard_blend:
        # Use the same masks that defined the inpainted region for blend boundary.
        blend_mask_paths = mask_paths[:n_orig]
        orig_frame_paths_blend = frame_paths[:n_orig]
        print(f"[hard_blend] applying ({len(repair_frames_bgr)} frames) ...")
        repair_frames_bgr = apply_hard_blend(
            repair_frames_bgr, orig_frame_paths_blend, blend_mask_paths
        )
        print("[hard_blend] done")

    # ── Write outputs ──────────────────────────────────────────────────────
    write_video(inpaint_out, repair_frames_bgr, fps=args.fps)
    print(f"[video] inpaint_out.mp4 → {inpaint_out}")

    blend_mode = "soft_blend" if args.soft_blend else ("hard_blend" if args.hard_blend else "raw")
    write_manifest(
        out_dir, args.seq, args.version, n_orig, elapsed,
        num_steps=args.num_steps, iterations=args.iterations,
        mask_protocol=mask_protocol, hard_blend=(blend_mode == "hard_blend"),
        mask_dir_used=str(masks_dir),
    )
    write_metrics(out_dir, args.seq, args.version, n_orig, elapsed)
    write_experiment_card(
        out_dir, args.seq, args.version, mask_protocol,
        n_orig, elapsed, hard_blend=(blend_mode == "hard_blend"),
    )
    sync_deliverable(out_dir, args.seq, exp_name)

    baseline_mp4 = RESULTS_ROOT / args.seq / "direction_c" / BASELINE / "inpaint_out.mp4"
    print(f"\n[OK] MiniMax-Remover complete for {args.seq}")
    print(f"\nVisual comparison:")
    print(f"  MiniMax : {inpaint_out}")
    print(f"  Baseline: {baseline_mp4}")


if __name__ == "__main__":
    main()
