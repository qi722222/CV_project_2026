"""
run_direction_b_pi3_transplant.py
Direction B: Pi3 -> VGGT4D Algorithm Transplant

Core idea: Replace VGGT backbone with Pi3 (stronger 3D foundation model).
Extract Q/K attention features from Pi3's cross-frame decoder blocks,
then apply VGGT4D's Gram Similarity dynamic discovery algorithm.

Pi3 uses DINOv2 encoder + RoPE-augmented cross-frame Transformer decoder.
The decoder processes ALL frames' tokens jointly (like VGGT's global attention),
making the Q/K features conceptually equivalent for dynamic cue mining.

Three stages:
  Stage 1: Pi3 encoder + decoder -> extract QK dict -> Gram Similarity -> rough masks
  Stage 2: Refine extrinsics with masks (using Pi3 depth/poses)
  Stage 3: Projection gradient refinement (reuse VGGT4D's RefineDynMask)

Usage:
  PYTHONPATH=/data3/jli657/VGGT4D:/data3/jli657/Pi3 \
  python part3/run_direction_b_pi3_transplant.py \
    --sequences tennis blackswan horsejump-low bmx-trees car-shadow koala
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VGGT4D_REPO = "/data3/jli657/VGGT4D"
PI3_REPO = "/data3/jli657/Pi3"
VGGT4D_CKPT = "/data3/jli657/VGGT4D/ckpts/model_tracker_fixed_e20.pt"
DAVIS_FRAMES = Path("/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p")
DAVIS_GT = Path("/home/jli657/shared_data/project3/DAVIS/Annotations/480p")
WILD_FRAMES = Path("/data3/jli657/project3/wild_frames/wild_video-1person")

SEQUENCE_VIDEO_ROOTS: Dict[str, Path] = {
    "tennis": DAVIS_FRAMES / "tennis",
    "bmx-trees": DAVIS_FRAMES / "bmx-trees",
    "blackswan": DAVIS_FRAMES / "blackswan",
    "car-shadow": DAVIS_FRAMES / "car-shadow",
    "horsejump-low": DAVIS_FRAMES / "horsejump-low",
    "koala": DAVIS_FRAMES / "koala",
    "wild_video-1person": WILD_FRAMES,
}

MAX_FRAMES_PER_SEQ = 60


# ---------------------------------------------------------------------------
# Pi3 QK Extractor (hook-based)
# ---------------------------------------------------------------------------

class Pi3QKExtractor:
    """
    Hooks into Pi3's decoder BlockRope layers to capture Q and K tensors
    ONLY from cross-frame attention blocks (odd-indexed blocks).

    Pi3's decoder alternates between:
      Even blocks (i=0,2,4,...): per-frame attention  -> x.shape = [B*N, hw, C]
      Odd blocks  (i=1,3,5,...): cross-frame attention -> x.shape = [B, N*hw, C]

    We capture only the cross-frame (odd) blocks since these encode
    inter-frame dynamics, equivalent to VGGT's global cross-frame attention.

    Output QK dict matches VGGT4D's organize_qk_dict format:
      global_q: [n_layer, 1, 1, n_head, n_img*n_tok, head_dim]
      global_k: [n_layer, 1, 1, n_head, n_img*n_tok, head_dim]
    """

    def __init__(self, model: nn.Module, n_img: int, n_tok: int, n_register: int = 5):
        self.n_img = n_img
        self.n_tok = n_tok          # patch tokens per image (H_tok * W_tok)
        self.n_register = n_register
        self.n_per_frame = n_tok + n_register   # total tokens per image (with registers)
        self.cross_frame_ntok = n_img * self.n_per_frame  # expected N for cross-frame
        self.q_list: List[torch.Tensor] = []
        self.k_list: List[torch.Tensor] = []
        self._hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        from pi3.models.layers.attention import FlashAttentionRope
        for name, module in model.named_modules():
            # Only hook FlashAttentionRope modules inside the main decoder
            # (not point_decoder, conf_decoder, camera_decoder)
            if isinstance(module, FlashAttentionRope) and name.startswith("decoder."):
                hook = module.register_forward_hook(self._capture_cross_frame_qk)
                self._hooks.append(hook)

    def _capture_cross_frame_qk(self, module, inputs, output):
        """
        Hook: extract Q and K ONLY from cross-frame attention.
        Cross-frame: x.shape[1] == n_img * (n_tok + n_register)
        Per-frame: x.shape[1] == n_tok + n_register
        """
        x = inputs[0]  # [B, N, C]
        B, N, C = x.shape

        # Only capture cross-frame attention (odd decoder blocks)
        # Detect by token count: cross-frame has N == n_img * n_per_frame
        if N != self.cross_frame_ntok:
            return  # Skip per-frame attention blocks

        with torch.no_grad():
            qkv = module.qkv(x.float()).reshape(B, N, 3, module.num_heads,
                                         C // module.num_heads).transpose(1, 3)
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            q = module.q_norm(q.to(x.dtype))  # [B, n_head, N, head_dim]
            k = module.k_norm(k.to(x.dtype))
        self.q_list.append(q.detach().cpu().float())
        self.k_list.append(k.detach().cpu().float())

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_qk_dict(self) -> dict:
        """
        Build VGGT4D-compatible qk_dict from captured cross-frame Q/K tensors.

        Cross-frame tokens layout: [B=1, n_img * (n_register + n_tok), C]
        We KEEP the register tokens to maintain VGGT4D's token indexing convention:
          organize_qk_dict splits: [0]=cam, [1:5]=register, [5:]=patch
        In Pi3: [0:5]=register, [5:]=patch → same split position works!

        VGGT4D organize_qk_dict expects:
          global_q: [n_layer, 1, 1, n_head, n_img*(n_register+n_tok), head_dim]
        """
        if not self.q_list:
            return {}

        global_q_all = []
        global_k_all = []
        n_per = self.n_per_frame   # n_register + n_tok per image
        n_img = self.n_img

        for q, k in zip(self.q_list, self.k_list):
            # q: [1, n_head, n_img*n_per_frame, head_dim]
            # Rearrange to [n_img, n_head, n_per_frame, head_dim], then flatten cross-frame
            # VGGT4D format needs: [n_layer, 1, 1, n_head, n_img*n_per_frame, head_dim]
            global_q_all.append(q.unsqueeze(0).unsqueeze(0))   # [1, 1, n_head, n*n_per, hdim]
            global_k_all.append(k.unsqueeze(0).unsqueeze(0))

        global_q = torch.cat(global_q_all, dim=0)  # [n_cross_layers, 1, 1, n_head, n_img*n_per, hdim]
        global_k = torch.cat(global_k_all, dim=0)

        # For frame_q/frame_k, VGGT4D expects [n_layer, 1, n_img, n_head, n_per, head_dim]
        # We reshape to per-frame format
        n_l = global_q.shape[0]
        n_head = global_q.shape[3]
        hdim = global_q.shape[-1]
        # [n_l, 1, 1, n_head, n_img*n_per, hdim] -> [n_l, 1, n_img, n_head, n_per, hdim]
        frame_q = global_q.reshape(n_l, 1, n_img, n_head, n_per, hdim).permute(0, 2, 1, 3, 4, 5)
        # wait organize_qk_dict does: rearrange(frame_q, "n_layer 1 n_img n_head n_tok c -> ...")
        # so frame_q shape should be [n_layer, 1, n_img, n_head, n_tok, c]
        frame_q_final = global_q.reshape(n_l, 1, n_img, n_head, n_per, hdim)
        frame_k_final = global_k.reshape(n_l, 1, n_img, n_head, n_per, hdim)

        return {
            "global_q": global_q,
            "global_k": global_k,
            "frame_q": frame_q_final,
            "frame_k": frame_k_final,
        }


# ---------------------------------------------------------------------------
# Pi3 forward with QK extraction
# ---------------------------------------------------------------------------

def pi3_inference_with_qk(model, images_vggt: torch.Tensor) -> Tuple[dict, dict, torch.Tensor, int, int]:
    """
    Run Pi3 inference while extracting QK attention from cross-frame decoder blocks.
    
    Args:
        images_vggt: [N, 3, H, W] preprocessed by VGGT's load_and_preprocess_images
                     (ImageNet-normalized, resized to ~518px)
    Returns: (predictions, qk_dict, enc_feat, n_h, n_w)
    """
    device = next(model.parameters()).device
    n_img = images_vggt.shape[0]

    # Pi3 normalizes internally, but it expects [0, 1] input range.
    # Undo VGGT preprocessing (ImageNet denorm) to get [0, 1] images for Pi3
    # VGGT uses ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images_01 = images_vggt * IMAGENET_STD + IMAGENET_MEAN
    images_01 = images_01.clamp(0.0, 1.0)

    H, W = images_vggt.shape[-2:]
    n_h, n_w = H // model.patch_size, W // model.patch_size
    n_tok = n_h * n_w

    # Install QK extractor hooks
    extractor = Pi3QKExtractor(model, n_img=n_img, n_tok=n_tok,
                                n_register=model.patch_start_idx)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Pi3 forward pass: input [B=1, N, 3, H, W]
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(images_01[None])

    qk_dict = extractor.get_qk_dict()
    extractor.remove_hooks()

    # Extract DINOv2 encoder features (per-frame patch embeddings)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            # Pi3 encoder does its own ImageNet normalization internally  
            imgs_normalized = images_01.to(device)
            enc_out = model.encoder(imgs_normalized, is_training=True)
            enc_feat = enc_out["x_norm_patchtokens"] if isinstance(enc_out, dict) else enc_out
            enc_feat = enc_feat.float().cpu()  # [N, H_tok*W_tok, C]

    return predictions, qk_dict, enc_feat, n_h, n_w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sorted_frames(video_dir: Path, max_frames: Optional[int] = None) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in exts],
                   key=lambda p: p.stem)
    if max_frames is not None and len(paths) > max_frames:
        idxs = np.linspace(0, len(paths) - 1, max_frames, dtype=int)
        paths = [paths[i] for i in idxs]
    return paths


def compute_jm_jr_f(pred_dir: Path, gt_dir: Path) -> Dict[str, float]:
    if not gt_dir.exists():
        return {"JM": -1.0, "JR": -1.0, "F": -1.0}
    ious, recalls, f_scores = [], [], []
    for gt_path in sorted(gt_dir.glob("*.png"), key=lambda p: p.stem):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt_arr = np.array(Image.open(gt_path).convert("L")) > 0
        pred_img = Image.open(pred_path).convert("L")
        if pred_img.size != (gt_arr.shape[1], gt_arr.shape[0]):
            pred_img = pred_img.resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
        pred_arr = np.array(pred_img) > 127
        inter = np.logical_and(gt_arr, pred_arr).sum()
        union = np.logical_or(gt_arr, pred_arr).sum()
        iou = inter / (union + 1e-8)
        ious.append(iou)
        recalls.append(float(iou >= 0.5))
        gt_cont = cv2.Canny(gt_arr.astype(np.uint8) * 255, 50, 150) > 0
        pred_cont = cv2.Canny(pred_arr.astype(np.uint8) * 255, 50, 150) > 0
        tp = np.logical_and(gt_cont, pred_cont).sum()
        prec = tp / (pred_cont.sum() + 1e-8)
        rec = tp / (gt_cont.sum() + 1e-8)
        f_scores.append(2 * prec * rec / (prec + rec + 1e-8))
    if not ious:
        return {"JM": 0.0, "JR": 0.0, "F": 0.0}
    return {"JM": float(np.mean(ious)), "JR": float(np.mean(recalls)), "F": float(np.mean(f_scores))}


# ---------------------------------------------------------------------------
# Pi3-adapted dynamic map extraction
# ---------------------------------------------------------------------------

def _compute_gram_similarity_chunked(
    q_ref: torch.Tensor,   # [n_layer, n_head, n_tok, hdim]
    q_srcs: torch.Tensor,  # [n_src, n_layer, n_head, n_tok, hdim]
    n_h: int, n_w: int,
    mode: str = "mean",   # "mean" or "std"
) -> torch.Tensor:
    """Memory-efficient Gram Similarity: process one source frame at a time."""
    n_src = q_srcs.shape[0]
    device = q_ref.device
    acc = torch.zeros(n_h, n_w, device=device, dtype=torch.float32)
    if mode == "std":
        acc_sq = torch.zeros(n_h, n_w, device=device, dtype=torch.float32)

    q_ref_f = q_ref.half()  # [n_layer, n_head, n_tok, hdim]
    for si in range(n_src):
        q_src_f = q_srcs[si].half()  # [n_layer, n_head, n_tok, hdim]
        # [n_layer, n_head, n_tok, n_tok]
        attn_i = q_ref_f @ q_src_f.transpose(-2, -1)
        # -> [n_h, n_w, n_layer*n_head, n_tok]
        attn_i = attn_i.reshape(attn_i.shape[0], attn_i.shape[1], n_h, n_w, attn_i.shape[-1])
        attn_i = attn_i.permute(2, 3, 0, 1, 4).reshape(n_h, n_w, -1, attn_i.shape[-1])
        val = attn_i.mean(dim=(2, 3)).float()  # [n_h, n_w]
        acc += val
        if mode == "std":
            acc_sq += val ** 2
        del attn_i, q_src_f

    mean_map = acc / n_src
    if mode == "std":
        std_map = (acc_sq / n_src - mean_map ** 2).clamp(min=0).sqrt()
        return std_map
    return mean_map


def extract_dyn_map_pi3(qk_dict: dict, images: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient Pi3 adaptation of VGGT4D's Gram Similarity dynamic map extraction.
    Processes one source frame at a time to avoid OOM on the n_tok x n_tok attention matrix.
    """
    from tqdm import tqdm

    # Keep QK on CPU, move per-frame slices to GPU as needed
    global_q_cpu = qk_dict["global_tok_q"]  # [n_img, n_layer, n_head, n_tok, head_dim] CPU
    n_img, n_layers = global_q_cpu.shape[:2]
    n_img_imgs = images.shape[0]
    device = images.device if images.is_cuda else torch.device("cuda")

    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    # Layer groups proportional to Pi3's 18 layers
    layer_early = list(range(0, n_layers // 3))                           # 0-5
    layer_mid   = list(range(n_layers // 3, n_layers * 2 // 3))           # 6-11
    layer_late  = list(range(n_layers * 2 // 3, n_layers))                # 12-17

    dyn_maps = []
    print(f"[Pi3] Extracting dynamic maps for {n_img_imgs} images (n_layers={n_layers})")

    for ref_id in tqdm(range(n_img_imgs)):
        window = torch.tensor([-6, -4, -2, 2, 4, 6])
        src_ids_t = ref_id + window
        src_ids_t = src_ids_t[(src_ids_t >= 0) & (src_ids_t < n_img_imgs)]
        src_ids = src_ids_t.tolist()

        # Move ref frame layers to GPU
        q_ref_e = global_q_cpu[ref_id][layer_early].to(device)    # [n_early, n_head, n_tok, hdim]
        q_srcs_e = global_q_cpu[src_ids][:, layer_early].to(device)  # [n_src, n_early, ...]

        map_early = _compute_gram_similarity_chunked(q_ref_e, q_srcs_e, n_h, n_w, mode="mean")
        map_early = (map_early - map_early.min()) / (map_early.max() - map_early.min() + 1e-6)
        del q_ref_e, q_srcs_e
        torch.cuda.empty_cache()

        if layer_mid:
            q_ref_m = global_q_cpu[ref_id][layer_mid].to(device)
            q_srcs_m = global_q_cpu[src_ids][:, layer_mid].to(device)
            map_mid = _compute_gram_similarity_chunked(q_ref_m, q_srcs_m, n_h, n_w, mode="std")
            map_mid = (map_mid - map_mid.min()) / (map_mid.max() - map_mid.min() + 1e-6)
            del q_ref_m, q_srcs_m
            torch.cuda.empty_cache()
        else:
            map_mid = torch.ones(n_h, n_w, device=device)

        if layer_late:
            q_ref_l = global_q_cpu[ref_id][layer_late].to(device)
            q_srcs_l = global_q_cpu[src_ids][:, layer_late].to(device)
            map_late = _compute_gram_similarity_chunked(q_ref_l, q_srcs_l, n_h, n_w, mode="mean")
            map_late = (map_late - map_late.min()) / (map_late.max() - map_late.min() + 1e-6)
            map_late_var = _compute_gram_similarity_chunked(q_ref_l, q_srcs_l, n_h, n_w, mode="std")
            map_late_var = (map_late_var - map_late_var.min()) / (map_late_var.max() - map_late_var.min() + 1e-6)
            del q_ref_l, q_srcs_l
            torch.cuda.empty_cache()
        else:
            map_late = torch.ones(n_h, n_w, device=device)
            map_late_var = torch.ones(n_h, n_w, device=device)

        dyn_map = (1 - map_early) * map_mid * (1 - map_late) * map_late_var
        mn, mx = dyn_map.min(), dyn_map.max()
        dyn_map = (dyn_map - mn) / (mx - mn + 1e-6)
        dyn_maps.append(dyn_map.cpu())

    return torch.stack(dyn_maps)


def load_images_for_vggt4d(frame_paths: List[Path], device) -> torch.Tensor:
    """Load and preprocess frames to [N, 3, H, W] normalized tensor."""
    from vggt.utils.load_fn import load_and_preprocess_images
    return load_and_preprocess_images([str(p) for p in frame_paths]).to(device)


# ---------------------------------------------------------------------------
# Main per-sequence runner
# ---------------------------------------------------------------------------

def run_pi3_transplant_on_sequence(
    pi3_model,
    seq_name: str,
    video_dir: Path,
    output_dir: Path,
    device: torch.device,
    max_frames: int = MAX_FRAMES_PER_SEQ,
) -> dict:
    """Run Pi3-backbone VGGT4D algorithm on a sequence."""
    from vggt4d.masks.dynamic_mask import (
        adaptive_multiotsu_variance,
        cluster_attention_maps,
    )
    from vggt4d.masks.refine_dyn_mask import RefineDynMask
    from vggt4d.utils.model_utils import organize_qk_dict

    frame_paths = load_sorted_frames(video_dir, max_frames=max_frames)
    if not frame_paths:
        return {"status": "error", "error": "no frames"}

    print(f"  [{seq_name}] {len(frame_paths)} frames, Pi3 transplant")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_images_for_vggt4d(frame_paths, device)
    n_img, _, h_img, w_img = images.shape

    # --- Stage 1: Pi3 inference + QK extraction ---
    print(f"  [{seq_name}] Stage 1: Pi3 attention mining...")
    predictions, qk_dict_raw, enc_feat, n_h, n_w = pi3_inference_with_qk(
        pi3_model, images)

    # Organize QK dict to get global_tok_q etc.
    qk_dict = organize_qk_dict(qk_dict_raw, n_img)
    print(f"  [{seq_name}] QK organized, n_layers={qk_dict['global_tok_q'].shape[1]}")

    # Use Pi3-adapted dynamic map extraction (handles different layer count)
    dyn_maps = extract_dyn_map_pi3(qk_dict, images)

    feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=n_h, w=n_w)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
    del enc_feat, feat_map

    upsampled_map = F.interpolate(
        rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
        size=(h_img, w_img), mode="bilinear", align_corners=False)
    upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres
    print(f"  [{seq_name}] Stage 1 done. coverage={dyn_masks.float().mean():.3f}")

    # --- Stage 2: Use Pi3 depth/poses for projection gradient refinement ---
    print(f"  [{seq_name}] Stage 3: projection gradient refinement (Pi3 geometry)...")
    try:
        # Use depth from Pi3's local_points z-channel [B, S, H, W, 3]
        local_pts = predictions["local_points"][0]  # [S, H, W, 3]
        depth_pi3 = local_pts[..., 2].cpu().float().numpy()  # [S, H, W]
        # Resize depth to match image resolution if needed
        if depth_pi3.shape[1:] != (h_img, w_img):
            import torch.nn.functional as F_
            depth_t = torch.tensor(depth_pi3).unsqueeze(1)  # [S, 1, H', W']
            depth_t = F_.interpolate(depth_t, size=(h_img, w_img), mode='bilinear', align_corners=False)
            depth_pi3 = depth_t.squeeze(1).numpy()  # [S, H, W]

        available_keys = list(predictions.keys())
        print(f"  [{seq_name}] Pi3 prediction keys: {available_keys}")

        if "camera_poses" in predictions:
            cam2world = predictions["camera_poses"][0].cpu().float().numpy()  # [N, 4, 4]
        else:
            print(f"  [{seq_name}] Using identity camera poses (Pi3 no camera_poses)")
            cam2world = np.eye(4, dtype=np.float32)[None].repeat(n_img, axis=0)

        f_x = w_img
        f_y = w_img
        cx, cy = w_img / 2.0, h_img / 2.0
        intrinsic = np.array([
            [f_x, 0, cx],
            [0, f_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)[None].repeat(n_img, axis=0)  # [N, 3, 3]

        torch.cuda.empty_cache()
        refiner = RefineDynMask(
            images,
            torch.tensor(depth_pi3).to(device),
            dyn_masks.to(device),
            torch.tensor(cam2world).float().to(device),
            torch.tensor(intrinsic).to(device),
            device,
        )
        refined_masks = refiner.refine_masks()
        del refiner
        torch.cuda.empty_cache()
        print(f"  [{seq_name}] Refinement done. coverage={refined_masks.float().mean():.3f}")
        use_refined = True
    except Exception as e:
        print(f"  [{seq_name}] Refinement failed ({e}), using Stage 1 masks")
        refined_masks = dyn_masks
        use_refined = False

    # Save refined masks
    for i, fp in enumerate(frame_paths):
        mask_np = refined_masks[i].cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_np).save(str(output_dir / f"{fp.stem}.png"))

    # Save rough masks for ablation
    rough_dir = output_dir.parent / (output_dir.name + "_rough")
    rough_dir.mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(frame_paths):
        mask_np = dyn_masks[i].cpu().numpy().astype(np.uint8) * 255
        Image.fromarray(mask_np).save(str(rough_dir / f"{fp.stem}.png"))

    return {
        "status": "success",
        "sequence": seq_name,
        "n_frames": len(frame_paths),
        "output_dir": str(output_dir),
        "use_refined": use_refined,
        "mask_coverage_stage1": float(dyn_masks.float().mean().item()),
        "mask_coverage_stage3": float(refined_masks.float().mean().item()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direction B: Pi3 transplant dynamic masks")
    p.add_argument("--sequences", nargs="+",
                   default=["tennis", "blackswan", "horsejump-low", "koala",
                            "bmx-trees", "car-shadow"])
    p.add_argument("--output_root",
                   default="/data3/jli657/project3/part3/outputs/direction_b/pi3_transplant")
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_SEQ)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_csv",
                   default="/home/jli657/my_storage2_1T/project3/eval/direction_b_pi3_results.csv")
    p.add_argument("--ckpt", default=None,
                   help="Pi3 checkpoint (None = load from HuggingFace)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    output_root = Path(args.output_root)

    if VGGT4D_REPO not in sys.path:
        sys.path.insert(0, VGGT4D_REPO)
    if PI3_REPO not in sys.path:
        sys.path.insert(0, PI3_REPO)

    if not args.eval_only:
        from pi3.models.pi3 import Pi3
        print("Loading Pi3 model...")
        if args.ckpt:
            model = Pi3().to(device).eval()
            from safetensors.torch import load_file
            model.load_state_dict(load_file(args.ckpt))
        else:
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        print("Pi3 loaded.")
    else:
        model = None

    rows = []
    for seq in args.sequences:
        video_dir = SEQUENCE_VIDEO_ROOTS.get(seq)
        if video_dir is None or not video_dir.exists():
            print(f"[skip] {seq}: video dir not found")
            continue

        mask_dir = output_root / seq
        run_meta: dict = {"sequence": seq}

        if not args.eval_only:
            try:
                result = run_pi3_transplant_on_sequence(
                    model, seq, video_dir, mask_dir, device,
                    max_frames=args.max_frames)
                run_meta.update(result)
                mask_dir.mkdir(parents=True, exist_ok=True)
                with open(mask_dir / "run_meta.json", "w") as f:
                    json.dump(run_meta, f, indent=2)
            except Exception as e:
                import traceback
                traceback.print_exc()
                run_meta["status"] = f"error: {e}"

        gt_dir = DAVIS_GT / seq
        if mask_dir.exists() and gt_dir.exists():
            metrics = compute_jm_jr_f(mask_dir, gt_dir)
            run_meta.update(metrics)
            print(f"  [{seq}] JM={metrics['JM']:.4f} JR={metrics['JR']:.4f}")
        else:
            run_meta.update({"JM": -1.0, "JR": -1.0, "F": -1.0})

        rows.append(run_meta)

    import csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "status", "n_frames", "JM", "JR", "F",
              "mask_coverage_stage1", "mask_coverage_stage3", "use_refined", "output_dir"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {out_csv}")

    print("\n=== Direction B (Pi3 Transplant) Results ===")
    print(f"{'Seq':<20} {'JM':>8} {'JR':>8} {'F':>8}")
    for r in rows:
        jm = r.get("JM", -1.0)
        jr = r.get("JR", -1.0)
        f = r.get("F", -1.0)
        print(f"  {r['sequence']:<18} {jm:>8.4f} {jr:>8.4f} {f:>8.4f}")


if __name__ == "__main__":
    main()
