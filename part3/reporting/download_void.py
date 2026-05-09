"""
Download VOID model components via hf-mirror.com
- VOID inference code (Space: sam-motamed/VOID)
- CogVideoX-Fun V1.5 5b-InP: text_encoder, tokenizer, vae, scheduler, config
  (skip transformer - void_pass1.safetensors IS the fine-tuned transformer)
- VOID weights: netflix/void-model (pass1 only for memory efficiency)
"""

import os
import sys
import shutil
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download, HfApi

VOID_BASE = Path("/data3/jli657/void-model")
COGVX_BASE = Path("/data3/jli657/void-model/CogVideoX-Fun-V1.5-5b-InP")
VOID_CKPT = Path("/data3/jli657/void-model/checkpoints")

VOID_BASE.mkdir(parents=True, exist_ok=True)
COGVX_BASE.mkdir(parents=True, exist_ok=True)
VOID_CKPT.mkdir(parents=True, exist_ok=True)


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Step 1: Download VOID Space code ───────────────────────────────────────
log("Step 1: Downloading VOID Space code from sam-motamed/VOID...")

VOID_CODE = VOID_BASE / "code"
VOID_CODE.mkdir(exist_ok=True)

CODE_FILES = [
    "app.py",
    "requirements.txt",
    "videox_fun/__init__.py",
    "videox_fun/models/__init__.py",
    "videox_fun/models/cache_utils.py",
    "videox_fun/models/cogvideox_transformer3d.py",
    "videox_fun/models/cogvideox_vae.py",
    "videox_fun/pipeline/__init__.py",
    "videox_fun/pipeline/pipeline_cogvideox_fun.py",
    "videox_fun/pipeline/pipeline_cogvideox_fun_inpaint.py",
    "videox_fun/utils/__init__.py",
    "videox_fun/utils/fp8_optimization.py",
    "videox_fun/utils/optical_flow_utils.py",
    "videox_fun/utils/utils.py",
    "videox_fun/dist/__init__.py",
    "videox_fun/dist/cogvideox_xfuser.py",
]

api = HfApi(endpoint="https://hf-mirror.com")

for fname in CODE_FILES:
    local_path = VOID_CODE / fname
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        log(f"  [skip] {fname}")
        continue
    try:
        p = hf_hub_download(
            "sam-motamed/VOID", fname, repo_type="space",
            local_dir=str(VOID_CODE)
        )
        log(f"  [ok] {fname}")
    except Exception as e:
        log(f"  [WARN] {fname}: {e}")

log(f"Code downloaded to {VOID_CODE}")


# ─── Step 2: Download CogVideoX-Fun components (NO transformer) ─────────────
log("Step 2: Downloading CogVideoX-Fun V1.5-5b-InP (text_encoder, vae, configs)...")

BASE_REPO = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"

# Files to download (skip transformer - 10GB, void_pass1 replaces it)
BASE_FILES = [
    "model_index.json",
    "configuration.json",
    "scheduler/scheduler_config.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",  # ~4.65 GB
    "text_encoder/model-00002-of-00002.safetensors",  # ~4.65 GB
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",       # ~0.40 GB
    "transformer/config.json",  # needed for architecture config
]

for fname in BASE_FILES:
    local_path = COGVX_BASE / fname
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 1000:
        log(f"  [skip] {fname} ({local_path.stat().st_size/1024/1024:.1f} MB)")
        continue
    log(f"  Downloading {fname}...")
    try:
        p = hf_hub_download(
            BASE_REPO, fname,
            local_dir=str(COGVX_BASE),
            force_download=False
        )
        sz = Path(p).stat().st_size
        log(f"  [ok] {fname} ({sz/1024/1024:.1f} MB)")
    except Exception as e:
        log(f"  [FAIL] {fname}: {e}")


# ─── Step 3: Download VOID checkpoints ─────────────────────────────────────
log("Step 3: Downloading VOID checkpoints from netflix/void-model...")

VOID_WEIGHTS = [
    "void_pass1.safetensors",   # ~10.38 GB - VOID fine-tuned transformer
    # void_pass2.safetensors - optional temporal refinement, skip for now
]

for fname in VOID_WEIGHTS:
    local_path = VOID_CKPT / fname
    if local_path.exists() and local_path.stat().st_size > 1024*1024*1024:
        log(f"  [skip] {fname} ({local_path.stat().st_size/1024/1024/1024:.2f} GB)")
        continue
    log(f"  Downloading {fname} (~10.4 GB, please wait)...")
    try:
        p = hf_hub_download(
            "netflix/void-model", fname,
            local_dir=str(VOID_CKPT),
            force_download=False
        )
        sz = Path(p).stat().st_size
        log(f"  [ok] {fname} ({sz/1024/1024/1024:.2f} GB)")
    except Exception as e:
        log(f"  [FAIL] {fname}: {e}")


log("=" * 60)
log("Download complete. Summary:")
log(f"  Code:        {VOID_CODE}")
log(f"  Base model:  {COGVX_BASE}")
log(f"  Checkpoints: {VOID_CKPT}")

# Verify key files
for f in [VOID_CKPT / "void_pass1.safetensors",
          COGVX_BASE / "vae/diffusion_pytorch_model.safetensors",
          COGVX_BASE / "text_encoder/model-00001-of-00002.safetensors"]:
    if f.exists():
        log(f"  [EXISTS] {f.name}: {f.stat().st_size/1024/1024/1024:.2f} GB")
    else:
        log(f"  [MISSING] {f}")
