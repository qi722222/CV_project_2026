"""
generate_vlm_captions.py — Task 5 Phase 1: 生成 VLM captions

在 gdino_env 中运行（有 transformers），对 4 个 DAVIS 序列生成 BLIP caption，
同时应用 token_map 映射到 SAM3 prompt，保存到 JSON 文件供后续 SAM3 步骤使用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image

SEQUENCES = ["tennis", "blackswan", "horsejump-low", "koala"]

VIDEO_ROOTS = {
    "tennis": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
    "blackswan": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/blackswan",
    "horsejump-low": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/horsejump-low",
    "koala": "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/koala",
}

VLM_MODEL = "Salesforce/blip-image-captioning-base"
HF_CACHE = "/data3/jli657/hf_cache"
OUT_JSON = "/home/jli657/my_storage2_1T/project3/eval/vlm_captions_direction_b.json"


def get_first_frame(video_dir: str):
    p = Path(video_dir)
    frames = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
                    key=lambda x: x.stem)
    return frames[0] if frames else None


def caption_to_prompts(caption: str, seq_name: str) -> Tuple[List[str], str]:
    """Apply token_map logic: caption -> VLM tokens -> SAM3 prompts."""
    text = caption.lower().strip()
    tokens: List[str] = []

    if any(w in text for w in ("person", "man", "woman", "pedestrian", "athlete", "player", "rider")):
        tokens.append("person")
    if any(w in text for w in ("bicycle", "bike", "cycle")):
        tokens.append("bicycle")
    if any(w in text for w in ("tennis", "racket")):
        if "person" not in tokens:
            tokens.append("person")
        tokens.append("tennis racket")
        if "ball" in text:
            tokens.append("tennis ball")
    if any(w in text for w in ("car", "vehicle", "automobile")):
        tokens.append("car")
    if any(w in text for w in ("horse", "equestrian")):
        tokens.append("horse")
    if any(w in text for w in ("bird", "swan")):
        tokens.append("bird")
    if any(w in text for w in ("koala",)):
        tokens.append("koala")
    if any(w in text for w in ("bear",)) and "koala" not in text:
        tokens.append("bear")
    if any(w in text for w in ("bag", "backpack", "handbag")):
        tokens.append("bag")

    # Dedup preserve order
    uniq: List[str] = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)

    if not uniq:
        print(f"  [WARN] No tokens mapped for '{caption}', using fallback 'object'")
        uniq = ["object"]

    return uniq, caption


def main():
    import os
    os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
    os.environ["HF_HOME"] = HF_CACHE

    from transformers import pipeline as hf_pipeline
    print(f"Loading VLM: {VLM_MODEL}")
    vlm_pipe = hf_pipeline("image-to-text", model=VLM_MODEL)
    print("VLM loaded.")

    results = {}
    for seq in SEQUENCES:
        video_dir = VIDEO_ROOTS.get(seq)
        if not video_dir or not Path(video_dir).exists():
            print(f"[skip] {seq}: video dir not found")
            continue

        first_frame = get_first_frame(video_dir)
        if first_frame is None:
            print(f"[skip] {seq}: no frames found")
            continue

        print(f"\n[{seq}] Running VLM on: {first_frame}")
        img = Image.open(str(first_frame)).convert("RGB")
        out = vlm_pipe(img, max_new_tokens=40)
        caption = str(out[0].get("generated_text", "")).lower().strip() if out else ""
        prompts, _ = caption_to_prompts(caption, seq)

        print(f"  Caption: '{caption}'")
        print(f"  Prompts: {prompts}")

        results[seq] = {
            "caption": caption,
            "vlm_prompts": prompts,
            "first_frame": str(first_frame),
        }

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {OUT_JSON}")
    return results


if __name__ == "__main__":
    results = main()
    print("\n=== VLM Caption Results ===")
    for seq, r in results.items():
        print(f"  {seq}: caption='{r['caption']}' -> prompts={r['vlm_prompts']}")
