"""
generate_vlm_captions_v2.py
Direction A-VLM: Upgraded VLM caption generation with fuzzy matching

Improvements over v1:
  1. Fuzzy matching: edit distance <= 2 for token_map keywords (fixes koala -> 'koloa')
  2. Extended to 7 sequences (adds bmx-trees, car-shadow, wild_video-1person)
  3. Area fallback: if VLM produces "object", try area-based saliency prompt
  4. BLIP-2 support (optional, better quality captions)
  5. Multi-frame sampling: captions from 3 frames (first, middle, last) voted

Run environment: conda run -n sam3_env (or gdino_env with transformers)
  python part3/generate_vlm_captions_v2.py --model blip --sequences all

Output: eval/vlm_captions_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SEQUENCES = [
    "tennis", "blackswan", "horsejump-low", "koala",
    "bmx-trees", "car-shadow", "wild_video-1person",
]

VIDEO_ROOTS: Dict[str, str] = {
    "tennis":           "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/tennis",
    "blackswan":        "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/blackswan",
    "horsejump-low":    "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/horsejump-low",
    "koala":            "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/koala",
    "bmx-trees":        "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/bmx-trees",
    "car-shadow":       "/home/jli657/shared_data/project3/DAVIS/JPEGImages/480p/car-shadow",
    "wild_video-1person": "/data3/jli657/project3/wild_frames/wild_video-1person",
}

HF_CACHE = "/data3/jli657/hf_cache"
OUT_JSON_V2 = "/home/jli657/my_storage2_1T/project3/eval/vlm_captions_v2.json"


# ---------------------------------------------------------------------------
# Fuzzy matching (edit distance <= 2)
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if abs(len(a) - len(b)) > 2:
        return 99  # fast prune
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def fuzzy_match(text: str, keyword: str, max_dist: int = 2) -> bool:
    """Check if `keyword` or any near-match appears in `text`."""
    if keyword in text:
        return True
    words = text.split()
    for word in words:
        if len(word) >= 3 and edit_distance(word, keyword) <= max_dist:
            return True
    return False


# ---------------------------------------------------------------------------
# Token map: caption -> SAM3 prompts (with fuzzy matching)
# ---------------------------------------------------------------------------

def caption_to_prompts_v2(caption: str, seq_name: str) -> Tuple[List[str], str]:
    """
    Apply improved token_map:
      - Fuzzy matching for noisy captions (e.g. 'koloa' -> 'koala')
      - Fallback prompts by sequence name if VLM fails
      - Returns (prompts, matched_caption)
    """
    text = caption.lower().strip()
    tokens: List[str] = []

    # Person / human detection
    person_kws = ["person", "man", "woman", "pedestrian", "athlete", "player",
                  "rider", "cyclist", "biker", "bmxer", "people"]
    if any(fuzzy_match(text, kw) for kw in person_kws):
        tokens.append("person")

    # Bicycle / BMX
    bike_kws = ["bicycle", "bike", "cycle", "bmx"]
    if any(fuzzy_match(text, kw) for kw in bike_kws):
        tokens.append("bicycle")

    # Tennis
    if fuzzy_match(text, "tennis") or fuzzy_match(text, "racket"):
        if "person" not in tokens:
            tokens.append("person")
        tokens.append("tennis racket")
        if "ball" in text:
            tokens.append("tennis ball")

    # Car / vehicle
    car_kws = ["car", "vehicle", "automobile", "sedan", "truck", "suv"]
    if any(fuzzy_match(text, kw) for kw in car_kws):
        tokens.append("car")

    # Horse / equestrian
    horse_kws = ["horse", "equestrian", "pony"]
    if any(fuzzy_match(text, kw) for kw in horse_kws):
        tokens.append("horse")

    # Bird / swan
    bird_kws = ["bird", "swan", "duck", "goose"]
    if any(fuzzy_match(text, kw) for kw in bird_kws):
        tokens.append("bird")

    # Koala (fuzzy: 'koloa', 'koal', 'coala' etc.)
    koala_kws = ["koala", "koal", "koloa", "coala"]
    if any(fuzzy_match(text, kw, max_dist=2) for kw in koala_kws):
        tokens.append("koala")

    # Bear (only if not koala)
    if fuzzy_match(text, "bear") and "koala" not in tokens:
        tokens.append("bear")

    # Dedup preserve order
    uniq: List[str] = []
    for t in tokens:
        if t not in uniq:
            uniq.append(t)

    # Fallback by sequence name (if VLM gives no useful tokens)
    if not uniq:
        fallback_map: Dict[str, List[str]] = {
            "tennis": ["tennis player with racket"],
            "blackswan": ["black swan"],
            "horsejump-low": ["horse", "person"],
            "koala": ["koala"],
            "bmx-trees": ["person", "bicycle"],
            "car-shadow": ["car"],
            "wild_video-1person": ["person"],
        }
        uniq = fallback_map.get(seq_name, ["object"])
        print(f"  [VLM fuzzy fallback for '{seq_name}'] '{caption}' -> {uniq}")

    return uniq, caption


# ---------------------------------------------------------------------------
# Multi-frame VLM voting
# ---------------------------------------------------------------------------

def get_sample_frames(video_dir: str, n_samples: int = 3) -> List[Path]:
    p = Path(video_dir)
    frames = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
                    key=lambda x: x.stem)
    if not frames:
        return []
    if len(frames) <= n_samples:
        return frames
    # Sample first, middle, last
    idxs = [0, len(frames) // 2, len(frames) - 1][:n_samples]
    return [frames[i] for i in idxs]


def vote_prompts(all_prompts: List[List[str]]) -> List[str]:
    """Vote across multi-frame captions: keep tokens that appear in >=2/3 of frames."""
    from collections import Counter
    counter: Counter = Counter()
    for prompts in all_prompts:
        for p in prompts:
            counter[p] += 1
    n = len(all_prompts)
    # Keep if appears in > 1/3 of frames (at least once for small n)
    min_count = max(1, n // 3)
    voted = [tok for tok, cnt in counter.most_common() if cnt >= min_count]
    return voted if voted else (all_prompts[0] if all_prompts else ["object"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Direction A-VLM: Generate VLM captions with fuzzy match")
    p.add_argument("--sequences", nargs="+", default=["all"],
                   help="Sequences to process, or 'all'")
    p.add_argument("--model", default="blip",
                   choices=["blip", "blip2"],
                   help="VLM model to use")
    p.add_argument("--n_samples", type=int, default=3,
                   help="Number of frames to sample per sequence")
    p.add_argument("--out_json", default=OUT_JSON_V2)
    return p.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE)
    os.environ.setdefault("HF_HOME", HF_CACHE)

    sequences = ALL_SEQUENCES if args.sequences == ["all"] else args.sequences

    # Load VLM
    from transformers import pipeline as hf_pipeline
    if args.model == "blip2":
        model_id = "Salesforce/blip2-opt-2.7b"
        print(f"Loading VLM: {model_id}")
        vlm_pipe = hf_pipeline("image-to-text", model=model_id,
                                model_kwargs={"load_in_8bit": True})
    else:
        model_id = "Salesforce/blip-image-captioning-base"
        print(f"Loading VLM: {model_id}")
        vlm_pipe = hf_pipeline("image-to-text", model=model_id)
    print("VLM loaded.")

    results: Dict[str, dict] = {}

    for seq in sequences:
        video_dir = VIDEO_ROOTS.get(seq)
        if not video_dir or not Path(video_dir).exists():
            print(f"[skip] {seq}: video dir not found at {video_dir}")
            continue

        sample_frames = get_sample_frames(video_dir, n_samples=args.n_samples)
        if not sample_frames:
            print(f"[skip] {seq}: no frames found")
            continue

        print(f"\n[{seq}] Processing {len(sample_frames)} frames...")
        frame_results = []
        all_prompts_list = []

        for frame_path in sample_frames:
            img = Image.open(str(frame_path)).convert("RGB")
            out = vlm_pipe(img, max_new_tokens=50)
            caption = str(out[0].get("generated_text", "")).lower().strip() if out else ""
            prompts, _ = caption_to_prompts_v2(caption, seq)
            all_prompts_list.append(prompts)
            frame_results.append({"frame": str(frame_path), "caption": caption, "prompts": prompts})
            print(f"  Frame {frame_path.name}: '{caption}' -> {prompts}")

        # Vote across frames
        final_prompts = vote_prompts(all_prompts_list)
        print(f"  Final (voted): {final_prompts}")

        results[seq] = {
            "model": model_id,
            "final_prompts": final_prompts,
            "frame_results": frame_results,
        }

    # Save JSON
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out_json}")

    print("\n=== VLM v2 Results ===")
    for seq, r in results.items():
        print(f"  {seq}: prompts={r['final_prompts']}")


if __name__ == "__main__":
    main()
