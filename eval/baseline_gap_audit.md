# Baseline Gap Audit (Part1/Part2 on DAVIS + WildVideo)

## Scope
- Project root: `/home/jli657/my_storage2_1T/project3`
- Goal: verify whether Part1/Part2 baselines on DAVIS and WildVideo are fully runnable and archived for final report evidence.

## Current Status
- `eval` layer exists and is consistent:
  - `eval/davis_eval_targets.yaml`
  - `eval/eval_davis_masks.py`
  - `eval/results_davis_masks.csv`
- Part1/Part2 code exists, but runtime outputs are missing under current root:
  - no `part1/outputs/*`
  - no `part2/masks_cache/*`
  - no `part2/outputs/*`
- ControlNet output folders currently only include `manifest.json` in this root.

## Data Availability Check
- DAVIS under `/home/jli657/shared_data/project3/DAVIS` currently contains:
  - `README.md`, `SOURCES.md`, `ImageSets/*`
  - no discovered frame/annotation files under:
    - `DAVIS/JPEGImages/480p/*/*.jpg`
    - `DAVIS/Annotations/480p/*/*.png`
- WildVideo assets are not discovered under current `/home/jli657` project paths.

## Impact
- Cannot reproduce or extend Part1/Part2 baseline runs in this root now.
- Existing metric files in `eval/` cannot be refreshed or strictly validated against current disk outputs.
- Part3 high-standard claims are not yet backed by a complete baseline evidence chain in this root.

## Required Before Running Baselines
1. Provide accessible DAVIS data with both:
   - `JPEGImages/480p/<sequence>/*.jpg`
   - `Annotations/480p/<sequence>/*.png`
2. Provide at least one WildVideo source (mp4 or extracted frames).
3. Confirm a canonical data root to avoid path drift in scripts/configs.

## Execution Order Once Data Is Ready
1. Run Part1 baselines on target sequences (DAVIS + WildVideo).
2. Run Part2 masks and ProPainter outputs on same sequences.
3. Re-run `eval/eval_davis_masks.py` and regenerate report assets.
4. Freeze a baseline index table:
   - sequence
   - method (Part1/Part2)
   - output paths
   - metrics
   - visualization files

## Definition of Done for Baseline Closure
- At least 5 DAVIS sequences reproducibly run for Part1 and Part2.
- At least 1 WildVideo reproducibly run for Part1 and Part2.
- Every sequence has:
  - command record
  - output video/masks
  - evaluable artifacts
  - report-ready references
