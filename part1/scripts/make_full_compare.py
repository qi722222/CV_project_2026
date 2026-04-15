import cv2, numpy as np, os
from pathlib import Path

def read_jpg_frames(jpg_dir, target_wh):
    files = sorted(p for p in Path(jpg_dir).iterdir() if p.suffix.lower() == '.jpg')
    frames = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None: continue
        if (img.shape[1], img.shape[0]) != target_wh:
            img = cv2.resize(img, target_wh)
        frames.append(img)
    return frames

def read_video_frames(path, target_wh):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        if (f.shape[1], f.shape[0]) != target_wh:
            f = cv2.resize(f, target_wh)
        frames.append(f)
    cap.release()
    return frames

def make_grid(sources, labels, output_path, target_wh=(480, 270), fps=30):
    w, h = target_wh
    all_frames = []
    for name, src in sources:
        if os.path.isdir(src):
            fr = read_jpg_frames(src, target_wh)
        else:
            fr = read_video_frames(src, target_wh)
        print(f'  {name}: {len(fr)} frames from {src}')
        all_frames.append(fr)

    n = min(len(fr) for fr in all_frames)
    print(f'  using {n} frames (min across sources)')

    out_w = w * len(sources)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (out_w, h)
    )

    for i in range(n):
        row = [fr[i] for fr in all_frames]
        stacked = np.hstack(row)
        for j, lbl in enumerate(labels):
            cv2.putText(stacked, lbl, (j*w + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(stacked, lbl, (j*w + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(stacked)
    writer.release()
    print(f'wrote {output_path}: {n} frames, {out_w}x{h}')
    print()

# ============ bmx-trees ============
print('=== bmx-trees ===')
make_grid(
    sources=[
        ('orig',   '/data2/shared/project3/bmx-trees'),
        ('part1',  'outputs/sweep/bmx_C_k15_m25_w25.mp4'),
        ('part2',  '/data2/jli657/project3/part2/outputs/bmx-trees/bmx-trees/inpaint_out.mp4'),
    ],
    labels=['Original', 'Part 1 (C: k15 m2.5 w25)', 'Part 2 (SAM2+ProPainter)'],
    output_path='outputs/sweep/bmx_full_compare.mp4',
)

# ============ tennis ============
print('=== tennis ===')
make_grid(
    sources=[
        ('orig',   '/data2/shared/project3/tennis'),
        ('part1',  'outputs/sweep/tennis_C_k15_c02_w20.mp4'),
        ('part2',  '/data2/jli657/project3/part2/outputs/tennis/tennis/inpaint_out.mp4'),
    ],
    labels=['Original', 'Part 1 (C: k15 c0.2 w20)', 'Part 2 (SAM2+ProPainter)'],
    output_path='outputs/sweep/tennis_full_compare.mp4',
)
