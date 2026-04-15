import cv2, numpy as np
from pathlib import Path

def hstack_videos(input_paths, output_path, labels=None):
    caps = [cv2.VideoCapture(str(p)) for p in input_paths]
    n_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30

    out_w = w * len(caps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, h))

    for i in range(n_frames):
        row = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            row.append(frame)
        stacked = np.hstack(row)
        if labels:
            for j, lbl in enumerate(labels):
                # 白字黑边,任何背景都能看清
                cv2.putText(stacked, lbl, (j*w + 15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(stacked, lbl, (j*w + 15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(stacked)

    for c in caps: c.release()
    writer.release()
    print(f'wrote {output_path}: {n_frames} frames, {out_w}x{h}')

if __name__ == '__main__':
    hstack_videos(
        ['outputs/sweep/bmx_A_k13.mp4',
         'outputs/sweep/bmx_B_k13_m25.mp4',
         'outputs/sweep/bmx_C_k15_m25_w25.mp4'],
        'outputs/sweep/bmx_compare.mp4',
        labels=['A: k=13', 'B: k=13 m=2.5', 'C: k=15 m=2.5 w=25'])

    hstack_videos(
        ['outputs/sweep/tennis_A_k13_c02.mp4',
         'outputs/sweep/tennis_B_k15_c015.mp4',
         'outputs/sweep/tennis_C_k15_c02_w20.mp4'],
        'outputs/sweep/tennis_compare.mp4',
        labels=['A: k=13 c=0.2', 'B: k=15 c=0.15', 'C: k=15 c=0.2 w=20'])
