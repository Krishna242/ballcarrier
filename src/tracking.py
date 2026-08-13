"""Player detection, association, and the trajectories they produce.

Nothing here is novel and none of it should be. This stage is assembled from
pretrained components so that effort goes to the carrier question instead.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0          # COCO

# Raised from 960 and lowered from 0.35. At the old settings a player being
# tackled -- arms and legs tangled with another player, the exact moment
# possession is most in doubt -- frequently went undetected, so the frame lost
# its carrier label entirely. Chosen by looking at whether the ball carrier in
# a pile is detected at all, never by looking at downstream accuracy.
CONF = 0.20
IMGSZ = 1280
SMOOTH_WIN = 5            # frames at 60fps, centred velocity estimate
MIN_TRACK_LEN = 8         # drop blink-in detections
REF_FPS = 60.0            # both above are durations; 30fps clips halve them


def collect_tracks(video, start_s, duration_s, weights):
    """Run detection + tracking over a clip.

    Returns (frames, per_frame_boxes, fps) where per_frame_boxes[t] maps
    track id -> xyxy box.
    """
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    n_frames = int(duration_s * fps)

    frames, per_frame = [], []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(
            frame, persist=True, classes=[PERSON_CLASS], conf=CONF,
            imgsz=IMGSZ, tracker="bytetrack.yaml", verbose=False,
        )[0]

        boxes = {}
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            for tid, box in zip(ids, xyxy):
                boxes[int(tid)] = box
        frames.append(frame)
        per_frame.append(boxes)

    cap.release()
    return frames, per_frame, fps


def build_trajectories(per_frame, fps=REF_FPS):
    """Per-track position and smoothed velocity over time.

    Position is the bottom-centre of the box -- the player's ground contact
    point -- which is far more stable than the box centre when players are
    occluded from the waist up in a pile.
    """
    pos = defaultdict(dict)
    for t, boxes in enumerate(per_frame):
        for tid, (x1, y1, x2, y2) in boxes.items():
            pos[tid][t] = np.array([(x1 + x2) / 2.0, y2], dtype=np.float64)

    smooth = max(int(round(SMOOTH_WIN * fps / REF_FPS)), 1)
    min_len = max(int(round(MIN_TRACK_LEN * fps / REF_FPS)), 3)
    pos = {tid: d for tid, d in pos.items() if len(d) >= min_len}

    vel = defaultdict(dict)
    for tid, d in pos.items():
        ts = sorted(d)
        for i, t in enumerate(ts):
            lo = ts[max(0, i - smooth)]
            hi = ts[min(len(ts) - 1, i + smooth)]
            span = hi - lo
            vel[tid][t] = (d[hi] - d[lo]) / span if span > 0 else np.zeros(2)
    return pos, dict(vel)
