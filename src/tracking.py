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
IMGSZ_CPU = 640           # 1280 plus YOLO11m hits the NMS time limit on CPU
SMOOTH_WIN = 5            # frames at 60fps, centred velocity estimate
MIN_TRACK_LEN = 8         # drop blink-in detections
REF_FPS = 60.0            # both above are durations; 30fps clips halve them


def track_kwargs():
    """Ultralytics .track() args, scaled down on CPU so harvest can finish."""
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    return {
        "persist": True,
        "classes": [PERSON_CLASS],
        "conf": CONF,
        "imgsz": IMGSZ if cuda else IMGSZ_CPU,
        "tracker": "bytetrack.yaml",
        "verbose": False,
        "device": "0" if cuda else "cpu",
        "max_det": 50,
    }


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
        res = model.track(frame, **track_kwargs())[0]

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


def _camera_shift(cam, lo, hi):
    """Compose the frame-to-frame transforms over (lo, hi] into one affine.

    `cam[t]` carries frame t-1 into frame t, so the displacement the camera
    contributed between `lo` and `hi` is the composition of everything in
    between. Missing entries compose as identity, which is the right default:
    an unestimated frame contributes no correction rather than a guessed one.
    """
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for t in range(lo + 1, hi + 1):
        B = cam[t] if 0 <= t < len(cam) else None
        if B is None:
            continue
        M = np.vstack([A, [0, 0, 1]])
        A = (np.vstack([B, [0, 0, 1]]) @ M)[:2]
    return A


def build_trajectories(per_frame, fps=REF_FPS, cam=None):
    """Per-track position and smoothed velocity over time.

    Position is the bottom-centre of the box -- the player's ground contact
    point -- which is far more stable than the box centre when players are
    occluded from the waist up in a pile.

    When `cam` is supplied, velocity is measured against where the *camera*
    would have carried the player had they stood still, so a pan no longer
    reads as every player accelerating at once. Positions are left in screen
    coordinates deliberately: a rigid camera move shifts all players equally,
    so relative positions -- separation, crowding, who is near whom -- are
    already invariant to it, and rewriting them would only add drift.
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
            if span <= 0:
                vel[tid][t] = np.zeros(2)
                continue
            start = d[lo]
            if cam is not None:
                A = _camera_shift(cam, lo, hi)
                start = (start @ A[:, :2].T) + A[:, 2]
            vel[tid][t] = (d[hi] - start) / span
    return pos, dict(vel)
