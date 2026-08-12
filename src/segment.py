"""Temporal segmentation: where the camera cuts, and where the play starts.

Both exist because gameplay video is not a continuous observation of a game.
It is a directed sequence with cuts, replays and dead time, and possession is
only defined during part of it.
"""

from __future__ import annotations

import cv2
import numpy as np

CUT_THRESHOLD = 0.45
SNAP_SPEED = 1.5      # px/frame, mean over visible players
SNAP_SUSTAIN = 15     # frames the threshold must hold


def detect_cuts(video, upto_s=None):
    """Frame indices where the camera cuts.

    Every cut destroys track identity, so a carrier path spanning one is
    meaningless. We find cuts first and refuse to reason across them, rather
    than emitting a path that changes carrier whenever the camera changes shot.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    limit = int(upto_s * fps) if upto_s else None

    cuts, prev, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (limit is not None and i > limit):
            break
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([small], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev is not None:
            if cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_THRESHOLD:
                cuts.append(i)
        prev, i = hist, i + 1
    cap.release()
    return cuts, fps


def clamp_to_shot(start_s, duration_s, cuts, fps):
    """Trim a requested window so it lies inside a single continuous shot."""
    a, b = start_s, start_s + duration_s
    for c in cuts:
        t = c / fps
        if a < t < b:
            b = t - 1.0 / fps
            break
    return a, max(0.0, b - a)


def detect_snap(vel, n_frames):
    """First frame where the whole field starts moving.

    Before the snap nobody is the ball carrier, so scoring those frames yields
    noise -- and that noise is what makes an unguarded carrier path flicker.
    Rather than smoothing it away afterwards we decline to answer until the
    play is live.

    Returns (snap_frame_or_None, mean_speed_per_frame).
    """
    mean_speed = []
    for t in range(n_frames):
        v = [np.linalg.norm(vel[tid][t]) for tid in vel if t in vel[tid]]
        mean_speed.append(float(np.mean(v)) if v else 0.0)

    run = 0
    for t, s in enumerate(mean_speed):
        run = run + 1 if s > SNAP_SPEED else 0
        if run >= SNAP_SUSTAIN:
            return t - SNAP_SUSTAIN + 1, mean_speed
    return None, mean_speed
