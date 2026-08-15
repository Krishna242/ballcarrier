"""Localise the football in NFL Blitz frames.

A whole-frame colour search cannot name the carrier: the 1997 ball is a small
reddish-brown oval that collides with skin, red sleeves, and red jerseys, so
the loudest brown blob is usually not the ball. Association is done first
(learned model, with a pulled-back-camera fallback). This module then looks
for the ball *inside that player's hands* and only draws a marker when a
compact oval is actually there.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from . import dataset

REF_H = 720.0

# Dark, saturated football brown. Bright red sleeves sit above V~165.
BALL_LO = (0, 85, 60)
BALL_HI = (16, 255, 190)
BALL_LO2 = (170, 85, 60)
BALL_HI2 = (180, 255, 190)
SKIN_LO = (3, 35, 125)
SKIN_HI = (28, 175, 230)


def _scale(h):
    return h / REF_H


def _chest_crop(frame, box, pad=8):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    y2c = int(y1 + 0.72 * max(y2 - y1, 1))
    xa, ya = max(0, x1 - pad), max(0, y1 - 6)
    xb, yb = min(w, x2 + pad), min(h, y2c)
    crop = frame[ya:yb, xa:xb]
    return crop, xa, ya


def localize_in_box(frame, box):
    """Football point inside one player box, or None.

    Requires a compact dark-brown oval on the side of the torso, preferably
    next to skin. Tuned to miss rather than mark a sleeve as the ball.
    """
    crop, ox, oy = _chest_crop(frame, box)
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, BALL_LO, BALL_HI)
    brown = cv2.bitwise_or(brown, cv2.inRange(hsv, BALL_LO2, BALL_HI2))
    skin = cv2.inRange(hsv, SKIN_LO, SKIN_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    brown = cv2.morphologyEx(brown, cv2.MORPH_OPEN, k)
    skin_d = cv2.dilate(skin, k, iterations=1)
    ch, cw = crop.shape[:2]
    yy, xx = np.ogrid[:ch, :cw]
    chest = (yy > 0.10 * ch) & (yy < 0.72 * ch)
    cand = (brown > 0) & chest
    # Prefer blobs touching skin (ball in hands), but allow a miss on gloves.
    near_skin = cand & (skin_d > 0)
    if near_skin.sum() >= 12:
        cand = near_skin
    m = cand.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    s = _scale(frame.shape[0])
    min_a, max_a = 16 * s * s, 220 * s * s
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_a <= area <= max_a):
            continue
        if area > 0.06 * ch * cw:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        asp = bw / max(bh, 1)
        if not (0.35 <= asp <= 2.7):
            continue
        peri = cv2.arcLength(c, True)
        circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
        if circ < 0.32:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        side = abs(cx - cw / 2.0) / (cw / 2.0 + 1e-6)
        if side < 0.08:
            continue
        typical = 70.0 * s * s
        size_fit = 1.0 / (1.0 + abs(area - typical) / typical)
        score = (0.3 + min(side, 1.0)) * (0.4 + circ) * size_fit
        if best is None or score > best[0]:
            best = (score, ox + cx, oy + cy)
    if best is None or best[0] < 0.45:
        return None
    return (float(best[1]), float(best[2]), float(best[0]))


def localize_on_player(frame, boxes, tid):
    box = boxes.get(tid)
    if box is None:
        return None
    return localize_in_box(frame, box)


def detect_clip(video, per_frame, frame_a=0, path=None):
    """Per-frame ball point on `path[t]` if that player is known, else None.

    `path` is the carrier prediction. Searching every player independently
    is what produced the jersey false positives.
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_a))
    balls = []
    path = path or [None] * len(per_frame)
    for t, boxes in enumerate(per_frame):
        ok, frame = cap.read()
        if not ok:
            balls.append(None)
            continue
        tid = path[t] if t < len(path) else None
        balls.append(localize_on_player(frame, boxes, tid) if tid is not None
                     else None)
    cap.release()
    return {"balls": balls, "held": [{}] * len(balls)}


def load_or_detect(shot, cache, video, path=None):
    """Attach shot['ball'] from cache/ball.pkl, computing it if needed."""
    cache = Path(cache)
    pkl = cache / "ball.pkl"
    if pkl.exists() and path is None:
        with open(pkl, "rb") as fh:
            shot["ball"] = pickle.load(fh)
        return shot["ball"]
    video = Path(video)
    if not video.is_file():
        shot["ball"] = {"balls": [None] * shot["n"], "held": [{}] * shot["n"]}
        return shot["ball"]
    data = detect_clip(video, shot["d"]["per_frame"], shot["d"]["frame_a"],
                       path=path)
    if path is None:
        with open(pkl, "wb") as fh:
            pickle.dump(data, fh)
    shot["ball"] = data
    return data


def lowest_player(shot, t):
    cand = dataset.candidates(shot, t)
    if not cand:
        return None
    return max(cand, key=lambda tid: shot["pos"][tid][t][1])


def empty_box_rate(shot):
    n = max(shot["n"], 1)
    return sum(1 for t in range(shot["n"]) if not shot["d"]["per_frame"][t]) / n


# Clip 4 loses players when the camera pulls back (~15% empty frames).
# The other three clips sit near 0–1%. That is a zoom signal, not a label.
PULLBACK_EMPTY = 0.05


def camera_pulled_back(shot, thresh=PULLBACK_EMPTY):
    return empty_box_rate(shot) >= thresh


def blend_with_lowest(pred, shot, pan_thresh=None):
    """If this shot is a pulled-back camera, the carrier is lowest on screen.

    Per-frame pan gating was measured and lost accuracy: kickoff (clip 1)
    also has vertical camera motion, and lowest-on-screen is wrong there.
    Empty detections are what actually distinguishes the distance-change clip.
    """
    if not camera_pulled_back(shot):
        return list(pred)
    return [lowest_player(shot, t) for t in range(len(pred))]
