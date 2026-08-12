"""Harvesting carrier labels from the game's on-screen indicator.

NFL Blitz draws a reticle beneath the player under user control. If that
reticle can be found reliably, it yields a ground-truth carrier label on every
frame, automatically, for as much footage as we care to record -- removing the
manual annotation cost that normally dominates this problem.

STATUS: unvalidated. A first attempt using colour alone failed by locking onto
the TURBO meter, a fixed heads-up-display element of a similar blue. The fix
here is `static_occupancy`: HUD elements sit at the same pixels frame after
frame, whereas the reticle moves with the player. We suppress pixels that are
almost always lit, then filter what remains by shape.

`validate` exists because of that failure. It refuses to report success when
the detected point does not move -- the precise way the first attempt fooled
itself.
"""

from __future__ import annotations

import cv2
import numpy as np

# Saturated blue of the controlled-player reticle, in HSV.
BLUE_LO = (100, 150, 120)
BLUE_HI = (125, 255, 255)

STATIC_FRAC = 0.60     # lit in >60% of frames -> treat as HUD furniture
MIN_AREA = 150
MAX_AREA = 6000
MIN_ASPECT = 1.6       # the reticle is a wide, flat ellipse


def colour_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, BLUE_LO, BLUE_HI)


def static_occupancy(frames):
    """Fraction of frames in which each pixel falls inside the colour mask.

    High values mark fixed interface elements. This is what distinguishes a
    scoreboard or a turbo bar from a reticle that follows a running player.
    """
    acc = None
    for f in frames:
        m = (colour_mask(f) > 0).astype(np.float32)
        acc = m if acc is None else acc + m
    return acc / max(len(frames), 1)


FOOT_DY_ABOVE = 35     # px the reticle may sit above a player's box bottom
FOOT_DY_BELOW = 45     # ...and below it
FOOT_DX_PAD = 25       # horizontal slack around the box


def _at_a_players_feet(cx, cy, boxes):
    """Is this candidate positioned where a reticle is drawn -- at the feet of
    a detected player?

    This is the constraint no HUD element can satisfy, and it is what the
    earlier colour-and-motion filters lacked. Requiring only 'saturated blue'
    matched the TURBO meter; adding 'and it moves' then matched the animating
    scoreboard. Anchoring to a player box rules out both, because interface
    furniture is not drawn under a tracked person.
    """
    for (x1, _, x2, y2) in boxes.values():
        if x1 - FOOT_DX_PAD <= cx <= x2 + FOOT_DX_PAD and \
           y2 - FOOT_DY_ABOVE <= cy <= y2 + FOOT_DY_BELOW:
            return True
    return False


def find_reticle(frame, static_mask, boxes=None):
    """Locate the reticle in one frame. Returns (cx, cy, w, h) or None.

    `boxes` maps track id -> xyxy for this frame. It is optional only so the
    function can be exercised in isolation; in normal use it should always be
    supplied, because the player-anchoring test below is what makes this
    detector trustworthy rather than merely confident.
    """
    m = colour_mask(frame)
    m[static_mask > STATIC_FRAC] = 0

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w / max(h, 1) < MIN_ASPECT:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        if boxes is not None and not _at_a_players_feet(cx, cy, boxes):
            continue
        if best is None or area > best[0]:
            best = (area, cx, cy, w, h)
    return None if best is None else best[1:]


def harvest(frames, per_frame_boxes=None):
    """Per-frame reticle position across a clip. None where not found."""
    static = static_occupancy(frames)
    out = []
    for t, f in enumerate(frames):
        boxes = per_frame_boxes[t] if per_frame_boxes is not None else None
        out.append(find_reticle(f, static, boxes))
    return out, static


def assign_to_track(reticle, per_frame_boxes):
    """Map each reticle hit to the player track it sits beneath.

    The reticle is drawn at the player's feet, so we score boxes by horizontal
    containment and vertical proximity of the box bottom.
    """
    labels = []
    for t, r in enumerate(reticle):
        if r is None:
            labels.append(None)
            continue
        cx, cy, _, _ = r
        best, best_d = None, 1e9
        for tid, (x1, _, x2, y2) in per_frame_boxes[t].items():
            if not (x1 - 20 <= cx <= x2 + 20):
                continue
            d = abs(y2 - cy)
            if d < best_d:
                best, best_d = tid, d
        labels.append(best if best_d < 60 else None)
    return labels


def validate(reticle):
    """Self-check that guards against the failure that fooled the first attempt.

    Reports detection rate and how much the detected point actually moves. A
    high rate with near-zero movement means we have locked onto static HUD
    furniture again, not the reticle -- so that case is reported as a failure
    however good the hit rate looks.
    """
    hits = [r for r in reticle if r is not None]
    rate = len(hits) / max(len(reticle), 1)
    if len(hits) < 2:
        return {"rate": rate, "x_std": 0.0, "y_std": 0.0, "plausible": False,
                "reason": "too few detections"}

    xs = np.array([h[0] for h in hits])
    ys = np.array([h[1] for h in hits])
    x_std, y_std = float(xs.std()), float(ys.std())

    moving = x_std > 15.0 or y_std > 15.0
    return {
        "rate": round(rate, 3),
        "x_std": round(x_std, 1),
        "y_std": round(y_std, 1),
        "plausible": bool(rate > 0.3 and moving),
        "reason": "ok" if moving else "detection does not move — likely static HUD element",
    }
