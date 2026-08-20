"""Harvesting carrier labels from the game's on-screen indicator.

NFL Blitz draws a reticle beneath the player under user control. If that
reticle can be found reliably, it yields a ground-truth carrier label on every
frame, automatically, for as much footage as we care to record -- removing the
manual annotation cost that normally dominates this problem.

STATUS: validated by inspection, on both game generations. Thirty-six
uniformly random labelled frames were drawn at native resolution and looked at
one by one (`scripts/audit_sample.py`). On every live-play frame where the
ball was visible, the reticle sat on the player holding it. The label is
therefore trustworthy *while a play is running*, and only then -- before the
snap it marks a selected player who does not have the ball, and after the
whistle it marks whoever was tackled. `evaluate.play_mask` excludes both.

Getting there took four fixes, and every one of them was found by looking at
frames rather than at a metric, because each failure produced confident output:

  - colour alone locked onto the TURBO meter;
  - colour plus "and it moves" locked onto the animating scoreboard;
  - anchoring to a player's feet still admitted the blue line of scrimmage,
    which the game paints across the turf and which crosses every player's
    feet by construction -- hence MAX_WIDTH_RATIO;
  - `static_occupancy` suppression had never once fired, because its threshold
    was 0.60 and nothing in the frame exceeds 0.592.

`validate` remains as a guard against the second of those. It is necessary and
nowhere near sufficient: it only asks whether the detection moves.
"""

from __future__ import annotations

import cv2
import numpy as np

# Saturated blue of the controlled-player reticle, in HSV.
BLUE_LO = (100, 150, 120)
BLUE_HI = (125, 255, 255)

# Lit in >20% of frames -> interface furniture, not a reticle.
#
# This was 0.60, which suppressed nothing: the highest occupancy anywhere in
# 9,764 frames is 0.592, so the mask was empty and the filter was inert from
# the day it was written. Measured, the separation is wide and the threshold
# is not delicate -- genuine reticle detections sit at occupancy 0.03 at the
# 90th percentile, because a reticle that follows a running player is never
# over the same pixel for long, while the TURBO bar reaches 0.59 and every
# pixel above 0.20 in the whole frame belongs to it.
STATIC_FRAC = 0.20

# Size and offset thresholds below are quoted at 720p and scaled by frame
# height at use. The 1997 arcade capture is 1356x1016, where a reticle is half
# again as wide and twice the area, and fixed pixel counts silently reject
# every one of them. Ratios and aspects are already scale-free and are not
# scaled.
REF_H = 720.0
MIN_AREA = 150         # at 720p; scales with the square of the height
MAX_AREA = 6000
MIN_ASPECT = 1.6       # the reticle is a wide, flat ellipse
MAX_ASPECT = 4.5       # ...but an ellipse, not a bar. See below.
MAX_WIDTH_RATIO = 2.5  # ...drawn around one player, so scaled to that player


def scale_of(frame):
    return frame.shape[0] / REF_H


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


FOOT_DY_ABOVE = 35     # px at 720p the reticle may sit above the box bottom
FOOT_DY_BELOW = 45     # ...and below it
FOOT_DX_PAD = 25       # horizontal slack around the box


def _at_a_players_feet(cx, cy, boxes, s=1.0):
    """The box this candidate sits at the feet of, or None.

    This is the constraint no HUD element can satisfy, and it is what the
    earlier colour-and-motion filters lacked. Requiring only 'saturated blue'
    matched the TURBO meter; adding 'and it moves' then matched the animating
    scoreboard. Anchoring to a player box rules out both, because interface
    furniture is not drawn under a tracked person.

    It does not rule out everything, which is why the caller applies shape
    tests to what this returns. Two blue things in this game *do* pass the
    anchor test: the line of scrimmage the game paints across the turf, which
    crosses every player's feet by construction, and the TURBO bar whenever a
    player happens to run in front of it.
    """
    for (x1, _, x2, y2) in boxes.values():
        if x1 - FOOT_DX_PAD * s <= cx <= x2 + FOOT_DX_PAD * s and \
           y2 - FOOT_DY_ABOVE * s <= cy <= y2 + FOOT_DY_BELOW * s:
            return (x1, x2)
    return None


CLOSE_W = 60           # px at 720p; horizontal reach of the gap-bridging close
CLOSE_H = 7            # ...and vertical, kept small on purpose


def _bridge_segments(m, s):
    """Reconnect a reticle drawn as separate arcs rather than a solid ellipse.

    Some captures render the indicator as a checkered ring -- blue arcs
    alternating with the other team colour -- and the colour mask then yields
    three or four small crescents where the shape tests expect one ellipse.
    On an eighteen-minute arcade capture that dropped the hit rate to 0.8%.

    The kernel is wide and short because the thing being rebuilt is a flat
    ellipse: bridging horizontally joins arcs of the same ring, while staying
    short vertically avoids welding the ring onto the player standing in it,
    whose navy uniform also falls inside the reticle blue.
    """
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(int(CLOSE_W * s) | 1, 3), max(int(CLOSE_H * s) | 1, 3)))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def _best_candidate(m, s, boxes, prefer_flat=False):
    """Best reticle candidate in a mask, or None.

    `prefer_flat` ranks by aspect instead of area. It is used only on the
    segmented-ring fallback, where the competing blue object is the numbered
    badge that floats beside the controlled player. Both mark the same player,
    but the badge sits at chest height while the reticle sits at the feet, and
    the track assignment downstream matches on the box *bottom* -- so picking
    the badge quietly costs labels. They separate cleanly on shape: the badge
    measures about 1.6 wide-to-tall, the ring about 4.2.
    """
    lo_area, hi_area = MIN_AREA * s * s, MAX_AREA * s * s
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if not (lo_area <= area <= hi_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        if boxes is not None:
            anchor = _at_a_players_feet(cx, cy, boxes, s)
            if anchor is None:
                continue
            # The reticle is drawn around one player, so it is about as wide as
            # that player. The line of scrimmage is as wide as the field, and
            # passes the anchor test at every player it crosses.
            if w > MAX_WIDTH_RATIO * max(anchor[1] - anchor[0], 1):
                continue
        rank = aspect if prefer_flat else area
        if best is None or rank > best[0]:
            best = (rank, cx, cy, w, h)
    return None if best is None else best[1:]


def find_reticle(frame, static_mask, boxes=None):
    """Locate the reticle in one frame. Returns (cx, cy, w, h) or None.

    `boxes` maps track id -> xyxy for this frame. It is optional only so the
    function can be exercised in isolation; in normal use it should always be
    supplied, because the player-anchoring test is what makes this detector
    trustworthy rather than merely confident.

    The segmented-ring pass runs only when the ordinary one finds nothing, so
    footage where the indicator is already a solid ellipse is unaffected --
    the fallback can add detections but can never change an existing one.
    """
    m = colour_mask(frame)
    m[static_mask > STATIC_FRAC] = 0
    s = scale_of(frame)

    hit = _best_candidate(m, s, boxes)
    if hit is not None:
        return hit
    return _best_candidate(_bridge_segments(m, s), s, boxes, prefer_flat=True)


def harvest(frames, per_frame_boxes=None):
    """Per-frame reticle position across a clip. None where not found."""
    static = static_occupancy(frames)
    out = []
    for t, f in enumerate(frames):
        boxes = per_frame_boxes[t] if per_frame_boxes is not None else None
        out.append(find_reticle(f, static, boxes))
    return out, static


def assign_to_track(reticle, per_frame_boxes, scale=1.0):
    """Map each reticle hit to the player track it sits beneath.

    The reticle is drawn centred on the player's ground contact point, so the
    match is to the nearest box *bottom-centre* -- the same point the rest of
    the pipeline uses as a player's position.

    The earlier rule accepted any box horizontally containing the reticle and
    then minimised vertical distance alone. Audit of a new source found that
    failing exactly where it matters: with the reticle sitting between two
    players, a small distant box whose bottom happened to align in y won over
    the large near one the reticle was actually drawn under. Horizontal
    distance carries real information here and was being discarded once the
    containment test passed.
    """
    pad, max_d = 20 * scale, 60 * scale
    labels = []
    for t, r in enumerate(reticle):
        if r is None:
            labels.append(None)
            continue
        cx, cy, _, _ = r
        best, best_d = None, 1e9
        for tid, (x1, _, x2, y2) in per_frame_boxes[t].items():
            if not (x1 - pad <= cx <= x2 + pad):
                continue
            # Vertical distance still dominates -- the reticle is at the feet,
            # not the waist -- but ties now break toward the nearer player.
            d = abs(y2 - cy) + 0.5 * abs((x1 + x2) / 2.0 - cx)
            if d < best_d:
                best, best_d = tid, d
        labels.append(best if best_d < max_d else None)
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
