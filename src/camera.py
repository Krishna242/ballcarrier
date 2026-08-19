"""Estimate how much of the apparent player motion is the camera moving.

Every velocity in this project was measured in screen pixels, which means a
camera pan makes all twenty-two players appear to accelerate at once --
including the ones standing still. Convergence, the feature the whole premise
rests on, is computed from those velocities, so a pan can manufacture the exact
signature the method is looking for and a pan away can erase it.

The evidence that this matters is in the clip names: the deliberate
"Camera Swing and Distance Change Test" is the weakest clip, and the static
"Easiest QB Test" is among the strongest.

What is corrected, and what is not
----------------------------------
A rigid camera move shifts every player by the same amount, so *relative*
positions between players survive it untouched. Separation and crowding
therefore need no correction. Velocity does not survive it, and neither does
anything derived from velocity.

So this module estimates the frame-to-frame background motion and exposes it as
a transform. Velocity then becomes

    v(t) = p(t) - A_t( p(t-1) )

where `A_t` maps the previous frame's coordinates into the current frame. That
is a *local* correction: it never accumulates, so there is no drift over a long
shot, which a stabilise-to-frame-zero approach would suffer from badly on a
thirty-second play.

Estimation tracks corner features on the turf with sparse optical flow, with
detected players masked out -- the field markings are high-contrast, rigid and
plentiful, and they are the only thing in frame that is guaranteed not to be
running.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

MAX_CORNERS = 400
QUALITY = 0.01
MIN_DIST = 12          # px at 720p between tracked corners
BOX_PAD = 14           # px at 720p grown around each player before masking
MIN_MATCHES = 12       # below this, refuse to estimate rather than guess
REF_H = 720.0


def _mask_players(shape, boxes, s=1.0):
    """255 where the background is, 0 over every detected player."""
    m = np.full(shape[:2], 255, np.uint8)
    pad = int(round(BOX_PAD * s))
    h, w = shape[:2]
    for (x1, y1, x2, y2) in boxes.values():
        a, b = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        c, d = min(int(x2) + pad, w), min(int(y2) + pad, h)
        if c > a and d > b:
            m[b:d, a:c] = 0
    return m


def estimate_pair(prev_gray, gray, prev_boxes, s=1.0):
    """Affine transform carrying the previous frame's points into this one.

    Returns a 2x3 matrix, or None when the background gives too little to go
    on -- a tight shot where players fill the frame, or a hard cut. Returning
    None is deliberate: a badly conditioned fit would inject motion that was
    never there, which is worse than the uncorrected value it replaces.
    """
    if cv2 is None or prev_gray is None:
        return None
    mask = _mask_players(prev_gray.shape, prev_boxes, s)
    p0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=MAX_CORNERS, qualityLevel=QUALITY,
        minDistance=max(int(MIN_DIST * s), 3), mask=mask, blockSize=7)
    if p0 is None or len(p0) < MIN_MATCHES:
        return None

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                         winSize=(21, 21), maxLevel=3)
    if p1 is None:
        return None
    st = st.reshape(-1).astype(bool)
    a, b = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
    if len(a) < MIN_MATCHES:
        return None

    A, inliers = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2000, confidence=0.99)
    if A is None or inliers is None or int(inliers.sum()) < MIN_MATCHES:
        return None
    return A


def apply(A, pts):
    """Map Nx2 points through a 2x3 affine. Identity when A is None."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if A is None:
        return pts
    return pts @ A[:, :2].T + A[:, 2]


def motion_magnitude(A, w, h):
    """How far this transform moves the frame centre, in pixels.

    A single scalar summarising 'how much did the camera do here', used to
    report how much of the footage is actually affected rather than assuming
    it matters everywhere.
    """
    if A is None:
        return 0.0
    c = np.array([[w / 2.0, h / 2.0]])
    return float(np.linalg.norm(apply(A, c) - c))


def estimate_shot(video, per_frame, frame_a, h=REF_H):
    """Per-frame affine transforms for one shot: index t maps t-1 -> t."""
    if cv2 is None:
        return [None] * len(per_frame)
    s = h / REF_H
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_a))

    out, prev_gray, prev_boxes = [], None, {}
    for t in range(len(per_frame)):
        ok, frame = cap.read()
        if not ok:
            out.append(None)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(None if prev_gray is None
                   else estimate_pair(prev_gray, gray, prev_boxes, s))
        prev_gray, prev_boxes = gray, per_frame[t]
    cap.release()
    return out
