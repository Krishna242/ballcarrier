"""Metrics for the carrier question, and the evaluation domain they apply to.

Three decisions are made here, and each of them can move a headline number by
more than any modelling choice downstream.

**Where possession is defined.** Gameplay video contains menus and frozen
banners. A prediction on a play-select screen is neither right nor wrong, so
those frames are excluded rather than scored.

**What counts as chance.** "42% accurate" means nothing without the number of
players it was choosing between. Every accuracy here is reported against the
per-frame reciprocal of the candidate count, computed on exactly the same
frames.

**What a possession change is.** The obvious definition -- the label's track id
changed -- is wrong, because a multi-object tracker reassigns ids to the same
player constantly, and every one of those churns would be counted as a turnover.
Possession changes are therefore defined spatially: the on-screen indicator
jumps from one place to a distant one. That definition does not depend on
track identity at all, which is the only way to measure possession change and
identity stability as separate things.
"""

from __future__ import annotations

import numpy as np

FIELD_GREEN_MIN = 0.15    # below this the field is not on camera
JUMP_PX = 150             # at 720p; a displacement that means another player
REF_H = 720.0
MATCH_TOL = 30            # frames a predicted change may miss a true one by

PLAY_SPEED = 1.2          # px/frame at 720p/60fps, mean over tracked players
PLAY_DILATE = 10          # frames of slack at each end of a live stretch
REF_FPS = 60.0


def scales(h=REF_H, fps=REF_FPS):
    """Spatial and temporal conversion factors for a clip.

    Two axes change between the EA-era 1280x720/60fps footage these constants
    were read off and the 1997 arcade captures, and they pull opposite ways.
    A bigger frame means more pixels per metre; a lower frame rate means more
    pixels per *frame* at the same real speed. A threshold in px/frame needs
    both, a threshold in pixels needs only the first, and a window measured in
    frames needs only the second -- so they are returned separately rather
    than collapsed into one fudge factor.
    """
    space = h / REF_H
    return {"space": space,
            "speed": space * (REF_FPS / max(fps, 1e-6)),
            "frames": max(fps, 1e-6) / REF_FPS}


# ---------------------------------------------------------------- domain

def field_mask(green, n, frame_a):
    """Frames where the field is on camera at all.

    Excludes the play-select screen, which yields both detections and reticle
    candidates and belongs in no accuracy number.
    """
    g = green[frame_a:frame_a + n]
    return np.array([bool(g[t] >= FIELD_GREEN_MIN) if t < len(g) else False
                     for t in range(n)])


def play_mask(vel, n, scale=1.0, frames_scale=1.0):
    """Frames where a play is actually running.

    Audit of a random sample of harvested labels found the reticle sitting on
    the ball carrier whenever a play was live, and sitting on a *pre-snap
    selected* player -- who does not have the ball, the centre does -- before
    the snap, and on a tackled player after the whistle. Scoring those frames
    measures agreement with a label that is not answering the carrier question.

    Liveness is decided from aggregate player motion, which says nothing about
    *which* player is the carrier, so it does not leak into any method's score.
    """
    mean_speed = np.zeros(n)
    for t in range(n):
        v = [np.linalg.norm(vel[tid][t]) for tid in vel if t in vel[tid]]
        mean_speed[t] = float(np.mean(v)) if v else 0.0

    live = mean_speed > PLAY_SPEED * scale
    dilate = max(int(round(PLAY_DILATE * frames_scale)), 1)
    out = live.copy()
    for t in np.flatnonzero(live):          # bridge brief dips within a play
        out[max(0, t - dilate):t + dilate + 1] = True
    return out


def live_mask(green, vel, n, frame_a, scale=1.0, frames_scale=1.0):
    """Frames on which possession is defined: field visible and play running."""
    m = field_mask(green, n, frame_a)
    if vel is None:
        return m
    return m & play_mask(vel, n, scale, frames_scale)


def candidate_counts(per_frame):
    return np.array([len(b) for b in per_frame])


# ---------------------------------------------------------------- accuracy

def frame_accuracy(pred, labels, mask):
    """Per-frame agreement, with the chance rate for the same frames.

    Chance is the mean of 1/candidates rather than 1/mean-candidates: the
    frames with few visible players are easy and common, and averaging the
    reciprocal is what a uniform guesser would actually score.
    """
    idx = [t for t in range(len(labels))
           if mask[t] and labels[t] is not None and pred[t] is not None]
    if not idx:
        return {"n": 0, "accuracy": float("nan"), "chance": float("nan")}
    hits = sum(1 for t in idx if pred[t] == labels[t])
    return {"n": len(idx), "accuracy": hits / len(idx)}


def chance_rate(n_candidates, labels, mask):
    """Score a uniform guesser over exactly the candidates a method ranks.

    `n_candidates[t]` must be the size of the set the predictor chose from on
    that frame, not the number of boxes the detector emitted. Those differ
    once short tracks are filtered out, and on the arcade clips they differ by
    a factor of two -- which is the whole difference between reporting four
    times chance and reporting twice chance.
    """
    idx = [t for t in range(len(labels))
           if mask[t] and labels[t] is not None and n_candidates[t] > 0]
    if not idx:
        return float("nan")
    return float(np.mean([1.0 / n_candidates[t] for t in idx]))


# ------------------------------------------------- possession changes

def true_changes(reticle, mask, labels=None, scale=1.0):
    """Frames where the indicator jumps to a different player.

    Consecutive *detected* positions are compared, so a gap in detection does
    not by itself create an event; but a long gap plus a large jump does, which
    is correct -- that is what a pass looks like.

    The two extra conditions come from auditing the events this produced with
    displacement alone. Roughly a third were artefacts: the reticle-blue
    control badge that floats beside the player, the TURBO bar along the
    bottom, and jumps of ninety-odd pixels that never left the same player's
    box. Requiring a jump wider than a player, and requiring both ends to land
    on two *different* tracked players, removes that class without removing
    any handoff or completed pass in the audited sample.
    """
    jump = JUMP_PX * scale
    events, prev = [], None
    for t, r in enumerate(reticle):
        if not mask[t] or r is None:
            continue
        if labels is not None and labels[t] is None:
            continue
        if prev is not None:
            d = np.hypot(r[0] - prev[1][0], r[1] - prev[1][1])
            same = labels is not None and labels[t] == labels[prev[0]]
            if d > jump and not same:
                events.append(t)
        prev = (t, r)
    return events


def pred_changes(pred, per_frame, mask, scale=1.0):
    """Frames where the prediction moves to a spatially distant player.

    Deliberately symmetric with `true_changes`: a predicted change is only
    counted when the new carrier is somewhere else on the field, so a tracker
    id churn on a stationary player is not scored as a turnover on either side
    of the comparison.
    """
    jump = JUMP_PX * scale
    events, prev = [], None
    for t in range(len(pred)):
        if not mask[t] or pred[t] is None or pred[t] not in per_frame[t]:
            continue
        x1, _, x2, y2 = per_frame[t][pred[t]]
        p = np.array([(x1 + x2) / 2.0, y2])
        if prev is not None and pred[t] != prev[0]:
            if np.hypot(*(p - prev[1])) > jump:
                events.append(t)
        prev = (pred[t], p)
    return events


def match_events(true_ev, pred_ev, tol=MATCH_TOL, frames_scale=1.0):
    """Greedy one-to-one matching of predicted to true events within `tol`."""
    tol = max(int(round(tol * frames_scale)), 1)
    unused = list(pred_ev)
    tp = 0
    for t in true_ev:
        best, bd = None, tol + 1
        for p in unused:
            d = abs(p - t)
            if d <= tol and d < bd:
                best, bd = p, d
        if best is not None:
            unused.remove(best)
            tp += 1
    fp, fn = len(pred_ev) - tp, len(true_ev) - tp
    prec = tp / max(tp + fp, 1e-9) if pred_ev else float("nan")
    rec = tp / max(tp + fn, 1e-9) if true_ev else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec and prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "n_true": len(true_ev), "n_pred": len(pred_ev)}


# ------------------------------------------------- tracker identity

def identity_churn(reticle, labels, mask, scale=1.0):
    """How often the carrier's *track id* changes while the carrier does not.

    This bounds what "track the ball carrier" can mean. If the underlying
    tracker hands the same running player three different ids in a second,
    then no carrier method built on those ids can emit a stable identity,
    however well it picks the right box on each individual frame.
    """
    churn, held, prev = 0, 0, None
    for t in range(len(labels)):
        if not mask[t] or labels[t] is None or reticle[t] is None:
            continue
        if prev is not None:
            moved = np.hypot(reticle[t][0] - prev[2][0],
                             reticle[t][1] - prev[2][1])
            if moved <= JUMP_PX * scale:  # same player, physically
                held += 1
                if labels[t] != prev[1]:  # ...but a new id
                    churn += 1
        prev = (t, labels[t], reticle[t])
    return {"same_player_frames": held, "id_changes": churn,
            "churn_rate": churn / max(held, 1)}
