"""Carrier inference from trajectories alone.

The central claim of the project: who holds the ball is recoverable from how
players move, without detecting the ball. Nothing in this module looks at
pixels -- its only inputs are positions and velocities over time.
"""

from __future__ import annotations

import numpy as np

CONVERGE_FALLOFF = 220.0  # px; distance at which a defender's approach halves
SWITCH_PENALTY = 2.2      # Viterbi cost of changing carrier between frames

WEIGHTS = {"conv": 1.0, "speed": 0.6, "sep": 0.4}


def convergence(tid, t, pos, vel, present):
    """How hard everyone else is closing on this player.

    For each other visible player we project their velocity onto the unit
    vector pointing at the candidate and keep only the approaching component,
    weighted by proximity. The ball carrier is, structurally, the player a
    defence converges on -- that is what pursuit is.
    """
    p_i = pos[tid][t]
    total = 0.0
    for tj in present:
        if tj == tid or t not in vel[tj]:
            continue
        delta = p_i - pos[tj][t]
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            continue
        approach = float(np.dot(vel[tj][t], delta / dist))
        if approach > 0:
            total += approach / (1.0 + dist / CONVERGE_FALLOFF)
    return total


def separation(tid, t, pos, present):
    """Distance from the crowd centroid. Carriers break away from the pile;
    linemen stay buried in it."""
    pts = np.array([pos[tj][t] for tj in present])
    return float(np.linalg.norm(pos[tid][t] - pts.mean(axis=0)))


def score_frames(pos, vel, n_frames, weights=None):
    """Per-frame, per-track carrier score built only from motion.

    Cues are z-normalised within each frame so they combine on equal terms and
    so the score is comparable across frames with different play intensity.
    """
    w = weights or WEIGHTS
    scores = []
    for t in range(n_frames):
        present = [tid for tid in pos if t in pos[tid] and t in vel.get(tid, {})]
        if len(present) < 3:
            scores.append({})
            continue

        raw = {}
        for tid in present:
            raw[tid] = {
                "conv": convergence(tid, t, pos, vel, present),
                "sep": separation(tid, t, pos, present),
                "speed": float(np.linalg.norm(vel[tid][t])),
            }

        for key in ("conv", "sep", "speed"):
            vals = np.array([raw[tid][key] for tid in present])
            mu, sd = vals.mean(), vals.std() + 1e-6
            for tid in present:
                raw[tid][key + "_z"] = (raw[tid][key] - mu) / sd

        scores.append({
            tid: sum(w[k] * raw[tid][k + "_z"] for k in w) for tid in present
        })
    return scores


def viterbi(scores, switch_penalty=SWITCH_PENALTY):
    """Highest-scoring carrier path, penalised for switching.

    Frame-independent argmax makes the carrier flicker between players several
    times a second, which is not how possession behaves. The penalty encodes
    'possession is piecewise constant' as a structural prior rather than as a
    filter applied after the fact.
    """
    scored_idx = [t for t, s in enumerate(scores) if s]
    if not scored_idx:
        return [None] * len(scores)

    dp: list[dict[int, float]] = []
    back: list[dict[int, int | None]] = []

    for k, t in enumerate(scored_idx):
        cur: dict[int, float] = {}
        ptr: dict[int, int | None] = {}
        if k == 0:
            for tid, s in scores[t].items():
                cur[tid], ptr[tid] = s, None
        else:
            prev = dp[k - 1]
            alt_tid = max(prev, key=prev.__getitem__)
            alt_val = prev[alt_tid] - switch_penalty
            for tid, s in scores[t].items():
                stay_val = prev.get(tid)
                if stay_val is not None and stay_val >= alt_val:
                    cur[tid], ptr[tid] = stay_val + s, tid
                else:
                    cur[tid], ptr[tid] = alt_val + s, alt_tid
        dp.append(cur)
        back.append(ptr)

    path: list[int | None] = [None] * len(scores)
    tid = max(dp[-1], key=dp[-1].__getitem__)
    for k in range(len(scored_idx) - 1, -1, -1):
        path[scored_idx[k]] = tid
        tid = back[k].get(tid)
        if tid is None and k > 0:
            tid = max(dp[k - 1], key=dp[k - 1].__getitem__)
    return path


def enforce_dwell(path, min_dwell):
    """Absorb carrier runs shorter than `min_dwell` frames into their neighbour.

    The switch penalty makes a change expensive but never impossible, so the
    decoded path still contains one- and two-frame flickers, and each of those
    is scored as a possession change followed immediately by another one. That
    is what drives change precision to 0.05: the method announces hundreds of
    turnovers in a clip that contains three.

    A dwell floor states the thing the penalty only implies -- nobody carries
    the ball for a twelfth of a second. It is a hard constraint rather than a
    cost, so no amount of emission confidence can buy a flicker.

    Runs are absorbed shortest-first, because collapsing a long run into a
    short neighbour would be the wrong direction, and the pass is repeated
    until nothing changes: absorbing one run can leave its neighbours adjacent
    and equal, which merges them into a single longer run.
    """
    if min_dwell <= 1:
        return list(path)

    out = list(path)
    while True:
        runs, start = [], 0
        for i in range(1, len(out) + 1):
            if i == len(out) or out[i] != out[start]:
                runs.append((start, i, out[start]))
                start = i
        short = [r for r in runs
                 if r[2] is not None and (r[1] - r[0]) < min_dwell]
        if not short:
            return out

        a, b, _ = min(short, key=lambda r: r[1] - r[0])
        k = next(i for i, r in enumerate(runs) if r[0] == a)
        prev_run = runs[k - 1] if k > 0 else None
        next_run = runs[k + 1] if k + 1 < len(runs) else None

        # Prefer whichever neighbour is longer; a flicker between two settled
        # carriers should fall back to the one that was actually established.
        cand = [r for r in (prev_run, next_run) if r and r[2] is not None]
        if not cand:
            for i in range(a, b):
                out[i] = None
        else:
            fill = max(cand, key=lambda r: r[1] - r[0])[2]
            for i in range(a, b):
                out[i] = fill


def decision_margin(scores):
    """Median gap between the best and second-best candidate.

    The honest confidence measure for this method. A small margin means the
    heuristic is guessing even when it happens to be right.
    """
    gaps = []
    for s in scores:
        if len(s) >= 2:
            v = sorted(s.values(), reverse=True)
            gaps.append(v[0] - v[1])
    return float(np.median(gaps)) if gaps else 0.0
