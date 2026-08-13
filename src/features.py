"""Per-candidate features for the carrier question, in two separable groups.

The split is the experiment. The project's claim is that possession is
recoverable from *trajectories alone*, so trajectory features are kept strictly
apart from features that describe where the camera has put a player on screen.
Mixed together they would produce one accuracy number that could not be
attributed, and a broadcast camera that follows the ball leaks the answer
through framing -- the "nearest frame centre" baseline already scores well
above chance on nothing else.

Trained separately, the two groups answer two different questions: how much the
motion actually carries, and how much the camera was telling us all along.

Every feature is z-normalised within its frame. The absolute scale of speed or
convergence depends on how far the camera has zoomed, which changes shot to
shot; the ranking among players on one frame does not.
"""

from __future__ import annotations

import numpy as np

from . import carrier

NEAR_R = 150.0     # px at 720p, radius counted as "in the pile"
LOOKBACK = 12      # frames at 60fps (~0.2s) for displacement and acceleration
REF_FPS = 60.0     # LOOKBACK is a duration, so it is rescaled for 30fps clips

TRAJ = ["conv", "speed", "sep", "accel", "disp", "n_near", "d_nearest",
        "closing", "speed_rank", "conv_rank"]
POS = ["x_norm", "y_norm", "d_centre", "box_h", "box_area", "y_rank",
       "h_rank"]


def _z(vals):
    a = np.asarray(vals, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-6)


def _rank(vals):
    """Fractional rank in [0, 1]; scale-free by construction."""
    a = np.asarray(vals, dtype=np.float64)
    order = a.argsort().argsort().astype(np.float64)
    return order / max(len(a) - 1, 1)


def frame_features(t, present, pos, vel, per_frame, w=1280.0, h=720.0,
                   lookback=LOOKBACK):
    """Feature rows for every candidate visible on frame `t`.

    Returns (track_ids, traj_matrix, pos_matrix).
    """
    near_r = NEAR_R * (h / 720.0)
    p = {tid: pos[tid][t] for tid in present}
    v = {tid: vel[tid][t] for tid in present}
    pts = np.array([p[tid] for tid in present])
    centroid = pts.mean(axis=0)

    conv, speed, sep, accel, disp, n_near, d_near, closing = ([] for _ in range(8))
    x_norm, y_norm, d_centre, box_h, box_area = ([] for _ in range(5))

    for tid in present:
        conv.append(carrier.convergence(tid, t, pos, vel, present))
        speed.append(float(np.linalg.norm(v[tid])))
        sep.append(float(np.linalg.norm(p[tid] - centroid)))

        t0 = t - lookback
        if t0 in pos[tid]:
            disp.append(float(np.linalg.norm(p[tid] - pos[tid][t0])))
        else:
            disp.append(0.0)
        if t0 in vel.get(tid, {}):
            accel.append(float(np.linalg.norm(v[tid] - vel[tid][t0])))
        else:
            accel.append(0.0)

        d = np.linalg.norm(pts - p[tid], axis=1)
        d = d[d > 1e-6]
        n_near.append(float((d < near_r).sum()))
        d_near.append(float(d.min()) if len(d) else 0.0)

        # How fast the nearest others are closing, irrespective of how many
        # there are -- a carrier in the open field has one pursuer, not eight.
        rate = 0.0
        for tj in present:
            if tj == tid:
                continue
            delta = p[tid] - p[tj]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6 or dist > 2 * near_r:
                continue
            rate = max(rate, float(np.dot(v[tj], delta / dist)))
        closing.append(rate)

        x1, y1, x2, y2 = per_frame[t][tid]
        x_norm.append(float(((x1 + x2) / 2) / w))
        y_norm.append(float(y2 / h))
        d_centre.append(float(abs((x1 + x2) / 2 - w / 2) / (w / 2)))
        box_h.append(float(y2 - y1))
        box_area.append(float((x2 - x1) * (y2 - y1)))

    traj = np.column_stack([
        _z(conv), _z(speed), _z(sep), _z(accel), _z(disp),
        _z(n_near), _z(d_near), _z(closing),
        _rank(speed), _rank(conv),
    ])
    posm = np.column_stack([
        np.asarray(x_norm), np.asarray(y_norm), np.asarray(d_centre),
        _z(box_h), _z(box_area), _rank(y_norm), _rank(box_h),
    ])
    return list(present), traj, posm


def build_dataset(pos, vel, per_frame, labels, mask, n, shot_id,
                  w=1280.0, h=720.0, fps=REF_FPS):
    """Flatten a shot into candidate rows, keeping frame grouping intact.

    `group` records which frame each row came from, because the prediction is
    an argmax *within a frame*, not an independent decision per row -- the
    evaluation and the training objective both need that structure.
    """
    lookback = max(int(round(LOOKBACK * fps / REF_FPS)), 2)
    X_t, X_p, y, group, tids = [], [], [], [], []
    for t in range(n):
        if not mask[t] or labels[t] is None:
            continue
        present = [tid for tid in pos
                   if t in pos[tid] and t in vel.get(tid, {})
                   and tid in per_frame[t]]
        if len(present) < 3 or labels[t] not in present:
            continue
        ids, traj, posm = frame_features(t, present, pos, vel, per_frame,
                                         w=w, h=h, lookback=lookback)
        X_t.append(traj)
        X_p.append(posm)
        y.append(np.array([1 if i == labels[t] else 0 for i in ids]))
        group.append(np.full(len(ids), t))
        tids.append(np.array(ids))
    if not X_t:
        empty = np.zeros((0, len(TRAJ))), np.zeros((0, len(POS)))
        return (*empty, np.zeros(0), np.zeros(0), np.zeros(0))
    return (np.vstack(X_t), np.vstack(X_p), np.concatenate(y),
            np.concatenate(group), np.concatenate(tids))
