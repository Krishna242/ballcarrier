"""Loading harvested caches, with each clip's own scale attached.

There is more than one source now: 1280x720 at 60fps from the Xbox Live Arcade
re-release, and 1356x1016 at 30fps from the 1997 arcade original. Every
threshold in this project was read off the first of those, so a loader that
hands back raw arrays invites each caller to re-derive the conversion and get
it slightly differently. `load` attaches the factors to the shot instead.

A "shot" from a six-second clip with no camera cut is the whole clip. That is
the correct unit either way: it is the span over which track identities mean
anything.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from . import evaluate as ev, tracking


def load(cache, source=None):
    """Read one harvested cache into a list of shot dicts."""
    cache = Path(cache)
    idx = json.loads((cache / "index.json").read_text())
    green = np.load(cache / "frame_stats.npz")["green"]

    fps = idx["fps"]
    h = idx.get("height", 720)
    w = idx.get("width", 1280)
    sc = ev.scales(h, fps)

    shots = []
    for meta in idx["shots"]:
        with open(cache / f"shot{meta['shot']:02d}.pkl", "rb") as fh:
            d = pickle.load(fh)
        n = len(d["per_frame"])
        pos, vel = tracking.build_trajectories(d["per_frame"], fps=fps)
        mask = ev.live_mask(green, vel, n, d["frame_a"],
                            scale=sc["speed"], frames_scale=sc["frames"])
        shots.append({
            "key": f"{source or cache.name}:{meta['shot']:02d}",
            "source": source or cache.name,
            "meta": meta, "d": d, "pos": pos, "vel": vel, "mask": mask,
            "n": n, "fps": fps, "w": w, "h": h, "sc": sc,
        })
    return shots


def load_many(specs):
    """specs: {source_name: cache_path}. Returns one flat list of shots."""
    out = []
    for name, path in specs.items():
        out.extend(load(path, source=name))
    return out


def labelled_live(shot):
    return sum(1 for t in range(shot["n"])
               if shot["mask"][t] and shot["d"]["labels"][t] is not None)


MIN_CANDIDATES = 3


def scored_mask(shot):
    """The frames every number in this project is computed over.

    Live play, a harvested label, and at least `MIN_CANDIDATES` players to
    choose between. The last condition exists because "pick the carrier from
    two tracked players" is a different and much easier question than the one
    being asked, and including those frames flatters a method without telling
    anyone anything -- but it has to be applied to the baselines and to the
    model alike, or their accuracies are quoted over different footage and are
    not comparable. They were not, at first: the baseline table covered 483
    arcade frames at chance 0.32 while the model table covered 328 at chance
    0.16, and the two were being read side by side.
    """
    out = np.zeros(shot["n"], bool)
    for t in range(shot["n"]):
        if not shot["mask"][t] or shot["d"]["labels"][t] is None:
            continue
        out[t] = len(candidates(shot, t)) >= MIN_CANDIDATES
    return out


def candidates(shot, t):
    """The tracks a method may choose between on frame `t`.

    Every predictor, the feature builder and the chance rate must agree on
    this set, or the headline lift is wrong. They did not agree at first: the
    chance rate counted every detection box while the predictors ranked only
    tracks that survived the minimum-length filter, which on the arcade clips
    meant dividing by roughly twice as many candidates as anything was
    actually choosing among.
    """
    pos, vel = shot["pos"], shot["vel"]
    boxes = shot["d"]["per_frame"][t]
    return [tid for tid in pos
            if t in pos[tid] and t in vel.get(tid, {}) and tid in boxes]
