"""Render before/after pairs for every ground-truth possession change.

The change metric rests on a definition -- the indicator jumped more than
`JUMP_PX` to a different place -- and that definition can be satisfied by a
real handoff or by the detector losing the reticle for half a second and
re-finding it downfield. Those are not the same event, and only looking tells
them apart.

Each row is one event: the last frame before it and the first frame after,
side by side, at native resolution.

    python scripts/audit_changes.py --video v.mp4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate as ev, tracking  # noqa: E402

W, H = 620, 350
ROWS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", default="data/interim/cache")
    ap.add_argument("--out", default="data/interim/audit_changes")
    args = ap.parse_args()

    cache = Path(args.cache)
    idx = json.loads((cache / "index.json").read_text())
    green = np.load(cache / "frame_stats.npz")["green"]
    cap = cv2.VideoCapture(str(args.video))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def draw(fa, t, r, tag):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fa + t)
        ok, frame = cap.read()
        if not ok:
            return np.zeros((H, W, 3), np.uint8)
        cv2.circle(frame, (int(r[0]), int(r[1])), 14, (0, 0, 255), 3)
        h, w = frame.shape[:2]
        ox = int(np.clip(r[0] - W / 2, 0, w - W))
        oy = int(np.clip(r[1] - H / 2, 0, h - H))
        tile = frame[oy:oy + H, ox:ox + W].copy()
        cv2.putText(tile, tag, (8, 24), 0, 0.65, (0, 255, 255), 2)
        return tile

    rows, page, n_ev = [], 0, 0
    for meta in idx["shots"]:
        with open(cache / f"shot{meta['shot']:02d}.pkl", "rb") as fh:
            d = pickle.load(fh)
        n = len(d["per_frame"])
        _, vel = tracking.build_trajectories(d["per_frame"])
        mask = ev.live_mask(green, vel, n, d["frame_a"])

        # Mirrors ev.true_changes exactly, but keeps the *pair* of endpoints so
        # the before/after can be drawn. Any drift between the two would make
        # this audit describe events the metric never scored.
        prev, events = None, []
        for t, r in enumerate(d["reticle"]):
            if not mask[t] or r is None or d["labels"][t] is None:
                continue
            if prev is not None:
                dist = np.hypot(r[0] - prev[1][0], r[1] - prev[1][1])
                if dist > ev.JUMP_PX and d["labels"][t] != d["labels"][prev[0]]:
                    events.append((prev[0], prev[1], t, r, dist))
            prev = (t, r)

        for (t0, r0, t1, r1, dist) in events:
            n_ev += 1
            gap = (t1 - t0) / d["fps"]
            a = draw(d["frame_a"], t0, r0,
                     f"s{meta['shot']} BEFORE {(d['frame_a']+t0)/d['fps']:.1f}s")
            b = draw(d["frame_a"], t1, r1,
                     f"AFTER {(d['frame_a']+t1)/d['fps']:.1f}s  "
                     f"gap {gap:.2f}s  jump {dist:.0f}px")
            rows.append(np.hstack([a, b]))
            if len(rows) == ROWS:
                cv2.imwrite(str(out / f"page{page}.png"), np.vstack(rows))
                rows, page = [], page + 1
    cap.release()

    if rows:
        cv2.imwrite(str(out / f"page{page}.png"), np.vstack(rows))
        page += 1
    print(f"{n_ev} ground-truth change events -> {page} pages in {out}")


if __name__ == "__main__":
    main()
