"""Score every candidate player for "is the football in their hands", and cache it.

The project's founding decision was to ignore the ball, justified by a previous
model that weighted a ball feature at 0.56 against 17.10 for movement. That is
evidence about an old ball detector, not about the ball: in this footage the
carrier is frequently holding a visibly distinct reddish-brown oval, and no
feature in the current model looks at it.

`src/ball.py` already localises the ball, but only inside the box the model has
*already chosen*, which makes it a drawing aid -- it can corroborate a
prediction and can never contradict one. Running the same detector over every
candidate turns it into evidence that can change the answer.

Writes `ball_evidence.pkl` into the cache: {shot: {frame: {track_id: score}}}.
One video decode per clip; the per-crop work is a colour threshold and a
contour pass, so the decode dominates.

    python scripts/ball_evidence.py --cache data/interim/arcade/clip1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ball, dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    for cache in args.cache:
        cache = Path(cache)
        shots = dataset.load(cache)
        idx = json.loads((cache / "index.json").read_text())

        video = Path(args.video) if args.video else None
        if video is None or not video.is_file():
            video = dataset.resolve_video({"video": idx.get("video")}, cache)
        if not Path(video).is_file():
            print(f"{cache.name}: video not found ({video})")
            continue

        out, t0 = {}, time.time()
        for sh in shots:
            k = sh["meta"]["shot"]
            scored = dataset.scored_mask(sh)
            want = {t for t in range(sh["n"]) if scored[t]}
            per_shot = {}

            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, sh["d"]["frame_a"])
            for t in range(sh["n"]):
                ok, frame = cap.read()
                if not ok:
                    break
                if t not in want:
                    continue
                row = {}
                for tid in dataset.candidates(sh, t):
                    box = sh["d"]["per_frame"][t].get(tid)
                    if box is None:
                        continue
                    hit = ball.localize_in_box(frame, box)
                    row[tid] = float(hit[2]) if hit is not None else 0.0
                per_shot[t] = row
            cap.release()
            out[k] = per_shot

            n_any = sum(1 for r in per_shot.values() if any(v > 0 for v in r.values()))
            n_cand = sum(len(r) for r in per_shot.values())
            n_pos = sum(1 for r in per_shot.values() for v in r.values() if v > 0)
            print(f"{cache.name} shot{k:02d}: {len(per_shot)} scored frames, "
                  f"ball found on {n_any} ({n_any/max(len(per_shot),1):.0%}); "
                  f"{n_pos}/{n_cand} candidate crops fire "
                  f"({n_pos/max(n_cand,1):.1%})")

        with open(cache / "ball_evidence.pkl", "wb") as fh:
            pickle.dump(out, fh)
        print(f"  wrote {cache / 'ball_evidence.pkl'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
