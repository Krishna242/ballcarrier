"""Draw a random sample of harvested labels for human scoring.

Every accuracy in this project is measured against labels produced by
`src/hud.py`, so the quality of those labels is a ceiling on the credibility of
everything else. The harvester's own `validate()` cannot establish it: that
function checks the detection moves, which rules out locking onto a scoreboard
and rules out nothing else.

The only way to know whether the reticle marks the player holding the ball is
to look at frames where the ball is visible and say so. This script produces a
uniform random sample across the whole evaluation domain -- not a hand-picked
one -- and lays it out at native resolution, which is what makes the ball
legible.

    python scripts/audit_sample.py --video v.mp4 --n 42 --seed 0
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

from src import dataset  # noqa: E402

CROP_W, CROP_H = 560, 350
COLS, ROWS = 2, 3
ap_scale = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", default="data/interim/cache")
    ap.add_argument("--n", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/interim/audit_sample")
    args = ap.parse_args()

    pool = []
    for sh in dataset.load(args.cache):
        d, n = sh["d"], sh["n"]
        for t in range(n):
            if sh["mask"][t] and d["labels"][t] is not None:
                pool.append((sh["meta"]["shot"], d["frame_a"], t,
                             d["labels"][t], d["per_frame"][t][d["labels"][t]],
                             d["reticle"][t], d["fps"]))

    rng = np.random.default_rng(args.seed)
    picks = [pool[i] for i in sorted(rng.choice(len(pool), args.n, replace=False))]
    print(f"pool: {len(pool)} labelled live frames; sampling {args.n}")

    cap = cv2.VideoCapture(str(args.video))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tiles, page, manifest = [], 0, []
    for k, (shot, fa, t, tid, box, ret, fps) in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fa + t)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.circle(frame, (int(ret[0]), int(ret[1])), 5, (0, 0, 255), -1)

        h, w = frame.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ox = int(np.clip(cx - CROP_W / 2, 0, w - CROP_W))
        oy = int(np.clip(cy - CROP_H / 2, 0, h - CROP_H))
        tile = frame[oy:oy + CROP_H, ox:ox + CROP_W].copy()
        cv2.putText(tile, f"#{k:02d}  s{shot} {(fa+t)/fps:.1f}s", (8, 22),
                    0, 0.6, (0, 255, 255), 2)
        tiles.append(tile)
        manifest.append({"i": k, "shot": shot, "t_s": round((fa + t) / fps, 2)})

        if len(tiles) == COLS * ROWS:
            grid = np.vstack([np.hstack(tiles[i:i + COLS])
                              for i in range(0, len(tiles), COLS)])
            cv2.imwrite(str(out / f"page{page}.png"), grid)
            tiles, page = [], page + 1
    cap.release()

    if tiles:
        while len(tiles) % COLS:
            tiles.append(np.zeros((CROP_H, CROP_W, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + COLS])
                          for i in range(0, len(tiles), COLS)])
        cv2.imwrite(str(out / f"page{page}.png"), grid)
        page += 1

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {page} pages to {out}")


if __name__ == "__main__":
    main()
