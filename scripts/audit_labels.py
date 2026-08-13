"""Render native-resolution crops around harvested labels, for human checking.

The harvester finds the reticle, and the reticle marks the player under user
control. That is not the same claim as "this player has the ball", and no
amount of internal consistency can close the gap -- only looking can. This
script produces the images to look at.

Crops are native resolution and centred on the labelled player, because the
question is whether *that* player is holding the ball, and a downscaled
montage throws away the pixels that answer it.

    python scripts/audit_labels.py --video path/to.mp4 --shot 3 --n 4
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CROP_W, CROP_H = 660, 420


def crop_around(frame, cx, cy):
    h, w = frame.shape[:2]
    x = int(np.clip(cx - CROP_W / 2, 0, max(w - CROP_W, 0)))
    y = int(np.clip(cy - CROP_H / 2, 0, max(h - CROP_H, 0)))
    return frame[y:y + CROP_H, x:x + CROP_W], x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", default="data/interim/cache")
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="data/interim/audit")
    args = ap.parse_args()

    cache = Path(args.cache)
    with open(cache / f"shot{args.shot:02d}.pkl", "rb") as fh:
        d = pickle.load(fh)

    labelled = [t for t, l in enumerate(d["labels"]) if l is not None]
    if not labelled:
        print(f"shot{args.shot:02d}: no labelled frames")
        return

    picks = [labelled[i] for i in
             np.linspace(0, len(labelled) - 1, args.n).astype(int)]

    cap = cv2.VideoCapture(str(args.video))
    tiles = []
    for t in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, d["frame_a"] + t)
        ok, frame = cap.read()
        if not ok:
            continue
        tid = d["labels"][t]
        x1, y1, x2, y2 = d["per_frame"][t][tid]
        rx, ry, _, _ = d["reticle"][t]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.circle(frame, (int(rx), int(ry)), 6, (0, 0, 255), -1)

        tile, ox, oy = crop_around(frame, (x1 + x2) / 2, (y1 + y2) / 2)
        cv2.putText(tile, f"s{args.shot} f{t} tid{tid} "
                    f"{(d['frame_a']+t)/d['fps']:.1f}s",
                    (8, 24), 0, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros((CROP_H, CROP_W, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + cols])
                      for i in range(0, len(tiles), cols)])
    p = out / f"shot{args.shot:02d}.png"
    cv2.imwrite(str(p), grid)
    print(f"wrote {p}  ({len(labelled)} labelled frames in shot)")


if __name__ == "__main__":
    main()
