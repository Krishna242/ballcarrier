"""Per-frame appearance statistics used to decide where possession is defined.

Gameplay video contains menus, replays and banner wipes. A carrier label
harvested from a play-select screen is not a wrong answer to the carrier
question, it is an answer to no question at all, and letting those frames into
an accuracy number inflates or deflates it for reasons that have nothing to do
with the method.

Two numbers per frame, both cheap enough to compute in one extra decode pass:

  green   fraction of pixels that are turf. Near zero on the play-select
          screen, high whenever the field is on camera.
  motion  mean absolute difference from the previous frame, downscaled. Low
          during frozen banners, high during a live play.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TURF_LO = (35, 60, 40)
# Upper hue widened from 90 for the 1997 arcade field, which is teal: its
# dominant hue measures 85-89, i.e. right on the old boundary, so half the
# turf fell outside the gate and whole plays read as "field not on camera".
TURF_HI = (100, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/interim/cache/frame_stats.npz")
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    green, motion, prev = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green.append(float((cv2.inRange(hsv, TURF_LO, TURF_HI) > 0).mean()))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, green=np.array(green, np.float32),
             motion=np.array(motion, np.float32))
    print(f"{len(green)} frames -> {out}")
    g = np.array(green)
    print(f"green: median {np.median(g):.3f}  "
          f"frames below 0.15: {(g < 0.15).sum()} ({(g < 0.15).mean():.1%})")


if __name__ == "__main__":
    main()
