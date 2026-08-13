"""Render the demo overlay for every arcade clip, each one held out in turn.

Each clip is drawn by a model trained on the 2012 re-release plus the *other
three* arcade clips. No clip is ever drawn by a model that saw it, which is
the only way the burnt-in agreement figure means anything.

    python scripts/make_demos.py
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHES = {i: f"data/interim/arcade/clip{i}" for i in (1, 2, 3, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="C:/Users/krish/Downloads")
    ap.add_argument("--out", default="data/interim/demo")
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=0.25)
    args = ap.parse_args()

    for i, cache in CACHES.items():
        hits = sorted(glob.glob(f"{args.videos}/Clip_{i} *.mp4"))
        if not hits:
            print(f"clip{i}: no video found in {args.videos}")
            continue
        video = hits[0]
        peers = [c for j, c in CACHES.items() if j != i]
        title = Path(video).stem.replace(f"Clip_{i} ", f"Clip {i} - ")
        subprocess.run([
            sys.executable, "scripts/demo.py",
            "--video", video, "--cache", cache,
            "--peers", *peers,
            "--title", title,
            "--penalty", str(args.penalty), "--dwell", str(args.dwell),
            "--out", f"{args.out}/clip{i}.mp4",
        ], cwd=ROOT)


if __name__ == "__main__":
    main()
