"""Render the demo overlay for every arcade clip, each one held out in turn.

Each clip is drawn by a model trained on the *other three* arcade clips
(and the 2012 cache if --train points at one that exists). No clip is ever
drawn by a model that saw it.

    python scripts/make_demos.py --videos data/raw/arcade
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHES = {i: f"data/interim/arcade/clip{i}" for i in (1, 2, 3, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/raw/arcade")
    ap.add_argument("--train", default=None,
                    help="optional 2012 cache; omitted for arcade-only demos")
    ap.add_argument("--out", default="data/interim/demo")
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=0.0)
    args = ap.parse_args()

    video_dir = Path(args.videos)
    if not video_dir.exists():
        alt = ROOT / "Videos"
        if alt.exists():
            video_dir = alt

    for i, cache in CACHES.items():
        hits = sorted(glob.glob(str(video_dir / f"Clip_{i} *.mp4")))
        if not hits:
            print(f"clip{i}: no video found in {video_dir}")
            continue
        video = hits[0]
        peers = [c for j, c in CACHES.items() if j != i]
        title = Path(video).stem.replace(f"Clip_{i} ", f"Clip {i} - ")
        cmd = [
            sys.executable, "scripts/demo.py",
            "--video", video, "--cache", cache,
            "--peers", *peers,
            "--title", title,
            "--penalty", str(args.penalty), "--dwell", str(args.dwell),
            "--out", str(ROOT / cache / "overlay.mp4"),
        ]
        if args.train:
            cmd.extend(["--train", args.train])
        subprocess.run(cmd, cwd=ROOT)
        overlay = ROOT / cache / "overlay.mp4"
        if overlay.exists() and args.out:
            dest = Path(args.out)
            if not dest.is_absolute():
                dest = ROOT / dest
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(overlay, dest / f"clip{i}.mp4")


if __name__ == "__main__":
    main()
