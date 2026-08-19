"""Estimate and cache per-frame camera motion for a harvested clip.

Writes `camera.pkl` next to the shot pickles. Everything downstream picks it up
automatically when present, so this is additive: a cache without it behaves
exactly as before.

Also reports how much camera motion each shot actually contains, because the
correction is only worth anything on footage where the camera moves, and that
should be measured rather than assumed.

    python scripts/estimate_camera.py --cache data/interim/arcade/clip1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import camera, dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True)
    ap.add_argument("--video", default=None,
                    help="override the video path recorded in index.json")
    args = ap.parse_args()

    for cache in args.cache:
        cache = Path(cache)
        idx = json.loads((cache / "index.json").read_text())
        h = idx.get("height", 720)
        w = idx.get("width", 1280)

        out = {}
        for meta in idx["shots"]:
            k = meta["shot"]
            with open(cache / f"shot{k:02d}.pkl", "rb") as fh:
                d = pickle.load(fh)

            video = Path(args.video) if args.video else None
            if video is None or not video.is_file():
                video = dataset.resolve_video({"video": idx.get("video")}, cache)
            if not Path(video).is_file():
                print(f"{cache.name} shot{k:02d}: video not found ({video})")
                continue

            cams = camera.estimate_shot(video, d["per_frame"],
                                        d["frame_a"], h=h)
            out[k] = cams

            mags = np.array([camera.motion_magnitude(A, w, h) for A in cams])
            got = sum(A is not None for A in cams)
            print(f"{cache.name} shot{k:02d}: estimated on {got}/{len(cams)} "
                  f"frames, median shift {np.median(mags):5.1f}px, "
                  f"p90 {np.percentile(mags, 90):5.1f}px, "
                  f"frames over 4px: {(mags > 4).mean():.0%}")

        if out:
            with open(cache / "camera.pkl", "wb") as fh:
                pickle.dump(out, fh)
            print(f"  wrote {cache / 'camera.pkl'}")


if __name__ == "__main__":
    main()
