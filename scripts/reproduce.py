"""Run every experiment behind the README, in order, from the caches.

Harvesting is the expensive step and is not repeated here; run
`scripts/harvest_video.py` once per source first. Everything after that is
cheap, so there is no excuse for a number in the README that cannot be
regenerated on demand.

The order matters, and it is the order the decisions were actually made in:
baselines before models, development source before held-out source, and the
decode parameters fixed on the development source before they are ever applied
to the held-out one.

    python scripts/reproduce.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EA = "data/interim/cache"
ARCADE = [f"data/interim/arcade/clip{i}" for i in (1, 2, 3, 4)]


def run(title, argv):
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}")
    r = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if r.returncode != 0:
        print(f"!! failed: {' '.join(argv)}")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwell", type=float, default=0.25)
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    run("Baselines and the original heuristic, 2012 re-release",
        ["scripts/evaluate.py", "--cache", EA, "--label", "ea"])

    run("Baselines and the original heuristic, 1997 arcade",
        ["scripts/evaluate.py", "--cache", *ARCADE, "--label", "arcade"])

    run("Learned model, leave-one-shot-out on the development source",
        ["scripts/train_eval.py", "--cache", EA,
         "--dwell", str(args.dwell),
         "--out", "data/interim/model_eval_ea.json"])

    run("Learned model, cross-domain: 2012 -> 1997, no in-domain data",
        ["scripts/train_eval.py", "--cache", EA, "--test", *ARCADE,
         "--dwell", str(args.dwell),
         "--out", "data/interim/model_eval_arcade_crossdomain.json"])

    run("Learned model, leave-one-clip-out on the 1997 arcade clips",
        ["scripts/train_eval.py", "--cache", EA, "--test", *ARCADE,
         "--include-test-in-train", "--dwell", str(args.dwell),
         "--out", "data/interim/model_eval_arcade_loco.json"])

    print(f"\n{'=' * 78}\nJSON written to data/interim/*.json")
    if not args.skip_demo:
        print("Demo overlays: scripts/make_demos.py")


if __name__ == "__main__":
    main()
