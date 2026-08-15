"""Arcade-only local run: harvest the four 1997 clips, evaluate, demos.

Does not need the 2012 development video. Leave-one-clip-out trains on the
other three clips only.

    python scripts/run_arcade.py
    python scripts/run_arcade.py --skip-harvest   # caches already exist
    python scripts/run_arcade.py --skip-demo
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "arcade"
CACHES = [ROOT / "data" / "interim" / "arcade" / f"clip{i}" for i in (1, 2, 3, 4)]
VIDEO_DIRS = [ROOT / "data" / "raw" / "arcade", ROOT / "Videos"]


def find_clip(i):
    for d in VIDEO_DIRS:
        hits = sorted(d.glob(f"Clip_{i} *.mp4"))
        if hits:
            return hits[0]
    return None


def run(title, argv):
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}")
    r = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(argv)}")
    return r.returncode


def stage_videos():
    RAW.mkdir(parents=True, exist_ok=True)
    staged = []
    for i in (1, 2, 3, 4):
        src = find_clip(i)
        if src is None:
            raise SystemExit(f"Clip_{i} *.mp4 not found in {VIDEO_DIRS}")
        dest = RAW / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        staged.append(dest)
        print(f"  clip{i}: {dest.name}")
    return staged


def snapshot_eval_into_caches():
    """Copy the leave-one-clip-out numbers into each clip cache."""
    import json
    loco = ROOT / "data" / "interim" / "model_eval_arcade_loco.json"
    base = ROOT / "data" / "interim" / "eval_arcade.json"
    if not loco.exists():
        return
    loco_d = json.loads(loco.read_text())
    base_d = json.loads(base.read_text()) if base.exists() else {}
    rec = (loco_d.get("sets", {}).get("hybrid")
           or loco_d.get("sets", {}).get("both") or {}).get(
               "penalty_3.0_dwell_0.0", {})
    by_key = {s["key"]: s for s in rec.get("per_shot", [])}
    for i, cache in enumerate(CACHES, start=1):
        key = f"clip{i}:00"
        payload = {
            "clip": i,
            "cache": str(cache),
            "chance": loco_d.get("chance"),
            "learned": by_key.get(key),
            "baselines": next((s for s in base_d.get("per_shot", [])
                               if s.get("key") == key), None),
        }
        (cache / "eval_snapshot.json").write_text(json.dumps(payload, indent=2))
        print(f"  {cache.name}/eval_snapshot.json")


def cache_ready(cache: Path) -> bool:
    return (cache / "index.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-harvest", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    ap.add_argument("--weights", default="yolo11m.pt")
    args = ap.parse_args()

    print("[1] staging 1997 arcade clips")
    videos = stage_videos()

    if not args.skip_harvest:
        for i, video in enumerate(videos, start=1):
            out = CACHES[i - 1]
            if cache_ready(out):
                print(f"[2] clip{i} cache exists, skipping harvest")
                continue
            run(f"Harvest clip{i}",
                ["scripts/harvest_video.py", "--video", str(video),
                 "--out", str(out), "--weights", args.weights])
    else:
        missing = [str(c) for c in CACHES if not cache_ready(c)]
        if missing:
            raise SystemExit(f"--skip-harvest but missing caches: {missing}")

    cache_args = [str(c) for c in CACHES]
    run("Baselines on 1997 arcade",
        ["scripts/evaluate.py", "--cache", *cache_args, "--label", "arcade",
         "--out", "data/interim/eval_arcade.json"])
    run("Learned model, leave-one-clip-out (arcade only)",
        ["scripts/train_eval.py", "--cache", *cache_args,
         "--dwell", "0.0",
         "--out", "data/interim/model_eval_arcade_loco.json"])
    print("[eval snapshots]")
    snapshot_eval_into_caches()

    if not args.skip_demo:
        run("Held-out demo overlays",
            ["scripts/make_demos.py", "--videos", str(RAW),
             "--out", "data/interim/demo"])


if __name__ == "__main__":
    main()
