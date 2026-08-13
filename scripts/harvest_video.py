"""Harvest tracks and HUD carrier labels for a whole video, shot by shot.

Everything downstream -- evaluation, baselines, model training -- reads this
cache instead of the video, so a metric can be recomputed in seconds rather
than re-running detection for twenty minutes.

Two passes over the file:

  1. colour-mask occupancy, to learn which pixels are permanent interface
     furniture. This needs no network and no boxes, so it is cheap.
  2. detection, tracking and reticle extraction, one shot at a time.

Frames are never accumulated. The earlier `tracking.collect_tracks` returned
every decoded frame in a list, which is fine for a six-second clip and is
27 GB for this one.

    python scripts/harvest_video.py --video path/to.mp4 --out data/interim/cache
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

from src import hud, segment  # noqa: E402

TURF_LO = (35, 60, 40)
TURF_HI = (100, 255, 255)

MIN_SHOT_S = 2.0          # shorter fragments are transitions, not gameplay


def occupancy_pass(video, stride=3):
    """One decode pass producing everything that needs no detector.

    Returns (static occupancy, turf fraction per frame, frame count). The turf
    fraction rides along because it is computed from the same decoded frame,
    and decoding this video twice to get it costs more than the statistic.

    Occupancy is sampled rather than exhaustive: interface furniture is static
    by definition, so every third frame measures it just as well.
    """
    cap = cv2.VideoCapture(str(video))
    acc, n, i, green = None, 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            m = (hud.colour_mask(frame) > 0).astype(np.float32)
            acc = m if acc is None else acc + m
            n += 1
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green.append(float((cv2.inRange(hsv, TURF_LO, TURF_HI) > 0).mean()))
        i += 1
    cap.release()
    return acc / max(n, 1), np.array(green, np.float32), i


def shot_bounds(cuts, n_frames, fps):
    """Cut list -> [(start_frame, end_frame)] for shots long enough to use."""
    edges = [0] + list(cuts) + [n_frames]
    return [(a, b) for a, b in zip(edges, edges[1:])
            if (b - a) / fps >= MIN_SHOT_S]


def track_shot(model, cap, a, b, static, weights):
    """Detect, track and find the reticle for frames [a, b).

    A fresh tracker per shot is the point: identities carried across a camera
    cut are identities invented out of nothing.
    """
    from ultralytics import YOLO

    model = YOLO(weights)          # fresh instance == fresh tracker state
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)

    per_frame, reticle = [], []
    for _ in range(b - a):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(
            frame, persist=True, classes=[0], conf=0.35, imgsz=960,
            tracker="bytetrack.yaml", verbose=False,
        )[0]

        boxes = {}
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            for tid, box in zip(ids, xyxy):
                boxes[int(tid)] = box.astype(np.float32)

        per_frame.append(boxes)
        reticle.append(hud.find_reticle(frame, static, boxes))

    return per_frame, reticle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/interim/cache")
    ap.add_argument("--weights", default="yolo11m.pt")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/3] shot boundaries")
    cuts, fps = segment.detect_cuts(args.video)

    print("[2/3] static occupancy")
    t0 = time.time()
    static, green, n_frames = occupancy_pass(args.video)
    print(f"      {n_frames} frames, {time.time() - t0:.0f}s")

    shots = shot_bounds(cuts, n_frames, fps)
    print(f"      {len(cuts)} cuts -> {len(shots)} usable shots")

    print("[3/3] tracking + reticle, per shot")
    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    index = []
    for k, (a, b) in enumerate(shots):
        t0 = time.time()
        per_frame, reticle = track_shot(None, cap, a, b, static, args.weights)
        labels = hud.assign_to_track(reticle, per_frame, scale=height / 720.0)
        n_lab = sum(1 for x in labels if x is not None)

        with open(out / f"shot{k:02d}.pkl", "wb") as fh:
            pickle.dump({
                "shot": k, "frame_a": a, "frame_b": b, "fps": fps,
                "per_frame": per_frame, "reticle": reticle, "labels": labels,
            }, fh)

        index.append({
            "shot": k, "frame_a": a, "frame_b": b,
            "start_s": round(a / fps, 2), "duration_s": round((b - a) / fps, 2),
            "n_frames": len(per_frame),
            "reticle_rate": round(sum(r is not None for r in reticle) / max(len(reticle), 1), 3),
            "label_rate": round(n_lab / max(len(labels), 1), 3),
            "n_tracks": len({t for f in per_frame for t in f}),
        })
        print(f"      shot{k:02d}  {a/fps:6.1f}s +{(b-a)/fps:5.1f}s  "
              f"reticle {index[-1]['reticle_rate']:.2f}  "
              f"label {index[-1]['label_rate']:.2f}  "
              f"tracks {index[-1]['n_tracks']:3d}  ({time.time()-t0:.0f}s)")

    cap.release()
    np.save(out / "static.npy", static)
    np.savez(out / "frame_stats.npz", green=green)
    (out / "index.json").write_text(json.dumps({
        "video": str(args.video), "fps": fps, "n_frames": n_frames,
        "width": width, "height": height,
        "cuts": [int(c) for c in cuts], "shots": index,
    }, indent=2))
    print(f"\nwrote {out}/index.json")


if __name__ == "__main__":
    main()
