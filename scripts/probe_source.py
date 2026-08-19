"""Check whether the label harvester works on a new video, cheaply.

Harvesting an eighteen-minute capture costs about an hour of GPU. The reticle
detector is tuned to one game's rendering, and a different capture can differ
in turf colour, HUD furniture and indicator style -- so it is worth an eighty-
second answer to "will this produce labels at all?" before committing to that.

Samples frames across the whole file, runs detection on them, builds the static
occupancy map from the same sample, and reports the reticle hit rate. Also
writes a montage of hits so the answer can be *looked at* rather than trusted,
which is how every other detector bug in this project was found.

    python scripts/probe_source.py --video data/raw/arcade_long.mp4 --n 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import hud, tracking  # noqa: E402

TURF_LO = (35, 60, 40)
TURF_HI = (100, 255, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--out", default="data/interim/probe")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    idxs = np.linspace(0, max(total - 2, 0), args.n).astype(int)
    print(f"{args.video}: {total} frames, {w}x{h}; sampling {len(idxs)}")

    frames, boxes_all, green = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.predict(frame, classes=[tracking.PERSON_CLASS],
                            conf=tracking.CONF, imgsz=tracking.IMGSZ,
                            verbose=False)[0]
        b = {}
        if res.boxes is not None:
            for k, xy in enumerate(res.boxes.xyxy.cpu().numpy()):
                b[k] = xy
        frames.append(frame)
        boxes_all.append(b)
        small = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
        green.append(float((cv2.inRange(small, TURF_LO, TURF_HI) > 0).mean()))
    cap.release()

    static = hud.static_occupancy(frames)
    on_field = np.array(green) >= 0.15

    hits, tiles = 0, []
    for frame, b, fld in zip(frames, boxes_all, on_field):
        if not fld:
            continue
        r = hud.find_reticle(frame, static, b)
        if r is None:
            continue
        hits += 1
        if len(tiles) < 6:
            v = frame.copy()
            cv2.circle(v, (int(r[0]), int(r[1])), int(26 * h / 720), (0, 0, 255), 3)
            tiles.append(cv2.resize(v, (620, int(620 * h / w))))

    n_field = int(on_field.sum())
    print(f"  frames with field visible : {n_field}/{len(frames)}")
    print(f"  reticle detected          : {hits}/{max(n_field,1)} "
          f"({hits/max(n_field,1):.1%})")
    print(f"  detections per frame      : "
          f"{np.mean([len(b) for b in boxes_all]):.1f}")
    print(f"  static map max occupancy  : {float(static.max()):.3f}")

    if tiles:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2])
                          for i in range(0, len(tiles), 2)])
        p = out / (Path(args.video).stem[:24] + "_reticle.png")
        cv2.imwrite(str(p), grid)
        print(f"  wrote {p}  -- look at it before trusting the rate above")


if __name__ == "__main__":
    main()
