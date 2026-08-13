"""Render a demo overlay for one clip, with the ground truth shown alongside.

The model is trained on the 2012 re-release footage and run here on a 1997
arcade clip it has never seen. Both the prediction and the harvested label are
drawn, because an overlay that shows only the prediction is a highlight reel:
the frames it gets wrong look exactly as confident as the frames it gets right,
and the viewer has no way to tell which is which.

The running counter is over live-play frames only -- the same frames the
reported accuracy is computed on -- so the number on screen and the number in
the table are the same number.

    python scripts/demo.py --video "clip.mp4" --cache data/interim/arcade/clip1 \\
        --train data/interim/cache --out data/interim/demo/clip1.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset, features, render  # noqa: E402
from scripts.train_eval import (  # noqa: E402
    SETS, attach_dataset, make_model, stack, features_of, viterbi_by_frame,
)

C_PRED = (60, 190, 255)      # amber: the model's answer
C_TRUE = (90, 220, 120)      # green: the on-screen indicator's answer
C_DIM = (150, 140, 130)
C_TEXT = (245, 246, 244)
C_BAD = (70, 70, 235)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--train", default="data/interim/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="both", choices=list(SETS))
    ap.add_argument("--penalty", type=float, default=1.5)
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()

    train = attach_dataset(dataset.load(args.train, source="train"))
    which = SETS[args.features]
    X, y = stack(train, which)
    model = make_model(args.model).fit(X, y)

    shots = attach_dataset(dataset.load(args.cache, source="clip"))
    if not shots:
        print("no usable shot in this cache")
        return
    sh = shots[0]
    _, _, y_te, grp, tid = sh["ds"]
    p = model.predict_proba(features_of(sh, which))[:, 1]
    score = np.log(np.clip(p, 1e-6, 1 - 1e-6))
    pick = viterbi_by_frame(score, grp, tid, args.penalty)

    truth = {int(t): int(tid[grp == t][np.argmax(y_te[grp == t])])
             for t in np.unique(grp)}

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sh["d"]["frame_a"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    seen = hits = 0
    for t in range(sh["n"]):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     sh["fps"], (w, h))
        h, w = frame.shape[:2]
        s = h / 720.0
        canvas = frame.copy()

        pred = pick.get(t)
        gt = truth.get(t)
        scored = t in truth and pred is not None
        if scored:
            seen += 1
            hits += int(pred == gt)

        for k, box in sh["d"]["per_frame"][t].items():
            x1, y1, x2, y2 = [int(v) for v in box]
            if k == pred:
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), C_DIM, 1)

        if gt is not None and gt in sh["d"]["per_frame"][t]:
            x1, y1, x2, y2 = [int(v) for v in sh["d"]["per_frame"][t][gt]]
            cv2.line(canvas, (x1, y2), (x2, y2), C_TRUE, max(int(4 * s), 2))
            cv2.putText(canvas, "indicator", (x1, y2 + int(22 * s)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5 * s, C_TRUE, 1, cv2.LINE_AA)

        if pred is not None and pred in sh["d"]["per_frame"][t]:
            x1, y1, x2, y2 = [int(v) for v in sh["d"]["per_frame"][t][pred]]
            col = C_PRED if (not scored or pred == gt) else C_BAD
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, max(int(3 * s), 2))
            tag = f"CARRIER #{pred}" + ("" if not scored or pred == gt else "  WRONG")
            cv2.putText(canvas, tag, (x1, max(y1 - int(8 * s), 14)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55 * s, col, 1, cv2.LINE_AA)

        pw, ph = int(430 * s), int(74 * s)
        panel = canvas[int(8 * s):int(8 * s) + ph, int(8 * s):int(8 * s) + pw]
        panel[:] = (panel * 0.25 + np.array((34, 40, 36)) * 0.75).astype(np.uint8)
        cv2.putText(canvas, "trained on 2012 re-release; this is 1997 arcade",
                    (int(18 * s), int(30 * s)), cv2.FONT_HERSHEY_DUPLEX,
                    0.45 * s, C_TEXT, 1, cv2.LINE_AA)
        msg = (f"live-play frames {seen}   agreement {hits/max(seen,1):.0%}"
               if seen else "waiting for the play to start")
        cv2.putText(canvas, msg, (int(18 * s), int(58 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s,
                    C_TEXT if seen else C_DIM, 1, cv2.LINE_AA)

        writer.write(canvas)

    cap.release()
    if writer:
        writer.release()
    print(f"{out}  ({seen} scored frames, agreement {hits/max(seen,1):.1%})")


if __name__ == "__main__":
    main()
