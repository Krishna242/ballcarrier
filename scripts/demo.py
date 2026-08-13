"""Render a demo overlay for one clip, with the ground truth shown alongside.

Both the prediction and the harvested label are drawn, because an overlay that
shows only the prediction is a highlight reel: the frames it gets wrong look
exactly as confident as the frames it gets right, and the viewer has no way to
tell which is which. Here a disagreement turns the box red and says so.

The model is trained on every other clip plus the whole 2012 re-release, never
on the clip being drawn. The counter is over live-play frames only -- the same
frames the reported accuracy uses -- so the number burnt into the video and the
number in the README are the same number.

    python scripts/demo.py --video clip.mp4 --cache data/interim/arcade/clip1 \\
        --train data/interim/cache --peers data/interim/arcade/clip2 ... \\
        --out data/interim/demo/clip1.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import carrier, dataset  # noqa: E402
from scripts.train_eval import (  # noqa: E402
    SETS, attach_dataset, features_of, make_model, stack, viterbi_by_frame,
)

C_PRED = (60, 190, 255)      # amber: the model's answer
C_TRUE = (105, 220, 120)     # green: the on-screen indicator's answer
C_DIM = (140, 132, 124)
C_TEXT = (245, 246, 244)
C_BAD = (72, 72, 240)
C_PANEL = (28, 32, 30)


def panel(canvas, x, y, w, h, alpha=0.78):
    roi = canvas[y:y + h, x:x + w]
    if roi.size:
        roi[:] = (roi * (1 - alpha) + np.array(C_PANEL) * alpha).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--train", default="data/interim/cache")
    ap.add_argument("--peers", nargs="*", default=[],
                    help="other clips from the same source to train on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--features", default="both", choices=list(SETS))
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=0.25)
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()

    train = attach_dataset(dataset.load(args.train, source="train"))
    if args.peers:
        train += attach_dataset(dataset.load_many(
            {Path(p).name: p for p in args.peers}))

    which = SETS[args.features]
    X, y = stack(train, which)
    model = make_model(args.model).fit(X, y)

    shots = attach_dataset(dataset.load(args.cache, source="clip"))
    if not shots:
        print(f"{args.cache}: no usable shot")
        return
    sh = shots[0]
    _, _, y_te, grp, tid = sh["ds"]
    p = model.predict_proba(features_of(sh, which))[:, 1]
    score = np.log(np.clip(p, 1e-6, 1 - 1e-6))
    pick = viterbi_by_frame(score, grp, tid, args.penalty)

    pred = [None] * sh["n"]
    for t, v in pick.items():
        pred[t] = v
    if args.dwell > 0:
        pred = carrier.enforce_dwell(pred, int(round(args.dwell * sh["fps"])))

    truth = {int(t): int(tid[grp == t][np.argmax(y_te[grp == t])])
             for t in np.unique(grp)}
    scored = set(pick)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sh["d"]["frame_a"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    title = args.title or Path(args.video).stem

    seen = hits = 0
    for t in range(sh["n"]):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     sh["fps"], (w, h))
        s = h / 720.0
        canvas = frame.copy()

        pt, gt = pred[t], truth.get(t)
        if t in scored:
            seen += 1
            hits += int(pt == gt)

        for k, box in sh["d"]["per_frame"][t].items():
            if k == pt:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), C_DIM, 1)

        if gt is not None and gt in sh["d"]["per_frame"][t]:
            x1, y1, x2, y2 = [int(v) for v in sh["d"]["per_frame"][t][gt]]
            cv2.line(canvas, (x1, y2), (x2, y2), C_TRUE, max(int(5 * s), 3))
            cv2.putText(canvas, "game indicator", (x1, y2 + int(24 * s)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5 * s, C_TRUE, 1, cv2.LINE_AA)

        if pt is not None and pt in sh["d"]["per_frame"][t]:
            x1, y1, x2, y2 = [int(v) for v in sh["d"]["per_frame"][t][pt]]
            wrong = t in scored and pt != gt
            col = C_BAD if wrong else C_PRED
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, max(int(3 * s), 2))
            tag = "PREDICTED CARRIER" + ("  <- wrong" if wrong else "")
            cv2.putText(canvas, tag, (x1, max(y1 - int(10 * s), 16)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55 * s, col, 1, cv2.LINE_AA)

        pw, ph = int(500 * s), int(104 * s)
        panel(canvas, int(10 * s), int(10 * s), pw, ph)
        cv2.putText(canvas, title[:44], (int(22 * s), int(38 * s)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6 * s, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "carrier from trajectories + framing; clip held out",
                    (int(22 * s), int(64 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44 * s, C_DIM, 1, cv2.LINE_AA)
        msg = (f"live-play frames {seen}    agrees with game {hits/max(seen,1):.0%}"
               if seen else "waiting for the snap")
        cv2.putText(canvas, msg, (int(22 * s), int(92 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52 * s,
                    C_TEXT if seen else C_DIM, 1, cv2.LINE_AA)

        writer.write(canvas)

    cap.release()
    if writer:
        writer.release()
    print(f"{out}  {seen} scored frames, agreement {hits/max(seen,1):.1%}")


if __name__ == "__main__":
    main()
