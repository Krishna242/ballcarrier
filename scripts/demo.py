"""Render a demo overlay for one clip, with the ground truth shown alongside.

Both the prediction and the harvested label are drawn, because an overlay that
shows only the prediction is a highlight reel: the frames it gets wrong look
exactly as confident as the frames it gets right, and the viewer has no way to
tell which is which. Here a disagreement turns the box red and says so.

The model is trained on peer clips (and the 2012 cache if present), never
on the clip being drawn. The counter is over live-play frames only -- the same
frames the reported accuracy uses -- so the number burnt into the video and the
number in the README are the same number.

    python scripts/demo.py --video clip.mp4 --cache data/interim/arcade/clip1 \\
        --peers data/interim/arcade/clip2 ...

Writes into the cache: overlay.mp4 (full tracked video), snapshots/*.jpg,
and results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ball, carrier, dataset  # noqa: E402
from scripts.train_eval import (  # noqa: E402
    MIN_ROWS_ARCADE, SETS, attach_dataset, features_of, make_model, stack,
    viterbi_by_frame,
)

C_PRED = (60, 190, 255)      # amber: the model's answer
C_TRUE = (105, 220, 120)     # green: the on-screen indicator's answer
C_DIM = (140, 132, 124)
C_TEXT = (245, 246, 244)
C_BALL = (40, 60, 255)       # red: football in the predicted carrier's hands
C_PANEL = (34, 40, 36)
C_BAD = (50, 50, 220)


def panel(canvas, x, y, w, h, alpha=0.78):
    roi = canvas[y:y + h, x:x + w]
    if roi.size:
        roi[:] = (roi * (1 - alpha) + np.array(C_PANEL) * alpha).astype(np.uint8)


def draw_frame(frame, sh, t, pred, truth, scored, seen, hits, title, ball=None):
    h, w = frame.shape[:2]
    s = h / 720.0
    canvas = frame.copy()
    pt, gt = pred[t], truth.get(t)

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

    if ball is not None:
        bx, by = int(ball[0]), int(ball[1])
        r = max(int(10 * s), 6)
        cv2.circle(canvas, (bx, by), r, C_BALL, max(int(3 * s), 2))
        cv2.circle(canvas, (bx, by), 2, C_BALL, -1)
        cv2.putText(canvas, "BALL", (bx + r + 4, by + 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5 * s, C_BALL, 1, cv2.LINE_AA)

    pw, ph = int(500 * s), int(104 * s)
    panel(canvas, int(10 * s), int(10 * s), pw, ph)
    cv2.putText(canvas, title[:44], (int(22 * s), int(38 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 0.6 * s, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "ball in carrier hands; clip held out",
                (int(22 * s), int(64 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.44 * s, C_DIM, 1, cv2.LINE_AA)
    msg = (f"live-play frames {seen}    agrees with game {hits/max(seen,1):.0%}"
           if seen else "waiting for the snap")
    cv2.putText(canvas, msg, (int(22 * s), int(92 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52 * s,
                C_TEXT if seen else C_DIM, 1, cv2.LINE_AA)
    return canvas


def snapshot_wanted(t, scored_list, wrong, n_keep=10):
    if t in wrong:
        return True
    if not scored_list:
        return t == 0
    if t == scored_list[0] or t == scored_list[-1]:
        return True
    step = max(len(scored_list) // n_keep, 1)
    return t in scored_list[::step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--train", default=None,
                    help="optional development cache (2012). Omit for arcade-only.")
    ap.add_argument("--peers", nargs="*", default=[],
                    help="other clips from the same source to train on")
    ap.add_argument("--out", default=None,
                    help="overlay mp4. Default: <cache>/overlay.mp4")
    ap.add_argument("--snapshots", default=None,
                    help="still-frame dir. Default: <cache>/snapshots")
    ap.add_argument("--title", default=None)
    ap.add_argument("--features", default="both", choices=list(SETS))
    ap.add_argument("--penalty", type=float, default=3.0)
    ap.add_argument("--dwell", type=float, default=0.0)
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()

    train = []
    if args.train and Path(args.train).exists():
        train = attach_dataset(dataset.load(args.train, source="train"))
    elif args.train:
        print(f"train cache missing ({args.train}); using peers only")
    if args.peers:
        train += attach_dataset(dataset.load_many(
            {Path(p).name: p for p in args.peers}), min_rows=MIN_ROWS_ARCADE)
    if not train:
        print("no training shots: pass --peers and/or an existing --train cache")
        return

    which = SETS[args.features]
    X, y = stack(train, which)
    model = make_model(args.model).fit(X, y)

    shots = attach_dataset(dataset.load(args.cache, source="clip"), min_rows=1)
    if not shots:
        print(f"{args.cache}: no usable shot")
        return
    sh = shots[0]
    _, _, y_te, grp, tid = sh["ds"][:5]
    p = model.predict_proba(features_of(sh, which))[:, 1]
    score = np.log(np.clip(p, 1e-6, 1 - 1e-6))
    pick = viterbi_by_frame(score, grp, tid, args.penalty)

    pred = [None] * sh["n"]
    for t, v in pick.items():
        pred[t] = v
    pred = ball.blend_with_lowest(pred, sh)
    if args.dwell > 0:
        pred = carrier.enforce_dwell(pred, int(round(args.dwell * sh["fps"])))

    truth = {int(t): int(tid[grp == t][np.argmax(y_te[grp == t])])
             for t in np.unique(grp)}
    scored = set(pick)
    scored_list = sorted(scored)
    wrong = {t for t in scored_list if pred[t] != truth.get(t)}

    cache = Path(args.cache)
    out = Path(args.out) if args.out else cache / "overlay.mp4"
    snap_dir = Path(args.snapshots) if args.snapshots else cache / "snapshots"
    out.parent.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)
    for old in snap_dir.glob("*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sh["d"]["frame_a"])
    writer = None
    title = args.title or Path(args.video).stem
    frames_log = []
    n_snaps = 0

    seen = hits = 0
    for t in range(sh["n"]):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     sh["fps"], (w, h))
        pt, gt = pred[t], truth.get(t)
        if t in scored:
            seen += 1
            hits += int(pt == gt)
        b = ball.localize_on_player(frame, sh["d"]["per_frame"][t], pt)
        canvas = draw_frame(frame, sh, t, pred, truth, scored, seen, hits, title,
                            ball=b)
        writer.write(canvas)

        if snapshot_wanted(t, scored_list, wrong):
            tag = "wrong" if t in wrong else ("ok" if t in scored else "wait")
            cv2.imwrite(str(snap_dir / f"f{t:04d}_{tag}.jpg"), canvas,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            n_snaps += 1

        frames_log.append({
            "t": t,
            "pred": None if pt is None else int(pt),
            "label": None if gt is None else int(gt),
            "scored": t in scored,
            "agree": bool(t in scored and pt == gt),
            "ball": None if b is None else [round(b[0], 1), round(b[1], 1)],
        })

    cap.release()
    if writer:
        writer.release()

    summary = {
        "video": str(args.video),
        "cache": str(cache),
        "overlay": str(out),
        "snapshots": str(snap_dir),
        "n_snapshots": n_snaps,
        "scored_frames": seen,
        "hits": hits,
        "accuracy": round(hits / max(seen, 1), 4),
        "wrong_frames": sorted(wrong),
        "features": args.features,
        "penalty": args.penalty,
        "dwell": args.dwell,
        "frames": frames_log,
    }
    (cache / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"{out}  {seen} scored frames, agreement {hits/max(seen,1):.1%}")
    print(f"  snapshots {n_snaps} -> {snap_dir}")
    print(f"  wrote {cache / 'results.json'}")


if __name__ == "__main__":
    main()
