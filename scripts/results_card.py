"""Render a single summary image of the headline results.

A table in a terminal does not survive being pasted into a slide, and a number
without its chance rate or its interval is not a result. This draws all three
together so the figure cannot be quoted without them.

    python scripts/results_card.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H = 1660, 960
BG = (26, 29, 31)
FG = (238, 240, 238)
MUTE = (146, 152, 150)
ACCENT = (70, 190, 250)
GOOD = (110, 215, 130)
BAD = (78, 78, 236)
GRID = (52, 58, 56)
F = cv2.FONT_HERSHEY_DUPLEX
FS = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, x, y, scale=0.6, col=FG, font=FS, thick=1):
    cv2.putText(img, s, (x, y), font, scale, col, thick, cv2.LINE_AA)


def bar(img, x, y, w, h, frac, col, chance=None):
    cv2.rectangle(img, (x, y), (x + w, y + h), GRID, -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(frac, 0)), y + h), col, -1)
    if chance is not None:
        cx = x + int(w * chance)
        cv2.line(img, (cx, y - 4), (cx, y + h + 4), (200, 200, 200), 1)


def load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/interim")
    ap.add_argument("--penalty", default="3.0")
    ap.add_argument("--out", default="data/interim/results_card.png")
    args = ap.parse_args()

    d = Path(args.dir)
    ea_base = load(d / "eval_pooled.json", {})   # same footage as the headline
    pooled = load(d / "model_eval_pooled_cam.json", {})
    within = load(d / "model_eval_vid1_loso.json", {})
    across = load(d / "model_eval_v1_to_v2.json", {})
    clips = load(d / "model_eval_vid1_to_clips.json", {})

    img = np.full((H, W, 3), BG, np.uint8)
    text(img, "Ball carrier identification in NFL Blitz", 50, 66, 1.0, FG, F)
    text(img, "every figure is on plays the model never saw; the white tick "
              "on each bar is chance", 50, 100, 0.58, MUTE)

    # Keys are "penalty_P_dwell_D". Match on the penalty and take the
    # zero-dwell variant, rather than hardcoding a key format that has already
    # changed once and silently emptied half this figure.
    want = f"penalty_{float(args.penalty)}_"

    def get(blob, fs="both"):
        try:
            entries = blob["sets"][fs]
        except (KeyError, TypeError):
            return None
        for k, v in entries.items():
            if k.startswith(want) and k.endswith("_0.0"):
                return v
        for k, v in entries.items():
            if k.startswith(want):
                return v
        return None

    rows = []
    if pooled:
        rows.append(("Pooled, 8-fold CV over 112 plays", get(pooled),
                     pooled.get("chance"), "21,997 frames -- the headline"))
    if within:
        rows.append(("Held-out play, same capture", get(within),
                     within.get("chance"), "8,377 frames / 37 plays"))
    if across:
        rows.append(("Held-out capture entirely", get(across),
                     across.get("chance"), "13,611 frames / 73 plays"))
    if clips:
        rows.append(("Held-out clips, different resolution", get(clips),
                     clips.get("chance"), "337 frames / 4 clips"))

    y = 160
    text(img, "PER-FRAME CARRIER ACCURACY", 50, y, 0.62, ACCENT, F)
    y += 40
    for name, r, chance, note in rows:
        if not r:
            continue
        acc, ci = r["accuracy"], r.get("ci95", [0, 0])
        text(img, name, 50, y + 18, 0.58, FG)
        text(img, note, 50, y + 44, 0.48, MUTE)
        bar(img, 620, y, 560, 30, acc, ACCENT, chance)
        text(img, f"{acc:.1%}", 1200, y + 23, 0.72, FG, F)
        text(img, f"95% CI [{ci[0]:.0%}, {ci[1]:.0%}]    chance {chance:.0%}",
             1298, y + 21, 0.44, MUTE)
        y += 74

    y += 16
    text(img, "WHERE THE SIGNAL COMES FROM  (pooled, 112 plays)",
         50, y, 0.62, ACCENT, F)
    y += 38
    if pooled:
        for fs, label in [("traj", "trajectories only  (the original claim)"),
                          ("pos", "screen position only  (the camera)"),
                          ("both", "both"),
                          ("both+ball", "both + ball appearance")]:
            r = get(pooled, fs)
            if not r:
                continue
            text(img, label, 70, y + 20, 0.54,
                 FG if fs.startswith("both") else MUTE)
            bar(img, 620, y + 2, 560, 24, r["accuracy"],
                ACCENT if fs.startswith("both") else MUTE,
                pooled.get("chance"))
            text(img, f"{r['accuracy']:.1%}", 1200, y + 21, 0.6, FG, F)
            y += 40

    y += 20
    text(img, "BASELINES AND FAILURE MODES", 50, y, 0.62, ACCENT, F)
    y += 36
    lines = []
    if ea_base:
        s = ea_base.get("summary", {})
        for k, lab in [("heuristic (conv+speed+sep, viterbi)",
                        "prior hand-built heuristic"),
                       ("nearest frame centre", "nearest frame centre"),
                       ("lowest on screen", "lowest player on screen")]:
            if k in s:
                lines.append((lab, s[k]["accuracy"], ea_base.get("chance")))
    for lab, v, ch in lines:
        col = BAD if v < ch else MUTE
        text(img, lab, 70, y + 20, 0.54, MUTE)
        bar(img, 620, y + 2, 560, 24, v, col, ch)
        text(img, f"{v:.1%}", 1200, y + 21, 0.6, col, F)
        y += 38

    y += 12
    chg = get(pooled)
    if chg:
        c = chg["changes"]
        text(img, f"Possession changes: precision {c['precision']:.2f}, "
                  f"recall {c['recall']:.2f} over "
                  f"{pooled.get('true_changes', 0)} true events -- usable, "
                  f"not solved.", 50, y + 20, 0.56, GOOD)

    cv2.imwrite(str(args.out), img)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
