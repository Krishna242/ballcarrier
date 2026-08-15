"""Score the trajectory heuristic, and several baselines, against harvested labels.

Baselines are here because an accuracy number in isolation is unreadable. The
heuristic has to beat uniform choice among the visible players to have any
signal at all, and it has to beat "whoever is moving fastest" -- a one-line
rule needing no scoring, no normalisation and no Viterbi -- to justify its own
existence. "Nearest frame centre" is the one that matters most: the camera in
this game follows the ball, so a method that reads position off the screen is
being handed part of the answer, and any trajectory method that cannot beat it
has not earned its complexity.

    python scripts/evaluate.py --cache data/interim/cache
    python scripts/evaluate.py --cache data/interim/arcade/clip1 ... --label arcade
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ball, carrier, dataset, evaluate as ev, segment  # noqa: E402


def predict_heuristic(sh):
    pos, vel, n = sh["pos"], sh["vel"], sh["n"]
    scores = carrier.score_frames(pos, vel, n)
    snap, _ = segment.detect_snap(vel, n)
    if snap is not None:
        scores = [{} if t < snap else s for t, s in enumerate(scores)]
    return carrier.viterbi(scores)


def _argmax_over_candidates(sh, key):
    """Pick the best candidate per frame, over the canonical candidate set.

    Every baseline goes through here so that all of them -- and the chance
    rate -- are choosing from the same players.
    """
    out = []
    for t in range(sh["n"]):
        cand = dataset.candidates(sh, t)
        if not cand:
            out.append(None)
            continue
        out.append(max(cand, key=lambda tid: key(sh, tid, t)))
    return out


def predict_fastest(sh):
    return _argmax_over_candidates(
        sh, lambda s, tid, t: np.linalg.norm(s["vel"][tid][t]))


def predict_lowest(sh):
    return _argmax_over_candidates(sh, lambda s, tid, t: s["pos"][tid][t][1])


def predict_centre(sh):
    return _argmax_over_candidates(
        sh, lambda s, tid, t: -abs(s["pos"][tid][t][0] - s["w"] / 2.0))


def predict_pan_lowest(sh):
    """Lowest-on-screen, but only on camera-pan frames; else heuristic."""
    base = predict_heuristic(sh)
    return ball.blend_with_lowest(base, sh)


PREDICTORS = {
    "heuristic (conv+speed+sep, viterbi)": predict_heuristic,
    "fastest player": predict_fastest,
    "lowest on screen": predict_lowest,
    "nearest frame centre": predict_centre,
    "heuristic + lowest if pulled back": predict_pan_lowest,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", default=["data/interim/cache"])
    ap.add_argument("--label", default="ea")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shots = dataset.load_many({Path(p).name: p for p in args.cache})

    totals = {k: [0, 0] for k in PREDICTORS}
    events = {k: [0, 0, 0] for k in PREDICTORS}
    chance_num, chance_den, churn = 0.0, 0, [0, 0]
    n_true, per_shot = 0, []

    for sh in shots:
        d, sc = sh["d"], sh["sc"]
        mask = dataset.scored_mask(sh)
        n_lab = int(mask.sum())
        if n_lab < 10:
            per_shot.append({"key": sh["key"], "labelled_live": n_lab,
                             "skipped": "too few labelled live frames"})
            continue

        n_cand = [len(dataset.candidates(sh, t)) for t in range(sh["n"])]
        ch = ev.chance_rate(n_cand, d["labels"], mask)
        chance_num += ch * n_lab
        chance_den += n_lab

        c = ev.identity_churn(d["reticle"], d["labels"], mask,
                              scale=sc["space"])
        churn[0] += c["id_changes"]
        churn[1] += c["same_player_frames"]

        true_ev = ev.true_changes(d["reticle"], mask, d["labels"],
                                  scale=sc["space"])
        n_true += len(true_ev)

        row = {"key": sh["key"], "labelled_live": n_lab, "chance": round(ch, 4),
               "candidates_mean": round(float(np.mean(
                   [n_cand[t] for t in range(sh["n"]) if mask[t]])), 1),
               "true_possession_changes": len(true_ev),
               "carrier_id_churn": round(c["churn_rate"], 3), "methods": {}}

        for name, fn in PREDICTORS.items():
            pred = fn(sh)
            acc = ev.frame_accuracy(pred, d["labels"], mask)
            pe = ev.pred_changes(pred, d["per_frame"], mask, scale=sc["space"])
            m = ev.match_events(true_ev, pe, frames_scale=sc["frames"])
            row["methods"][name] = {
                "n": acc["n"],
                "accuracy": round(acc["accuracy"], 4) if acc["n"] else None,
            }
            if acc["n"]:
                totals[name][0] += acc["accuracy"] * acc["n"]
                totals[name][1] += acc["n"]
            events[name][0] += m["tp"]
            events[name][1] += m["fp"]
            events[name][2] += m["fn"]
        per_shot.append(row)

    chance = chance_num / max(chance_den, 1)
    print(f"\n[{args.label}]  {len(shots)} shots, chance {chance:.3f}, "
          f"{n_true} true possession changes\n")
    print(f"{'':38s} {'frames':>7s} {'acc':>7s} {'lift':>6s} "
          f"{'chgP':>6s} {'chgR':>6s}")
    summary = {}
    for name in PREDICTORS:
        hit, tot = totals[name]
        acc = hit / max(tot, 1)
        tp, fp, fn = events[name]
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        print(f"{name:38s} {tot:7d} {acc:7.3f} {acc/max(chance,1e-9):5.2f}x "
              f"{prec:6.3f} {rec:6.3f}")
        summary[name] = {
            "frames": tot, "accuracy": round(acc, 4),
            "lift": round(acc / max(chance, 1e-9), 2),
            "changes": {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(prec, 3), "recall": round(rec, 3)},
        }

    print(f"\ncarrier track-id churn while the carrier did not change: "
          f"{churn[0]}/{churn[1]} = {churn[0]/max(churn[1],1):.1%} of frames")

    out = args.out or f"data/interim/eval_{args.label}.json"
    Path(out).write_text(json.dumps(
        {"label": args.label, "chance": round(chance, 4),
         "true_changes": n_true, "summary": summary,
         "identity_churn": {"changes": churn[0], "frames": churn[1],
                            "rate": round(churn[0] / max(churn[1], 1), 4)},
         "per_shot": per_shot}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
