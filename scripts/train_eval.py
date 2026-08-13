"""Train a carrier classifier on harvested labels; score it on held-out footage.

Held out by *shot*, never by frame. Neighbouring frames of one play are nearly
the same picture, so a random frame split would let the model memorise a play
and then be tested on it, which is the standard way this problem is accidentally
solved on paper and not in fact.

Three regimes, and the differences between them are the interesting result:

  --cache only                        leave-one-shot-out within one source
  --cache + --test                    train on one source, test on another
  --cache + --test + --include-...    leave-one-clip-out, in-domain peers

The middle one is the hardest and the most honest. The training source is the
2012 re-release at 1280x720/60fps; the test source is the 1997 arcade original
at 1356x1016/30fps, with different sprites, a different field colour and a
different camera. Nothing about that generation was available at training time.

The third is what a deployed system would actually have: the held-out clip is
still never trained on, but its siblings are. Reporting the middle number
alone understates a fielded system; reporting the third alone overstates how
well the method travels to footage unlike anything it has seen. Both are
printed, and the README carries both.

Three feature sets are trained, because the project's claim is specifically
about trajectories:

  traj   motion only -- the claim under test
  pos    where the camera has put the player on screen -- the confound
  both   everything

If `pos` alone matches `both`, the trajectories are contributing nothing and
the method is a camera-framing detector wearing a trajectory costume.

    python scripts/train_eval.py --cache data/interim/cache
    python scripts/train_eval.py --cache data/interim/cache \\
        --test data/interim/arcade/clip1 data/interim/arcade/clip2
    python scripts/train_eval.py --cache data/interim/cache \\
        --test data/interim/arcade/clip1 ... --include-test-in-train

`--dwell` sets the minimum time a carrier must be held, in seconds. It is a
decode constraint, not a model change, so it is fixed on the development
source and then applied unchanged to the held-out one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src import carrier, dataset, evaluate as ev, features  # noqa: E402

MIN_ROWS = 120
SETS = {"traj": 0, "pos": 1, "both": None}


def attach_dataset(shots):
    for s in shots:
        s["ds"] = features.build_dataset(
            s["pos"], s["vel"], s["d"]["per_frame"], s["d"]["labels"],
            s["mask"], s["n"], s["meta"]["shot"],
            w=s["w"], h=s["h"], fps=s["fps"])
    return [s for s in shots if len(s["ds"][2]) >= MIN_ROWS]


def make_model(kind):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=250, learning_rate=0.06,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0)


# The time-averaged trajectory features are the last four columns of the
# trajectory block, so switching them off is a slice rather than a second
# feature builder -- the ablation is identical to the full model in every
# other respect.
#
# They are off by default because they were measured and did not earn their
# place. Half-second averages of convergence, speed, separation and crowding
# were added on the theory that a single frame cannot separate a carrier from
# a lead blocker while half a second of pursuit can. On the development source
# they lift the trajectory-only model (0.353 -> 0.371) and *cost* the combined
# model (0.631 -> 0.607). The intervals overlap, so this is not proof they are
# harmful; it is an absence of evidence that they help, and the simpler
# feature set wins by default. `--temporal` reproduces the measurement.
DROP_TEMPORAL = True
N_TEMPORAL = 4


def _trim(Xt):
    return Xt[:, :-N_TEMPORAL] if DROP_TEMPORAL else Xt


def stack(shots, which):
    Xt = _trim(np.vstack([s["ds"][0] for s in shots]))
    Xp = np.vstack([s["ds"][1] for s in shots])
    y = np.concatenate([s["ds"][2] for s in shots])
    if which == 0:
        return Xt, y
    if which == 1:
        return Xp, y
    return np.hstack([Xt, Xp]), y


def features_of(shot, which):
    Xt, Xp = _trim(shot["ds"][0]), shot["ds"][1]
    return Xt if which == 0 else Xp if which == 1 else np.hstack([Xt, Xp])


def argmax_by_frame(score, group, tids):
    out = {}
    for g in np.unique(group):
        sel = group == g
        out[int(g)] = int(tids[sel][int(np.argmax(score[sel]))])
    return out


def viterbi_by_frame(score, group, tids, penalty):
    """Same emissions, decoded with the switch penalty possession implies."""
    frames = sorted(int(g) for g in np.unique(group))
    scores = {}
    for g in frames:
        sel = group == g
        scores[g] = {int(i): float(v) for i, v in zip(tids[sel], score[sel])}
    dense = [scores.get(t, {}) for t in range(max(frames) + 1)]
    path = carrier.viterbi(dense, switch_penalty=penalty)
    return {t: path[t] for t in frames if path[t] is not None}


def bootstrap_ci(per_shot, iters=2000, seed=0):
    """Shot-level bootstrap of the pooled accuracy.

    Resampling frames would give a hopelessly narrow interval: adjacent frames
    of one play are not independent observations. The shot is the unit that
    actually varies, so the interval is wide -- which is the honest report,
    not a defect of the method.
    """
    rng = np.random.default_rng(seed)
    a = np.array([r["hits"] for r in per_shot], float)
    b = np.array([r["frames"] for r in per_shot], float)
    if len(a) < 2:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(iters):
        i = rng.integers(0, len(a), len(a))
        d = b[i].sum()
        out.append(a[i].sum() / d if d else np.nan)
    return (float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)))


def score_fold(train, test_shot, model_kind):
    """Fit on `train`, return log-probabilities for `test_shot` per feature set."""
    out = {}
    for name, which in SETS.items():
        X, y = stack(train, which)
        m = make_model(model_kind).fit(X, y)
        p = m.predict_proba(features_of(test_shot, which))[:, 1]
        out[name] = np.log(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/interim/cache")
    ap.add_argument("--test", nargs="*", default=None,
                    help="cache dirs to test on; omit for leave-one-shot-out")
    ap.add_argument("--include-test-in-train", action="store_true",
                    help="also train on the test source, minus the held-out "
                         "clip. Leave-one-clip-out rather than cross-domain.")
    ap.add_argument("--temporal", action="store_true",
                    help="include the time-averaged trajectory features; "
                         "off by default, see DROP_TEMPORAL")
    ap.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
    ap.add_argument("--penalties", default="0.0,1.5,3.0,6.0")
    ap.add_argument("--dwell", default="0.0",
                    help="minimum seconds a carrier must be held; comma-"
                         "separated to sweep. Applied to cached scores, so a "
                         "sweep costs no extra training.")
    ap.add_argument("--out", default="data/interim/model_eval.json")
    args = ap.parse_args()

    global DROP_TEMPORAL
    DROP_TEMPORAL = not args.temporal
    penalties = [float(x) for x in args.penalties.split(",")]
    dwells = [float(x) for x in str(args.dwell).split(",")]
    base = attach_dataset(dataset.load(args.cache, source="train"))

    if args.test:
        test_pool = attach_dataset(dataset.load_many(
            {Path(p).name: p for p in args.test}))
        if args.include_test_in_train:
            folds = [(base + [o for o in test_pool if o is not s], s)
                     for s in test_pool]
            regime = (f"leave-one-clip-out over {len(test_pool)} test clips, "
                      f"training also on {args.cache}")
        else:
            folds = [(base, s) for s in test_pool]
            regime = (f"cross-domain: train={args.cache} -> "
                      f"{len(test_pool)} held-out clips, no in-domain data")
    else:
        folds = [([o for o in base if o is not s], s) for s in base]
        regime = f"leave-one-shot-out within {args.cache}"

    print(f"{regime}\n{len(folds)} folds, dwell={args.dwell}s\n")

    chance_n, chance_d, total_true = 0.0, 0, 0
    cached = []
    for train, held in folds:
        _, _, y_te, grp_te, tid_te = held["ds"]
        for t in np.unique(grp_te):
            chance_n += 1.0 / int((grp_te == t).sum())
            chance_d += 1
        truth = {int(t): int(tid_te[grp_te == t][np.argmax(y_te[grp_te == t])])
                 for t in np.unique(grp_te)}
        true_ev = ev.true_changes(held["d"]["reticle"], held["mask"],
                                  held["d"]["labels"], scale=held["sc"]["space"])
        total_true += len(true_ev)
        cached.append({"key": held["key"], "held": held, "truth": truth,
                       "true_ev": true_ev, "grp": grp_te, "tid": tid_te,
                       "scores": score_fold(train, held, args.model)})
        print(f"  {held['key']}: {len(truth)} test frames, "
              f"{len(true_ev)} true possession changes")

    chance = chance_n / max(chance_d, 1)
    summary = {"regime": regime, "chance": round(chance, 4),
               "model": args.model, "true_changes": total_true, "sets": {}}

    print(f"\nchance = {chance:.3f}   {total_true} true possession changes\n")
    print(f"{'feats':6s} {'pen':>5s} {'dwell':>6s} {'frames':>7s} {'acc':>6s} "
          f"{'95% CI':>15s} {'lift':>6s} {'chgP':>6s} {'chgR':>6s}")

    for name in SETS:
        summary["sets"][name] = {}
        for pen, dwell in [(p_, d_) for p_ in penalties for d_ in dwells]:
            per_shot, tp, fp, fn = [], 0, 0, 0
            for e in cached:
                pick = (argmax_by_frame(e["scores"][name], e["grp"], e["tid"])
                        if pen == 0.0 else
                        viterbi_by_frame(e["scores"][name], e["grp"],
                                         e["tid"], pen))
                pred = [None] * e["held"]["n"]
                for t, p in pick.items():
                    pred[t] = p
                sc = e["held"]["sc"]
                if dwell > 0:
                    pred = carrier.enforce_dwell(
                        pred, int(round(dwell * e["held"]["fps"])))
                # Scored over the frames the model was asked about, not the
                # frames it still has an answer for: dwell can blank a frame,
                # and dropping those from the denominator would turn a refusal
                # to answer into free accuracy.
                scored_frames = list(pick)
                hits = sum(1 for t in scored_frames
                           if pred[t] == e["truth"].get(t))
                pe = ev.pred_changes(pred, e["held"]["d"]["per_frame"],
                                     e["held"]["mask"], scale=sc["space"])
                mm = ev.match_events(e["true_ev"], pe,
                                     frames_scale=sc["frames"])
                tp += mm["tp"]; fp += mm["fp"]; fn += mm["fn"]
                per_shot.append({"key": e["key"], "hits": hits,
                                 "frames": len(scored_frames),
                                 "accuracy": round(
                                     hits / max(len(scored_frames), 1), 3)})

            n = sum(r["frames"] for r in per_shot)
            acc = sum(r["hits"] for r in per_shot) / max(n, 1)
            lo, hi = bootstrap_ci(per_shot)
            prec = tp / max(tp + fp, 1e-9)
            rec = tp / max(tp + fn, 1e-9)
            print(f"{name:6s} {pen:5.2f} {dwell:6.2f} {n:7d} {acc:6.3f} "
                  f"[{lo:.3f},{hi:.3f}] {acc/chance:5.2f}x "
                  f"{prec:6.3f} {rec:6.3f}")
            summary["sets"][name][f"penalty_{pen}_dwell_{dwell}"] = {
                "frames": n, "accuracy": round(acc, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "lift": round(acc / chance, 2),
                "changes": {"tp": tp, "fp": fp, "fn": fn,
                            "precision": round(prec, 3),
                            "recall": round(rec, 3)},
                "per_shot": per_shot,
            }
        print()

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
