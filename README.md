# ballcarrier

Identifying which player is carrying the ball, frame by frame, in NFL Blitz
gameplay video — and measuring how well that actually works.

## Answer

**Partly, and not reliably enough to trust unsupervised.** On plays it has
never seen, the model identifies the carrier on **63.1%** of live-play frames
in the 2012 re-release (95% CI [0.55, 0.72]) against a 13.4% chance rate, and
**67.3%** on 1997 arcade clips ([0.56, 0.79], chance 16.6%). That is roughly
4.7× chance and roughly one frame in three wrong.

**Possession changes are not detected.** At the chosen operating point,
precision is 0.26 over 31 true events — three of every four announced
turnovers are false. No setting of the decode gives usable precision and
recall together.

The project's founding claim — that possession is recoverable from
trajectories alone — is **not supported**. Trajectory features reach 35.3%;
features describing only where the camera has put a player on screen reach
50.8%. The camera, which follows the ball, carries more of the answer than the
motion does.

![results](docs/results_card.png)

Demo overlays for the four 1997 arcade clips are in [`docs/demo/`](docs/demo).
Each is drawn by a model trained on the 2012 footage plus the *other three*
clips, never on the clip being drawn. Amber is the prediction, green marks the
game's own indicator, and the box turns red and says `<- wrong` whenever the
two disagree — so the failures are as visible as the successes.

## How the labels exist at all

NFL Blitz draws a blue ellipse under the player under user control, in both the
1997 arcade original and the 2012 re-release. `src/hud.py` finds it, which
yields a carrier label on every frame for free — no manual annotation. That is
the only reason this question is answerable at this scale.

The label is a *proxy*: it marks the controlled player, not provably the ball
carrier. `scripts/audit_sample.py` drew 36 uniformly random labelled frames at
native resolution and they were inspected one by one. On every live-play frame
where the ball was visible, the reticle was on the player holding it. The
failures were all of one kind — pre-snap and post-tackle frames, where the
reticle marks a *selected* player and "ball carrier" is not defined. Hence the
live-play gate.

Four detector bugs were found by looking at frames rather than at metrics, and
every one of them had been reporting success:

| symptom | cause |
|---|---|
| carrier jumps mid-play | the blue **line of scrimmage** painted on the turf crosses every player's feet, satisfying the "at a player's feet" anchor |
| carrier jumps to the corner | the **TURBO bar**, whenever a player ran in front of it |
| static-HUD suppression never fired | threshold was 0.60; the highest occupancy anywhere in 9,764 frames is **0.592**, so the mask was empty and the filter was inert from the day it was written |
| tackled carriers lost their label | detection at `conf=0.35/imgsz=960` missed players in a pile — exactly when possession is most in doubt. One arcade frame found 1 of 4 visible players, and the missing one had the ball |

## Where the numbers apply

A frame is scored only if the field is on camera, players are in motion, a
label exists, and at least three players are tracked to choose between. That
last condition has to apply to the baselines and the model alike or their
accuracies are quoted over different footage; `dataset.scored_mask` enforces it
in one place.

Possession changes are defined **spatially** — the indicator jumps more than a
player's width, to a *different* tracked player — never as "the label's track
id changed". A tracker reassigns ids to the same player constantly, and
counting those as turnovers invents dozens per play.

## Results

Two sources. Development is the 2012 Xbox Live Arcade re-release: 163s,
1280×720/60fps, 12 usable shots. Test is four clips of the **1997 arcade
original**, 1356×1016/30fps — different sprites, teal field, different camera.
Thresholds were read off the first and are rescaled for the second by
`evaluate.scales`, which separates spatial scale from frame rate because they
pull in opposite directions.

**Held out by shot, never by frame.** Adjacent frames of one play are nearly
the same picture; a random frame split lets a model memorise a play and be
tested on it.

### Per-frame carrier accuracy

| | 2012, leave-one-shot-out | 1997 arcade, cross-domain | 1997 arcade, leave-one-clip-out |
|---|---|---|---|
| chance | 0.134 | 0.166 | 0.166 |
| prior hand-built heuristic | 0.117 | 0.235 | — |
| fastest player | 0.128 | 0.186 | — |
| lowest on screen | 0.299 | **0.581** | — |
| nearest frame centre | 0.343 | 0.273 | — |
| learned, trajectory only | 0.353 | 0.541 | 0.600 |
| learned, screen position only | 0.508 | 0.476 | 0.532 |
| **learned, both** | **0.631** [0.55, 0.72] | **0.550** [0.37, 0.76] | **0.673** [0.56, 0.79] |

3,787 scored frames (2012) and 340 (arcade). Intervals are shot-level
bootstraps — the shot is the unit that varies, not the frame.

Cross-domain means no arcade footage was in training at all. Leave-one-clip-out
means the held-out clip is still never trained on, but its three siblings are —
what a deployed system would actually have. Per clip: 83.7%, 57.6%, 72.3%,
55.9%.

### Possession changes

31 ground-truth events. The switch penalty trades one failure for the other
and never gets both:

| penalty | accuracy | change precision | change recall |
|---|---|---|---|
| 0.0 | 0.599 | 0.08 | 0.90 |
| 3.0 *(chosen)* | 0.631 | 0.26 | 0.48 |
| 10.0 | 0.663 | 0.30 | 0.32 |

Accuracy keeps rising with the penalty, but only by refusing to switch at all.
The operating point is fixed at 3.0 on the development source and applied
unchanged to the arcade clips.

### Two ideas that were tried and did not work

Both are kept in the code behind flags, because a negative result that is
deleted gets re-proposed.

- **Minimum dwell time** (`carrier.enforce_dwell`, `--dwell`). A hard floor on
  how long a carrier must be held, aimed at flicker. It raises change precision
  from 0.09 to 0.31 and costs recall 0.97 → 0.32, and it lowers accuracy at
  *every* penalty setting. Not used.
- **Half-second trajectory averages** (`--temporal`). Added on the theory that
  a single frame cannot separate a carrier from a lead blocker while half a
  second of pursuit can. They lift the trajectory-only model (0.353 → 0.371)
  and cost the combined model (0.631 → 0.607). Off by default.

### Tracking stability

The carrier's track id changes on 3.7% of 2012 frames and 10.7% of arcade
frames where the carrier did not change. Detection and tracking are not the
bottleneck on the 2012 footage; deciding *which* box is the carrier is.

## Pipeline

```
video ──► shot cuts ──► detect + track ──► trajectories ──┐
             │                                            ├─► score ─► decode ─► carrier
      refuse to reason          HUD reticle ──► labels ───┘       possession is
      across a cut                                                piecewise constant
```

| module | concern |
|---|---|
| `src/segment.py` | camera cuts, and where the play snaps |
| `src/tracking.py` | detection, association, trajectories |
| `src/hud.py` | harvesting carrier labels from the on-screen indicator |
| `src/features.py` | candidate features, split into trajectory vs. screen position |
| `src/carrier.py` | the original heuristic, Viterbi decode, dwell constraint |
| `src/evaluate.py` | the evaluation domain, the metrics, per-clip scaling |
| `src/dataset.py` | cache loading, the scored-frame rule, the candidate set |

## Run it

```bash
# 1. cache detections, tracks and harvested labels (the only slow step)
python scripts/harvest_video.py --video path/to.mp4 --out data/interim/cache

# 2. every number in this README
python scripts/reproduce.py

# 3. demo overlays, each clip drawn by a model that never saw it
python scripts/make_demos.py

# 4. the summary figure above
python scripts/results_card.py
```

`scripts/audit_sample.py` and `scripts/audit_changes.py` render the images the
label claims rest on. They are the most useful scripts here and should be run
before believing any number above.

## What these numbers do not cover

- **One development video.** 163 seconds, one matchup, one camera style. Twelve
  shots is twelve effective samples, which is why the intervals are wide.
- **The arcade result rests on 340 frames** across four clips. It shows the
  approach survives a change of game generation; it does not establish an
  accuracy to two significant figures.
- **31 possession-change events** is too few to characterise change detection
  beyond "it does not work yet".
- **Labels are the controlled player**, audited as equal to the ball carrier on
  live-play frames, not proven equal on every frame.
- **On the arcade clips a one-line baseline — lowest player on screen — scores
  0.581 against the learned model's cross-domain 0.550.** With in-domain peers
  the model reaches 0.673 and pulls ahead, but on genuinely unseen footage it
  does not beat sorting by screen position.

## Not used

Roughly 400 timestamped action annotations exist from earlier work. They record
*when* something happened; this is a spatial question about *which* player.
