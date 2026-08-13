# ballcarrier

Identifying which player is carrying the ball, frame by frame, in NFL Blitz
gameplay video — and measuring how well that actually works.

## Answer

**No, not reliably.** On held-out plays the best model gets the carrier right
on **63.7%** of live-play frames (95% CI [0.56, 0.71]) against a chance rate of
14.1% — about 4.5× chance, and roughly one frame in three wrong. **Possession
changes are not detected at all**: at the operating point that maximises
accuracy, precision is 0.17, so five out of six announced turnovers are false.

The project's founding claim — that possession is recoverable from trajectories
alone — is **not supported**. Trajectory features reach 36.0%; features
describing only where the camera has put a player on screen reach 55.2%. The
camera, which follows the ball, carries more of the answer than the motion does.

## How the labels exist at all

NFL Blitz draws a blue ellipse under the player currently under user control.
`src/hud.py` finds it, which yields a carrier label on every frame for free —
no manual annotation. That is the only reason this question is answerable at
this scale.

The label is a *proxy*: it marks the controlled player, not provably the ball
carrier. To check it, `scripts/audit_sample.py` drew 36 uniformly random
labelled frames at native resolution and they were inspected one by one. On
every live-play frame where the ball was visible, the reticle was on the player
holding it. The failures were all of one kind: pre-snap and post-tackle frames,
where the reticle marks a *selected* player and "ball carrier" is not defined.
Hence the live-play gate below.

Three detector bugs were found and fixed by looking at images rather than at
metrics, and each had been reporting success:

| symptom | cause |
|---|---|
| spurious carrier jumps mid-play | the blue **line of scrimmage** painted on the turf crosses every player's feet, satisfying the "at a player's feet" anchor |
| carrier jumps to the corner | the **TURBO bar** whenever a player ran in front of it |
| static-HUD suppression never fired | threshold was 0.60; the highest occupancy anywhere in 9,764 frames is **0.592**, so the mask was empty and the filter was inert from the day it was written |

## Where the numbers apply

Possession is only defined during a live play, so a frame is scored only if the
field is on camera, players are in motion, a label exists, and at least three
players are tracked to choose between. That last condition matters: applied to
the model but not the baselines, it silently compared 328 frames at chance 0.16
against 483 frames at chance 0.32. `dataset.scored_mask` now applies it to
everything.

Possession changes are defined **spatially** — the indicator jumps more than a
player's width to a *different* tracked player — never as "the label's track id
changed". A tracker reassigns ids to the same player constantly, and counting
those as turnovers would invent dozens per play.

## Results

Two sources. Training footage is the 2012 Xbox Live Arcade re-release,
163s, 1280×720/60fps, 12 usable shots. Test footage is four clips of the
**1997 arcade original**, 1356×1016/30fps — different sprites, teal field,
different camera. Every threshold was read off the first and is rescaled for
the second by `evaluate.scales`.

**Held out by shot, never by frame.** Adjacent frames of one play are nearly
the same picture; a random frame split lets a model memorise a play and be
tested on it.

### Per-frame carrier accuracy

| method | EA-era, leave-one-shot-out | 1997 arcade, never seen in training |
|---|---|---|
| chance (uniform over candidates) | 0.141 | 0.160 |
| existing heuristic (conv + speed + sep, Viterbi) | **0.095** — *below chance* | 0.133 |
| fastest player | 0.127 | 0.152 |
| nearest frame centre | 0.361 | 0.363 |
| lowest on screen | 0.316 | **0.616** |
| learned, trajectory features only | 0.360 | 0.540 |
| learned, screen-position features only | 0.552 | 0.524 |
| **learned, both** | **0.637** [0.556, 0.711] | **0.665** [0.462, 0.912] |

3,667 scored frames (EA) and 328 (arcade). Intervals are shot-level bootstraps
— the shot is the unit that varies, not the frame.

### Possession changes

19 ground-truth events in the EA footage. Sweeping the switch penalty trades
one failure for the other and never gets both:

| penalty | accuracy | change precision | change recall |
|---|---|---|---|
| 0.0 | 0.608 | 0.05 | 0.95 |
| 1.5 | 0.637 | 0.17 | 0.79 |
| 6.0 | 0.637 | 0.24 | 0.68 |

### Tracking stability

The carrier's track id changes on **3.9%** of frames where the carrier did not
change. Detection and tracking are not the bottleneck; deciding *which* box is
the carrier is.

## Pipeline

```
video ──► shot cuts ──► detect + track ──► trajectories ──┐
             │                                            ├──► score ──► decode ──► carrier
      refuse to reason          HUD reticle ──► labels ───┘        possession is
      across a cut                                                piecewise constant
```

| module | concern |
|---|---|
| `src/segment.py` | camera cuts, and where the play snaps |
| `src/tracking.py` | detection, association, trajectories |
| `src/hud.py` | harvesting carrier labels from the on-screen indicator |
| `src/features.py` | candidate features, split into trajectory vs. screen-position |
| `src/carrier.py` | the original hand-built heuristic and the Viterbi decode |
| `src/evaluate.py` | the evaluation domain and the metrics |
| `src/dataset.py` | cache loading, per-clip scale, the scored-frame rule |

## Run it

```bash
# 1. cache detections, tracks and harvested labels for a video
python scripts/harvest_video.py --video path/to.mp4 --out data/interim/cache

# 2. baselines and the original heuristic
python scripts/evaluate.py --cache data/interim/cache --label ea

# 3. learned model, leave-one-shot-out
python scripts/train_eval.py --cache data/interim/cache

# 4. learned model, trained here and tested on another source entirely
python scripts/train_eval.py --cache data/interim/cache \
    --test data/interim/arcade/clip1 data/interim/arcade/clip2

# 5. demo overlay, prediction and ground truth side by side
python scripts/demo.py --video clip.mp4 --cache data/interim/arcade/clip1 \
    --out data/interim/demo/clip1.mp4
```

`scripts/audit_sample.py` and `scripts/audit_changes.py` render the images the
label claims rest on. They are the most useful scripts here and should be run
before believing any number above.

## What these numbers do not cover

- **One training video.** 163 seconds, one matchup, one camera style. Twelve
  shots is twelve effective samples, which is why the intervals are wide.
- **The arcade result rests on 328 frames** across four clips, CI [0.46, 0.91].
  It shows the approach survives a change of game generation; it does not
  establish an accuracy. Per clip: 97.6%, 68.5%, 65.1%, 41.1% — the weakest is
  the deliberate camera-swing stress test.
- **19 possession-change events** is too few to characterise change detection
  beyond "it does not work yet".
- **Labels are the controlled player**, audited as equal to the ball carrier on
  live-play frames, not proven equal on every frame.
- On the arcade clips a one-line baseline — lowest player on screen — scores
  0.616 against the learned model's 0.665. On that footage the model has not
  yet earned its complexity.

## Not used

Roughly 400 timestamped action annotations exist from earlier work. They record
*when* something happened; this is a spatial question about *which* player.
