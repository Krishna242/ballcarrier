# ballcarrier

Identifying which player is carrying the ball, frame by frame, in NFL Blitz
gameplay video — from how players move, without detecting the ball.

## The claim being tested

Possession is recoverable from trajectories alone. The defence converges on
whoever has the ball; that convergence is measurable without ever locating an
orange-brown ellipse that is small, occluded, and often in flight.

Two pieces of evidence motivated this. [PathCRF](https://arxiv.org/pdf/2602.12080)
(2026) demonstrates ball-free possession inference from player trajectories in
football. And the previous project's own carrier model weighted its
ball-detection feature at `0.56` against `17.10` for its movement feature — the
ball contributed little even when available.

## Pipeline

```
video ──► shot cuts ──► detect + track ──► trajectories ──► score ──► decode ──► carrier
             │                                                          │
      refuse to reason                                        possession is
      across a cut                                          piecewise constant
```

| module | concern |
|---|---|
| `src/segment.py` | camera cuts, and where the play actually snaps |
| `src/tracking.py` | player detection, association, trajectories |
| `src/carrier.py` | trajectory features, scoring, Viterbi decode |
| `src/hud.py` | harvesting carrier labels from the on-screen indicator |
| `src/render.py` | overlay video |

Each stage reads files and writes files, so any stage can be evaluated alone.
The previous codebase put download, detection, tracking and analysis in one
`video_processor.py`, and when its numbers looked wrong there was no layer to
isolate.

## Run it

```bash
python scripts/run_carrier.py \
    --video data/raw/clip.mp4 --start 19 --duration 6 \
    --weights yolo11m.pt --harvest
```

Outputs `overlay.mp4` and `summary.json` under `data/interim/run/`.

## Where this stands

A prototype runs end to end on real footage. Measured so far:

| observation | value | consequence |
|---|---|---|
| players tracked through a live play | 22 | detection/tracking need no research |
| camera cuts in a 163-second clip | 12 | cuts destroy identity; must not reason across them |
| snap detected vs. annotated play start | 21.7s / 18.8s | play onset comes from motion, not annotations |
| spurious possession changes, before → after snap gating | 27 → 8 | most instability was pre-snap noise |
| median margin between top two candidates | 0.46 | **the heuristic is weak — this is the gap to close** |

It has **not** been evaluated against ground truth, because ground truth does
not yet exist. Current output is a plausible hypothesis, not a verified result.

## The open question

`src/hud.py` is the piece everything else depends on. If the game's
controlled-player reticle can be detected reliably, it yields a free per-frame
carrier label for unlimited footage — which is the whole reason this domain is
worth working in.

It is **not yet validated**. A first attempt using colour alone reported a 100%
hit rate while actually locking onto the TURBO meter, a static HUD element of
similar blue. `hud.validate()` exists specifically to catch that: it rejects
its own output when the detected point does not move, however good the hit rate
looks.

## Not used

Roughly 400 timestamped action annotations exist from earlier work. They are
temporal — they record *when* something happened — while this is a spatial
question about *which* player. The previous model's four annotation-derived
features carried coefficients between `−0.22` and `+0.15`; as binary features
that was their entire contribution. They were already doing almost nothing.
