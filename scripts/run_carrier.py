"""End-to-end carrier inference on one clip.

    python scripts/run_carrier.py --video data/raw/clip.mp4 --start 19 --duration 6

Add --harvest to also run the on-screen-indicator label harvester and report
whether it agrees with the trajectory-only prediction. That agreement rate is
the project's first real accuracy number, so it is worth printing even while
the harvester is unvalidated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import carrier, hud, render, segment, tracking  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--out", default="data/interim/run")
    ap.add_argument("--harvest", action="store_true",
                    help="also run the HUD indicator label harvester")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] shot boundaries")
    cuts, fps0 = segment.detect_cuts(args.video, upto_s=args.start + args.duration + 2)
    start, duration = segment.clamp_to_shot(args.start, args.duration, cuts, fps0)
    if duration < args.duration:
        print(f"      cut at {start + duration:.1f}s — trimmed to {duration:.1f}s")
    else:
        print("      no cut inside the requested window")

    print(f"[2/5] tracking  {start:.1f}s +{duration:.1f}s")
    frames, per_frame, fps = tracking.collect_tracks(
        args.video, start, duration, args.weights)
    print(f"      {len(frames)} frames @ {fps:.1f}fps")

    print("[3/5] trajectories")
    pos, vel = tracking.build_trajectories(per_frame)
    print(f"      {len(pos)} tracks")

    print("[4/5] snap + scoring + structured decode")
    snap, _ = segment.detect_snap(vel, len(frames))
    if snap is None:
        print("      no snap found — clip may be dead time; carrier undefined")
    else:
        print(f"      snap at frame {snap} ({start + snap / fps:.1f}s in video)")

    scores = carrier.score_frames(pos, vel, len(frames))
    if snap is not None:
        scores = [{} if t < snap else s for t, s in enumerate(scores)]
    path = carrier.viterbi(scores)
    resolved = [p for p in path if p is not None]
    switches = sum(1 for a, b in zip(resolved, resolved[1:]) if a != b)
    margin = carrier.decision_margin(scores)
    print(f"      resolved {len(resolved)}/{len(frames)} frames, "
          f"{switches} possession change(s), margin {margin:.2f}")

    summary = {
        "video": str(args.video),
        "clip": {"start_s": start, "duration_s": duration, "fps": fps},
        "cuts_s": [round(c / fps0, 2) for c in cuts],
        "frames": len(frames),
        "tracks": len(pos),
        "snap_frame": snap,
        "frames_resolved": len(resolved),
        "possession_changes": switches,
        "median_margin": round(margin, 3),
        "ball_detection_used": False,
    }

    if args.harvest:
        print("[+]   harvesting HUD indicator labels")
        reticle, _ = hud.harvest(frames)
        check = hud.validate(reticle)
        print(f"      {check}")
        summary["harvest"] = check
        if check["plausible"]:
            labels = hud.assign_to_track(reticle, per_frame)
            both = [(p, l) for p, l in zip(path, labels)
                    if p is not None and l is not None]
            if both:
                agree = sum(1 for p, l in both if p == l) / len(both)
                print(f"      trajectory prediction agrees with indicator on "
                      f"{agree:.1%} of {len(both)} labelled frames")
                summary["agreement"] = round(agree, 3)
                summary["labelled_frames"] = len(both)
        else:
            print("      harvester rejected its own output — not usable yet")

    print("[5/5] rendering overlay")
    render.render(frames, per_frame, path, scores, out_dir / "overlay.mp4", fps)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'overlay.mp4'}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
