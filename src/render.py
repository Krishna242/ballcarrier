"""Overlay rendering.

Carrier errors are obvious to a football-literate viewer within seconds and
nearly invisible in an accuracy table, so the overlay is a first-class output
rather than a debugging convenience.
"""

from __future__ import annotations

import cv2
import numpy as np

C_OTHER = (150, 140, 130)
C_CARRIER = (26, 118, 184)
C_TEXT = (245, 246, 244)
C_PANEL = (34, 40, 36)


def render(frames, per_frame, path, scores, out_path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for t, frame in enumerate(frames):
        canvas = frame.copy()
        carrier = path[t] if t < len(path) else None

        for tid, (x1, y1, x2, y2) in per_frame[t].items():
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            if tid == carrier:
                cv2.rectangle(canvas, p1, p2, C_CARRIER, 3)
                label = f"CARRIER #{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv2.rectangle(canvas, (p1[0], p1[1] - th - 8),
                              (p1[0] + tw + 8, p1[1]), C_CARRIER, -1)
                cv2.putText(canvas, label, (p1[0] + 4, p1[1] - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, p1, p2, C_OTHER, 1)

        panel = canvas[8:96, 8:250]
        panel[:] = (panel * 0.25 + np.array(C_PANEL) * 0.75).astype(np.uint8)
        cv2.putText(canvas, "carrier score (trajectory only)", (16, 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, C_TEXT, 1, cv2.LINE_AA)

        top = sorted(scores[t].items(), key=lambda kv: -kv[1])[:3] if t < len(scores) else []
        if not top:
            cv2.putText(canvas, "pre-snap - carrier undefined", (16, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_OTHER, 1, cv2.LINE_AA)
        for i, (tid, s) in enumerate(top):
            col = C_CARRIER if tid == carrier else C_OTHER
            cv2.putText(canvas, f"#{tid:<4} {s:+.2f}", (16, 48 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        writer.write(canvas)
    writer.release()
