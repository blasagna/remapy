"""Draw a pose skeleton onto a BGR frame from plain numpy landmarks.

Deliberately free of any ``mediapipe`` import, so tools that read *recorded* poses can
use it: a recording's ``meta/pose_connections`` already carries the 35-pair edge list
(see ``recording/recorder.py``), and importing :mod:`pose_estimation.estimator` just to
reach ``POSE_CONNECTIONS`` would pull the whole MediaPipe stack into ``annotate``.

Two things the live CLI never needed but a recording scrubber does:

- **NaN safety.** ``HDF5Recorder`` writes a full 33x3 NaN row for frames with no pose,
  and ``annotate`` scrubs freely across them. ``int(nan)`` raises, so points and bones
  touching NaN are skipped rather than drawn.
- **Visibility dimming.** MediaPipe *extrapolates* occluded landmarks instead of
  dropping them, so an invented coordinate is indistinguishable from a measured one by
  eye. Landmarks below ``min_visibility`` draw dimmed and thin -- the same 0.5 threshold
  ``motor_metrics.quality.Gate`` gates its metrics on, surfaced visually.
"""

import cv2
import numpy as np

BONE_COLOR = (0, 255, 0)
JOINT_COLOR = (0, 0, 255)
# Dimmed variants, for landmarks the model is extrapolating rather than seeing.
BONE_COLOR_DIM = (0, 90, 0)
JOINT_COLOR_DIM = (0, 0, 90)

# Matches motor_metrics.quality.Gate's default, so what looks solid here is what the
# metrics will actually accept.
MIN_VISIBILITY = 0.5

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_text(frame, text, org, scale, color, thickness: int = 1) -> None:
    """Draw text with a black outline so it stays readable over arbitrary footage.

    Both callers draw over whatever the camera happened to see — a recording being
    scrubbed in ``annotate``, or a live frame under the metrics overlay — so plain
    ``cv2.putText`` disappears against a bright wall or a light shirt. Lives here with
    :func:`draw_skeleton` because this module is already the shared, mediapipe-free
    OpenCV drawing surface.
    """
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(
    frame,
    landmarks_norm,
    connections,
    visibility=None,
    min_visibility: float = MIN_VISIBILITY,
) -> None:
    """Draw ``landmarks_norm`` and its ``connections`` onto ``frame`` in place.

    ``landmarks_norm`` is ``(K, 2+)`` with x/y as image fractions in ``[0, 1]`` (the
    ``landmarks_norm`` layout ``recording.reader.Recording`` exposes); only the first two
    columns are read. ``connections`` is any ``(E, 2)`` of landmark index pairs.
    ``visibility`` is an optional ``(K,)`` of per-landmark confidences; when omitted
    every landmark is drawn at full strength.

    Never raises on NaN or out-of-range indices -- a frame with no pose is a no-op.
    """
    pts = np.asarray(landmarks_norm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return

    h, w = frame.shape[:2]
    xy = pts[:, :2]
    good = ~np.isnan(xy).any(axis=1)
    if not good.any():
        return

    n = pts.shape[0]
    px = np.zeros((n, 2), dtype=np.int64)
    px[good] = np.column_stack((xy[good, 0] * w, xy[good, 1] * h)).astype(np.int64)

    if visibility is None:
        vis = np.ones(n)
    else:
        vis = np.asarray(visibility, dtype=np.float64).ravel()
        vis = np.nan_to_num(vis, nan=0.0)
        if vis.shape[0] < n:  # tolerate a short/absent confidence array
            vis = np.pad(vis, (0, n - vis.shape[0]), constant_values=1.0)

    # Bones first, so joints sit on top of the lines.
    for pair in np.asarray(connections).reshape(-1, 2):
        s, e = int(pair[0]), int(pair[1])
        if not (0 <= s < n and 0 <= e < n) or not (good[s] and good[e]):
            continue
        strong = min(vis[s], vis[e]) >= min_visibility
        cv2.line(
            frame,
            (int(px[s, 0]), int(px[s, 1])),
            (int(px[e, 0]), int(px[e, 1])),
            BONE_COLOR if strong else BONE_COLOR_DIM,
            2 if strong else 1,
        )

    for i in range(n):
        if not good[i]:
            continue
        strong = vis[i] >= min_visibility
        cv2.circle(
            frame,
            (int(px[i, 0]), int(px[i, 1])),
            3 if strong else 2,
            JOINT_COLOR if strong else JOINT_COLOR_DIM,
            -1,
        )
