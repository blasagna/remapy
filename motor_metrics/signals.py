"""Postural primitives derived from stored pose *world* landmarks.

Everything here is recomputed from raw stored signals and nothing is written back —
the same derive-on-read rule the recording format is built on (see
``recording/recorder.py`` and ``Recording.angles()``).

**The frame.** ``landmarks_world`` is metric (meters) and **hip-centered**: MediaPipe
defines the origin *as* the midpoint of the hips. Two consequences run through this
whole package:

1. A "center of mass" proxy taken as the mid-hip in this frame is **identically zero**
   on every frame. Sway measured on it is floating-point noise. The signal that
   actually carries postural information is :func:`trunk_vector` — the mid-shoulder
   expressed relative to the pelvis, i.e. trunk-over-pelvis excursion. The hip-centered
   origin is what makes that scale-free and calibration-free, so this is a better
   signal than a true COM proxy would be, not a consolation.
2. Translation across the floor is **not recoverable** from this frame. The only
   whole-body translation signal in a recording is :func:`com_norm`, in image
   fractions. See :mod:`motor_metrics.crawl` for what that does and does not buy.

**The vertical.** There is no gravity vector in a camera-only recording. Axes are
camera-relative (x right, y down, z toward camera), so :data:`WORLD_UP` is vertical
**only if the camera is level**. That assumption is not checked automatically — it is
checked by eye, via the notebook's calibration diagnostic over a ``calib;pose=upright``
segment. Every consumer records which vertical it used (``up_source``) so a tilted
session can be found later rather than quietly biasing a year of numbers.
"""

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark as L

from pose_estimation.angles import angle_between

# Up in the camera frame: MediaPipe world coords are y-DOWN, so up is -y.
# Valid only for a level camera — see the module docstring.
WORLD_UP = np.array([0.0, -1.0, 0.0])

_EPS = 1e-9


def mid(points: np.ndarray, i: int, j: int) -> np.ndarray:
    """Midpoint of landmarks ``i`` and ``j`` per frame; ``(N, 3)``."""
    p = np.asarray(points, dtype=np.float64)
    return (p[:, i, :] + p[:, j, :]) / 2.0


def trunk_vector(world: np.ndarray) -> np.ndarray:
    """``(N, 3)`` pelvis -> mid-shoulder, in meters.

    This is the package's core postural signal, used two ways: as a **position** whose
    excursion is sway (:mod:`motor_metrics.hold`), and as a **direction** whose angle
    from vertical is lean (:func:`trunk_from_vertical`).

    The ``- mid_hip`` term is very nearly a no-op, since mid-hip *is* the world frame's
    origin. It is written explicitly anyway: it costs nothing, it makes the signal's
    meaning legible without knowing MediaPipe's origin convention, and it stays correct
    if that convention ever shifts.
    """
    w = np.asarray(world, dtype=np.float64)
    return mid(w, L.LEFT_SHOULDER, L.RIGHT_SHOULDER) - mid(w, L.LEFT_HIP, L.RIGHT_HIP)


def trunk_from_vertical(world: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """``(N,)`` angle in degrees between the trunk and ``up``. 0 = upright.

    Reuses :func:`pose_estimation.angles.angle_between`, so there is one definition of
    "angle" in the codebase, and inherits its **unsigned** semantics: this cannot tell
    a forward lean from a backward or lateral one. Use :func:`project_horizontal` when
    the direction of lean matters. NaN frames propagate to NaN.
    """
    up_unit = _unit(np.asarray(up, dtype=np.float64))
    trunk = trunk_vector(world)
    origin = np.zeros(3)
    # Per-frame loop mirroring Recording.angles(); N is at most a few thousand frames.
    return np.array([angle_between(t, origin, up_unit) for t in trunk])


def project_horizontal(points: np.ndarray, up: np.ndarray = WORLD_UP) -> np.ndarray:
    """Project ``(N, 3)`` points onto the plane perpendicular to ``up``; ``(N, 2)``.

    Returns ``(ML, AP)`` — medio-lateral and antero-posterior — and the axis order is
    the point of the function. On a single camera, ML lies in the image plane and is
    measured well, while AP is MediaPipe's inferred depth and is markedly noisier. A
    caller that collapses these into one isotropic sway number averages a good estimate
    with a bad one and cannot tell afterwards which it was looking at.

    For a level camera this is exactly ``ML = world x``, ``AP = world z``. For a tilted
    ``up`` the basis is the closest thing to that which is still orthonormal.
    """
    p = np.asarray(points, dtype=np.float64)
    up_unit = _unit(np.asarray(up, dtype=np.float64))

    # ML is world-x with any `up` component removed, so it stays the image-plane axis.
    ml = _reject(np.array([1.0, 0.0, 0.0]), up_unit)
    if np.linalg.norm(ml) < 1e-6:
        # Degenerate: `up` is (nearly) world-x, i.e. the camera is rolled ~90 degrees.
        # Fall back to world-z so the basis stays defined rather than blowing up.
        ml = _reject(np.array([0.0, 0.0, 1.0]), up_unit)
    ml = _unit(ml)
    ap = _unit(np.cross(up_unit, ml))
    return np.stack([p @ ml, p @ ap], axis=-1)


def com_norm(norm: np.ndarray) -> np.ndarray:
    """``(N, 3)`` mid-hip in **normalized image coordinates**, not meters.

    The only whole-body translation signal a recording has: ``landmarks_world`` is
    hip-centered, so the pelvis cannot move in it by construction. Columns 0 and 1 are
    image fractions of width and height; column 2 is MediaPipe's relative depth, which
    is on a different and much weaker scale — do not mix it with the first two.

    Any speed derived from this is in image-widths per second. Converting it to m/s
    needs a scale reference; see :func:`motor_metrics.crawl.scale_m_per_norm` for what
    that costs in accuracy.
    """
    return mid(np.asarray(norm, dtype=np.float64), L.LEFT_HIP, L.RIGHT_HIP)


def estimate_up(rec, seg) -> np.ndarray:
    """Unit "up" from a ``calib;pose=upright`` segment: the median trunk direction.

    **Opt-in, and deliberately not the default.** It cannot separate "the camera is
    tilted" from "the child's trunk is not actually vertical" — and for a child with
    global developmental delay the second term is not small, so calibrating on it can
    bake a real postural asymmetry into the reference and hide it forever after.

    Use it when you know the camera was tilted and you trust the calibration pose. Pass
    the result as ``up=`` and the consuming metric will record ``up_source="calib"``.
    """
    trunk = trunk_vector(rec.landmarks_world)[seg.start : seg.stop]
    good = trunk[~np.isnan(trunk).any(axis=1)]
    if good.size == 0:
        raise ValueError("Calibration segment has no frames with a detected pose.")
    return _unit(np.median(good, axis=0))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _reject(v: np.ndarray, unit_axis: np.ndarray) -> np.ndarray:
    """Component of ``v`` perpendicular to the unit vector ``unit_axis``."""
    return v - np.dot(v, unit_axis) * unit_axis
