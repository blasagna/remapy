"""Lightweight stand-ins for the external objects the libraries consume.

These fakes avoid every heavyweight/external dependency in the test suite:

- :class:`FakeLandmark` / pose-result builders replace MediaPipe's landmark and
  ``PoseLandmarkerResult`` objects (duck-typed: only attribute access is used).
- :func:`fake_recording` replaces ``recording.reader.Recording``.
- :class:`FakeCapture` replaces an opened ``cv2.VideoCapture`` handle.
- :class:`FakeVideoWriter` replaces ``cv2.VideoWriter``.

Nothing here touches a camera, the network, the display, or a native model.
"""

from types import SimpleNamespace

import numpy as np


# --------------------------------------------------------------------------- #
# MediaPipe-shaped pose results
# --------------------------------------------------------------------------- #
def FakeLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0):
    """A single landmark with the attributes the libraries read."""
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=presence)


def make_landmarks(n=33, *, visibility=1.0):
    """A list of ``n`` landmarks spread across the unit square (metric-ish)."""
    lms = []
    for i in range(n):
        frac = i / max(1, n - 1)
        lms.append(FakeLandmark(x=frac, y=frac, z=frac * 0.1, visibility=visibility))
    return lms


def face_landmarks_in_box(x0, y0, x1, y1, w, h, n=33, *, visibility=1.0):
    """Landmarks whose first 11 (the face) fall inside pixel box ``(x0,y0,x1,y1)``.

    Coordinates are normalized (0..1) as MediaPipe emits them; ``w``/``h`` are the
    frame size used to place the face keypoints inside the requested pixel box.
    Remaining (body) landmarks are parked at the frame center.
    """
    face = []
    for i in range(11):
        fx = (x0 + (x1 - x0) * (i / 10)) / w
        fy = (y0 + (y1 - y0) * (i / 10)) / h
        face.append(FakeLandmark(x=fx, y=fy, visibility=visibility))
    body = [FakeLandmark(x=0.5, y=0.5, visibility=visibility) for _ in range(n - 11)]
    return face + body


def pose_result(poses_norm=None, poses_world=None):
    """A ``PoseLandmarkerResult``-shaped object.

    ``poses_norm`` / ``poses_world`` are lists of landmark-lists (one per detected
    pose). ``None`` / empty means "no pose detected".
    """
    return SimpleNamespace(
        pose_landmarks=poses_norm or [],
        pose_world_landmarks=poses_world or [],
    )


#: A result with no pose, as MediaPipe emits when it finds nothing.
NO_POSE = pose_result()


def pose_result_from_row(world_row, norm_row=None, *, visibility=1.0):
    """A pose result carrying one frame of a ``body_world``-style ``(33, 3)`` array.

    The bridge between the array-shaped fakes (which describe *anatomy*) and the
    result-shaped ones (which describe what the *model emits*). Live code consumes the
    latter — ``motor_metrics.live.LiveWindow.push`` takes a result, not rows — so a
    test that wants a synthetic body moving through a live buffer needs this to convert
    between them. ``norm_row`` defaults to ``world_row``, which is fine for anything
    reading only world coordinates.
    """
    world_row = np.asarray(world_row, dtype=np.float64)
    norm_row = world_row if norm_row is None else np.asarray(norm_row, dtype=np.float64)
    world = [
        FakeLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=visibility)
        for p in world_row
    ]
    norm = [
        FakeLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=visibility)
        for p in norm_row
    ]
    return pose_result([norm], [world])


# --------------------------------------------------------------------------- #
# Recording stand-in
# --------------------------------------------------------------------------- #
NUM_LANDMARKS = 33

# MediaPipe PoseLandmark indices, spelled out so the fakes need no mediapipe import.
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26


def body_world(
    trunk,
    *,
    hip_center=None,
    shoulder_width=0.25,
    hip_width=0.18,
    left_wrist=None,
    right_wrist=None,
    left_knee=None,
    right_knee=None,
):
    """``(N, 33, 3)`` world landmarks for a synthetic body.

    ``make_landmarks`` spreads points along a diagonal, which is fine for pass-through
    tests but is not a body — postural metrics need real anatomy. Here ``trunk`` is an
    ``(N, 3)`` pelvis -> mid-shoulder vector per frame; hips are placed symmetrically
    about ``hip_center`` (default the origin, mirroring MediaPipe's hip-centered world
    frame) and shoulders symmetrically about ``hip_center + trunk``, separated along
    world x. ``left_wrist``/``right_wrist``/``left_knee``/``right_knee`` are absolute
    ``(N, 3)`` positions. Unnamed landmarks stay at the origin.
    """
    trunk = np.atleast_2d(np.asarray(trunk, dtype=np.float32))
    count = trunk.shape[0]
    hips = np.zeros((count, 3), np.float32) if hip_center is None else np.broadcast_to(
        np.asarray(hip_center, dtype=np.float32), (count, 3)
    )
    shoulders = hips + trunk

    out = np.zeros((count, NUM_LANDMARKS, 3), dtype=np.float32)
    half_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    out[:, _L_HIP] = hips + half_x * (hip_width / 2)
    out[:, _R_HIP] = hips - half_x * (hip_width / 2)
    out[:, _L_SHOULDER] = shoulders + half_x * (shoulder_width / 2)
    out[:, _R_SHOULDER] = shoulders - half_x * (shoulder_width / 2)
    if left_wrist is not None:
        out[:, _L_WRIST] = np.asarray(left_wrist, dtype=np.float32)
    if right_wrist is not None:
        out[:, _R_WRIST] = np.asarray(right_wrist, dtype=np.float32)
    if left_knee is not None:
        out[:, _L_KNEE] = np.asarray(left_knee, dtype=np.float32)
    if right_knee is not None:
        out[:, _R_KNEE] = np.asarray(right_knee, dtype=np.float32)
    return out


def fake_recording(
    world=None,
    *,
    norm=None,
    visibility=1.0,
    presence=1.0,
    timestamps_ms=None,
    pose_present=None,
    annotations=None,
    n=None,
    fps=30.0,
):
    """Duck-typed stand-in for :class:`recording.reader.Recording`.

    Carries only what the :mod:`motor_metrics` functions read, so their tests need no
    HDF5 file: ``landmarks_world``, ``landmarks_norm``, ``visibility``, ``presence``,
    ``pose_present``, ``timestamps_ms``, ``annotations``.

    ``world``/``norm`` are ``(N, 33, 3)`` arrays (default zeros). ``visibility`` and
    ``presence`` take either a scalar (broadcast across all landmarks) or a full
    ``(N, 33)`` array. ``timestamps_ms`` defaults to a uniform ``fps`` grid.
    ``pose_present`` defaults to mirroring the real property exactly — the whole-row
    NaN check on landmark 0 — so a test can build no-pose frames by writing NaN rows
    and get the real semantics for free.
    """
    if world is None:
        world = np.zeros((n if n is not None else 10, NUM_LANDMARKS, 3), dtype=np.float32)
    world = np.asarray(world, dtype=np.float32)
    count = world.shape[0]
    if norm is None:
        norm = np.zeros_like(world)

    def _per_landmark(value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((count, NUM_LANDMARKS), float(arr), dtype=np.float32)
        return arr

    if timestamps_ms is None:
        timestamps_ms = (np.arange(count) * (1000.0 / fps)).astype(np.int64)
    if pose_present is None:
        pose_present = ~np.isnan(world[:, 0, 0])

    return SimpleNamespace(
        landmarks_world=world,
        landmarks_norm=np.asarray(norm, dtype=np.float32),
        visibility=_per_landmark(visibility),
        presence=_per_landmark(presence),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.int64),
        pose_present=np.asarray(pose_present, dtype=bool),
        annotations=list(annotations or []),  # Recording defaults this to [] too
    )


# --------------------------------------------------------------------------- #
# MediaPipe detection result (FaceDetector)
# --------------------------------------------------------------------------- #
def detection(origin_x, origin_y, width, height):
    """A single ``FaceDetector`` detection with a bounding box."""
    return SimpleNamespace(
        bounding_box=SimpleNamespace(
            origin_x=origin_x, origin_y=origin_y, width=width, height=height
        )
    )


def detector_result(*dets):
    """A ``FaceDetectorResult``-shaped object holding ``dets``."""
    return SimpleNamespace(detections=list(dets))


# --------------------------------------------------------------------------- #
# OpenCV stand-ins
# --------------------------------------------------------------------------- #
class FakeCapture:
    """Stand-in for an opened ``cv2.VideoCapture`` handle.

    ``frames`` is a list of frames (or ``None`` entries) returned by successive
    ``read()`` calls; once exhausted, ``read()`` returns ``(False, None)``.
    ``props`` maps ``cv2.CAP_PROP_*`` ints to reported values.
    """

    def __init__(self, frames=None, opened=True, props=None, backend="FAKE"):
        self._frames = list(frames) if frames is not None else []
        self._opened = opened
        self.props = dict(props or {})
        self._backend = backend
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        frame = self._frames.pop(0)
        return (frame is not None), frame

    def get(self, prop):
        return float(self.props.get(prop, 0.0))

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        self.props[prop] = value
        return True

    def getBackendName(self):
        return self._backend

    def release(self):
        self.released = True


class FakeVideoWriter:
    """Stand-in for ``cv2.VideoWriter`` — records written frames."""

    def __init__(self, path, fourcc, fps, size, opened=True):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self._opened = opened
        self.frames = []
        self.released = False

    def isOpened(self):
        return self._opened

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def release(self):
        self.released = True


def solid_frame(h=48, w=64, value=200):
    """A uniform BGR frame; a distinctive value makes redaction easy to detect."""
    return np.full((h, w, 3), value, dtype=np.uint8)


class FakeSerial:
    """Stand-in for a pyserial ``Serial`` handle feeding pre-baked bytes.

    Duck-types the little that :class:`adafruit_feather_sense.stream.FeatherSenseStream`
    uses: ``in_waiting``, ``read(n)``, ``close()``, ``port``. ``chunk`` caps how
    many bytes each ``read`` yields so tests can exercise cross-read buffering.
    """

    def __init__(self, blob=b"", chunk=8, port="FAKE"):
        self._blob = bytes(blob)
        self._chunk = chunk
        self.port = port
        self.closed = False

    @property
    def in_waiting(self):
        return min(self._chunk, len(self._blob))

    def read(self, n=1):
        n = n or 1
        out, self._blob = self._blob[:n], self._blob[n:]
        return out

    def close(self):
        self.closed = True


class FakePixel:
    """Stand-in for a `neopixel.NeoPixel`, recording what got written.

    Lets the status LED's write logic be tested off-board (`StatusLED(pixel=...)`),
    where `board`/`neopixel` don't exist.
    """

    def __init__(self):
        self.fills = []
        self.shows = 0

    def fill(self, color):
        self.fills.append(color)

    def show(self):
        self.shows += 1
