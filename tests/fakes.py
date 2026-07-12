"""Lightweight stand-ins for the external objects the libraries consume.

These fakes avoid every heavyweight/external dependency in the test suite:

- :class:`FakeLandmark` / pose-result builders replace MediaPipe's landmark and
  ``PoseLandmarkerResult`` objects (duck-typed: only attribute access is used).
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
