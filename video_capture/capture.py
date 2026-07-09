"""A thin, friendly wrapper around ``cv2.VideoCapture``.

The wrapper is a context manager that yields frames as NumPy arrays and takes
care of opening/releasing the underlying device.
"""

from typing import Iterator, Optional

import cv2
import numpy as np


class CaptureError(RuntimeError):
    """Raised when a capture device cannot be opened or read from."""


class VideoCapture:
    """Wrapper around :class:`cv2.VideoCapture`.

    Parameters
    ----------
    source:
        Camera index (``0`` is the built-in webcam by default) or a path/URL to
        a video file or stream.
    width, height:
        Optional requested frame size. The device may ignore these and pick the
        nearest supported resolution.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> VideoCapture:
        """Open the underlying device. Returns ``self`` for chaining."""
        cap = cv2.VideoCapture(self.source)
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            cap.release()
            raise CaptureError(f"Could not open video source: {self.source!r}")
        self._cap = cap
        return self

    def read(self) -> np.ndarray:
        """Read and return a single BGR frame.

        Raises
        ------
        CaptureError
            If the device is not open or no frame could be grabbed.
        """
        if self._cap is None:
            raise CaptureError("Capture is not open; call open() first.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CaptureError("Failed to read frame from video source.")
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the source is exhausted or an error occurs."""
        if self._cap is None:
            raise CaptureError("Capture is not open; call open() first.")
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield frame

    @property
    def resolution(self) -> tuple[int, int]:
        """The actual ``(width, height)`` reported by the device."""
        if self._cap is None:
            raise CaptureError("Capture is not open; call open() first.")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        """Release the underlying device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoCapture:
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.release()
