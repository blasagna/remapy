"""Probe for video capture devices compatible with :mod:`video_capture`.

Compatibility here means exactly what ``video_capture.capture.VideoCapture``
needs: a camera *index* that ``cv2.VideoCapture`` can open and read a frame
from. We probe indices the same way ``VideoCapture.open()`` does so that a
device reported here is one the other CLIs (``capture``/``pose``/``rerun``/
``record``) can actually use via ``--source <index>``.

On Linux we additionally read the V4L2 sysfs metadata (``/sys/class/video4linux``)
to attach a human-readable name to each index. That is purely informational; the
authoritative check is still whether OpenCV can open and read the index.
"""

import glob
import os
from dataclasses import dataclass
from typing import Optional

import cv2


@dataclass
class DeviceInfo:
    """A capture device that OpenCV successfully opened and read a frame from.

    Attributes
    ----------
    index:
        The camera index. Pass it to any CLI as ``--source <index>``.
    width, height:
        Actual frame size reported by the device.
    fps:
        Frame rate reported by the device (``0.0`` if unknown).
    backend:
        Name of the OpenCV capture backend that opened the device.
    name:
        Human-readable device name from V4L2 sysfs, if available (Linux only).
    node:
        The ``/dev/videoN`` path this index maps to, if it could be determined.
    """

    index: int
    width: int
    height: int
    fps: float
    backend: str
    name: Optional[str] = None
    node: Optional[str] = None

    @property
    def source_arg(self) -> str:
        """The value to pass to another CLI's ``--source`` flag."""
        return str(self.index)


def _v4l2_names() -> dict[int, str]:
    """Map ``/dev/videoN`` index -> device name via Linux V4L2 sysfs.

    Returns an empty mapping on non-Linux platforms or if sysfs is unavailable.
    """
    names: dict[int, str] = {}
    for name_path in glob.glob("/sys/class/video4linux/video*/name"):
        node = os.path.basename(os.path.dirname(name_path))  # e.g. "video0"
        try:
            idx = int(node.removeprefix("video"))
        except ValueError:
            continue
        try:
            with open(name_path, encoding="utf-8", errors="replace") as handle:
                names[idx] = handle.read().strip()
        except OSError:
            continue
    return names


def _highest_v4l2_index() -> Optional[int]:
    """Highest ``/dev/videoN`` index present, or ``None`` if none/non-Linux."""
    indices = []
    for node in glob.glob("/dev/video*"):
        suffix = os.path.basename(node).removeprefix("video")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices) if indices else None


def probe_index(index: int) -> Optional[DeviceInfo]:
    """Open ``index`` with OpenCV and read one frame; return info or ``None``.

    Mirrors ``VideoCapture.open()`` + ``read()``: an index only counts as a real
    device if it both opens and yields a frame, which filters out phantom nodes
    (e.g. metadata-only V4L2 nodes) that open but never deliver an image.
    """
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return DeviceInfo(
            index=index,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            backend=cap.getBackendName(),
        )
    finally:
        cap.release()


def enumerate_devices(max_index: int = 9) -> list[DeviceInfo]:
    """Return every openable, readable capture device.

    Parameters
    ----------
    max_index:
        Highest camera index to probe. Indices ``0..max_index`` are tried. On
        Linux the scan is also extended to cover any higher ``/dev/videoN`` node
        that exists, so external cameras enumerated above ``max_index`` are not
        missed.

    Probing empty indices makes the OpenCV backends log warnings; we silence
    OpenCV's logger for the duration of the scan and restore it afterwards.
    """
    names = _v4l2_names()
    highest = _highest_v4l2_index()
    upper = max(max_index, highest) if highest is not None else max_index

    prev_log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    try:
        devices: list[DeviceInfo] = []
        for index in range(upper + 1):
            info = probe_index(index)
            if info is None:
                continue
            info.name = names.get(index)
            if os.path.exists(f"/dev/video{index}"):
                info.node = f"/dev/video{index}"
            devices.append(info)
        return devices
    finally:
        cv2.utils.logging.setLogLevel(prev_log_level)
