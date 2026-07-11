"""Shared face-redaction primitives.

Both redaction backends (the ``FaceDetector``-based :class:`face_blur.blur.FaceBlurrer`
and the pose-keypoint :class:`face_blur.pose_blur.PoseFaceBlurrer`) turn a region
of interest into either a solid box or a mosaic. Those style helpers live here so
the two backends share one implementation and one set of style names.
"""

import cv2
import numpy as np

BLUR_STYLES = ("box", "mosaic")

# Solid-fill colour for the "box" style (BGR); black.
_BOX_COLOR = (0, 0, 0)


def validate_style(style: str) -> None:
    """Raise ``ValueError`` unless ``style`` is a supported redaction style."""
    if style not in BLUR_STYLES:
        raise ValueError(f"style must be one of {BLUR_STYLES}, got {style!r}")


def padded_bounds(frame, x, y, w, h, pad, top_pad=None):
    """Pad a box by ``pad`` (fraction of its size) and clamp to the frame.

    ``top_pad`` optionally overrides the padding fraction applied above the box;
    faces derived from keypoints benefit from extra headroom to cover the
    forehead/hair, which the keypoints themselves do not mark. Returns
    ``(x0, y0, x1, y1)`` slice indices, or ``None`` if the region is empty.
    """
    fh, fw = frame.shape[:2]
    dx = int(w * pad)
    dy = int(h * pad)
    dy_top = int(h * top_pad) if top_pad is not None else dy
    x0 = max(0, x - dx)
    y0 = max(0, y - dy_top)
    x1 = min(fw, x + w + dx)
    y1 = min(fh, y + h + dy)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _solid_box(frame, bounds) -> None:
    """Fill the region with a solid colour (irreversible)."""
    x0, y0, x1, y1 = bounds
    frame[y0:y1, x0:x1] = _BOX_COLOR


def _mosaic_region(frame, bounds, blocks) -> None:
    """Pixelate the region into roughly ``blocks``x``blocks`` cells (weak)."""
    x0, y0, x1, y1 = bounds
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    bx = max(1, min(blocks, rw))
    by = max(1, min(blocks, rh))
    small = cv2.resize(roi, (bx, by), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def redact_region(frame: np.ndarray, bounds, style: str, mosaic_blocks: int) -> None:
    """Redact ``bounds`` of ``frame`` in place using ``style``."""
    if style == "box":
        _solid_box(frame, bounds)
    else:
        _mosaic_region(frame, bounds, mosaic_blocks)
