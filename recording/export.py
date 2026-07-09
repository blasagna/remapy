"""Reconstruct a video file from the JPEG frames stored in a recording.

H.264 is not available in this OpenCV build, so the export uses the ``mp4v``
codec. The output fps defaults to the median inter-frame rate from the recorded
timestamps.

CLI::

    python -m recording.export session.h5 out.mp4 [--fps N]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .reader import Recording


def export_mp4(h5_path: Path | str, mp4_path: Path | str, fps: float | None = None) -> Path:
    """Write an mp4 (mp4v) from the stored frames. Returns the output path."""
    with Recording(h5_path) as rec:
        n = len(rec)
        if n == 0:
            raise ValueError("Recording has no frames to export.")
        if fps is None:
            dt = np.diff(rec.timestamps_ms).astype(np.float64)
            dt = dt[dt > 0]
            fps = float(round(1000.0 / np.median(dt))) if dt.size else 30.0
        width = int(rec.metadata["image_width"])
        height = int(rec.metadata["image_height"])

        writer = cv2.VideoWriter(
            str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {mp4_path!r}")
        try:
            for i in range(n):
                writer.write(rec.frame(i))
        finally:
            writer.release()
    return Path(mp4_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5", help="Input recording .h5 file.")
    parser.add_argument("mp4", help="Output .mp4 path.")
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output frame rate (default: median rate from recorded timestamps).",
    )
    args = parser.parse_args(argv)
    out = export_mp4(args.h5, args.mp4, args.fps)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
