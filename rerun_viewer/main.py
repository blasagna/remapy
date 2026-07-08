"""Example: stream webcam video + pose skeleton + joint-angle metrics to Rerun.

Ties together ``video_capture`` (frames), ``pose_estimation`` (MediaPipe pose),
and the Rerun viewer. Logs the video with a 2D skeleton overlay, a 3D skeleton
from world landmarks, and line plots of frame rate + joint angles.

Run::

    pixi run rerun                     # webcam -> spawns the Rerun viewer
    pixi run rerun --source 1          # a different camera
    pixi run rerun-headless            # no viewer; writes a .rrd recording

Open a saved recording later with:  rerun recording.rrd
Press Ctrl+C in the terminal to quit.
"""

from __future__ import annotations

import argparse
import time

from pose_estimation.estimator import PoseEstimator
from video_capture import CaptureError, VideoCapture

from .viewer import PoseRerunLogger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0, the built-in webcam) or a video file/URL.",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=None, help="Requested frame height.")
    parser.add_argument("--model", default=None, help="Path to a pose_landmarker .task file.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for headless/testing runs).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Write a Rerun .rrd recording to this path instead of spawning the viewer.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not spawn the viewer (e.g. to connect an already-running one).",
    )
    return parser.parse_args(argv)


def _resolve_source(source: str) -> int | str:
    """Treat a bare integer string as a camera index, otherwise a path/URL."""
    return int(source) if source.isdigit() else source


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = _resolve_source(args.source)

    try:
        logger = PoseRerunLogger(spawn=not args.no_spawn, save_path=args.save)
        with VideoCapture(source, width=args.width, height=args.height) as cap, \
                PoseEstimator(model_path=args.model) as pose:
            print(f"Opened source {source!r} at resolution {cap.resolution[0]}x{cap.resolution[1]}")
            if args.save:
                print(f"Recording to {args.save}. Press Ctrl+C to stop.")
            else:
                print("Streaming to the Rerun viewer. Press Ctrl+C to stop.")

            start = time.monotonic()
            prev = start
            count = 0
            for frame in cap.frames():
                count += 1
                now = time.monotonic()
                dt = now - prev
                fps = 1.0 / dt if dt > 0 else 0.0
                prev = now

                timestamp_ms = int((now - start) * 1000)
                result = pose.detect(frame, timestamp_ms)
                logger.log_frame(count, now - start, fps, frame, result)

                if args.max_frames is not None and count >= args.max_frames:
                    break
            print(f"Processed {count} frame(s).")
    except CaptureError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
