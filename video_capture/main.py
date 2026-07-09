"""Example script: display the webcam feed in a window.

Run with the built-in webcam::

    pixi run capture

Point at a different device or a video file::

    pixi run capture --source 1
    pixi run capture --source path/to/video.mp4

Press ``q`` or ``Esc`` in the window to quit.
"""

import argparse

import cv2

from .capture import CaptureError, VideoCapture

WINDOW_NAME = "remapy video_capture"
QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0, the built-in webcam) or a video file/URL.",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=None, help="Requested frame height.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for headless/testing runs).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a display window; just read frames (headless).",
    )
    return parser.parse_args(argv)


def _resolve_source(source: str) -> int | str:
    """Treat a bare integer string as a camera index, otherwise a path/URL."""
    return int(source) if source.isdigit() else source


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = _resolve_source(args.source)

    try:
        with VideoCapture(source, width=args.width, height=args.height) as cap:
            print(f"Opened source {source!r} at resolution {cap.resolution[0]}x{cap.resolution[1]}")
            if args.no_window:
                print("To exit cleanly: press Ctrl+C.")
            else:
                print("To exit cleanly: press 'q' or Esc in the window (or Ctrl+C).")
            count = 0
            for frame in cap.frames():
                count += 1
                if not args.no_window:
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                        break
                if args.max_frames is not None and count >= args.max_frames:
                    break
            print(f"Captured {count} frame(s).")
    except CaptureError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
