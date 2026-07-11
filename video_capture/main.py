"""Example script: display the webcam feed in a window.

Run with the built-in webcam::

    pixi run capture

Point at a different device or a video file::

    pixi run capture --source 1
    pixi run capture --source path/to/video.mp4

Press ``q`` or ``Esc`` in the window to quit.
"""

import argparse
import time

import cv2

from face_blur.factory import BLUR_METHODS, build_blurrer

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
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720).")
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
    parser.add_argument(
        "--blur-faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redact detected faces for privacy. On by default.",
    )
    parser.add_argument(
        "--blur-style",
        choices=["box", "mosaic"],
        default="box",
        help="box = solid fill (irreversible, default); mosaic = blocky pixelation.",
    )
    parser.add_argument(
        "--blur-method",
        choices=BLUR_METHODS,
        default="hybrid",
        help="hybrid (default) = pose keypoints when a pose is present, else the "
        "detector; pose = pose keypoints only (more reliable when a body is "
        "tracked); detector = standalone FaceDetector only. The pose/hybrid "
        "methods run pose estimation on each displayed frame.",
    )
    parser.add_argument("--face-model", default=None, help="Path to a face detector .tflite.")
    return parser.parse_args(argv)


def _resolve_source(source: str) -> int | str:
    """Treat a bare integer string as a camera index, otherwise a path/URL."""
    return int(source) if source.isdigit() else source


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = _resolve_source(args.source)

    blurrer = (
        build_blurrer(args.blur_method, style=args.blur_style, model_path=args.face_model)
        if args.blur_faces
        else None
    )
    # The pose/hybrid backends need a pose result per frame; this demo has no pose
    # loop, so spin up an estimator only when one of those backends is selected.
    pose = None
    if args.blur_faces and args.blur_method in ("pose", "hybrid") and not args.no_window:
        from pose_estimation.estimator import PoseEstimator

        pose = PoseEstimator()
    try:
        with VideoCapture(source, width=args.width, height=args.height) as cap:
            print(f"Opened source {source!r} at resolution {cap.resolution[0]}x{cap.resolution[1]}")
            if args.no_window:
                print("To exit cleanly: press Ctrl+C.")
            else:
                print("To exit cleanly: press 'q' or Esc in the window (or Ctrl+C).")
            start = time.monotonic()
            count = 0
            for frame in cap.frames():
                count += 1
                if not args.no_window:
                    if blurrer is not None:
                        result = (
                            pose.detect(frame, int((time.monotonic() - start) * 1000))
                            if pose is not None
                            else None
                        )
                        blurrer.blur(frame, result)
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
        if blurrer is not None:
            blurrer.close()
        if pose is not None:
            pose.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
