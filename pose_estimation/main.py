"""Example: live body-pose skeleton + joint angles from the webcam.

Uses the ``video_capture`` library for frames and MediaPipe's ``PoseLandmarker``
(lite model) for detection. The lite model is downloaded automatically on first
run and cached under ``pose_estimation/models/``.

Run::

    pixi run pose                 # webcam, windowed
    pixi run pose --source 1      # a different camera
    pixi run pose-headless        # no window; prints joint angles

Press ``q`` or ``Esc`` in the window to quit (or Ctrl+C in the terminal).
"""

import argparse
import time

import cv2

from face_blur.factory import BLUR_METHODS, build_blurrer
from video_capture.capture import CaptureError, VideoCapture

from .angles import joint_angles
from .estimator import POSE_CONNECTIONS, PoseEstimator

WINDOW_NAME = "remapy pose_estimation"
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
    parser.add_argument("--model", default=None, help="Path to a pose_landmarker .task file.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for headless/testing runs).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a display window; print joint angles instead (headless).",
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
        "tracked); detector = standalone FaceDetector only.",
    )
    parser.add_argument("--face-model", default=None, help="Path to a face detector .tflite.")
    return parser.parse_args(argv)


def _resolve_source(source: str) -> int | str:
    """Treat a bare integer string as a camera index, otherwise a path/URL."""
    return int(source) if source.isdigit() else source


def draw_pose(frame, pose_landmarks) -> None:
    """Draw one pose's skeleton (normalized landmarks) onto a BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
    for start, end in POSE_CONNECTIONS:
        cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


def draw_angles(frame, angles: dict[str, float]) -> None:
    """Overlay joint-angle readouts in the top-left corner."""
    for i, (name, deg) in enumerate(angles.items()):
        text = f"{name}: {deg:5.1f}" if deg == deg else f"{name}: n/a"  # deg==deg filters NaN
        cv2.putText(
            frame, text, (10, 20 + 18 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = _resolve_source(args.source)

    blurrer = (
        build_blurrer(args.blur_method, style=args.blur_style, model_path=args.face_model)
        if args.blur_faces
        else None
    )
    try:
        with VideoCapture(source, width=args.width, height=args.height) as cap, \
                PoseEstimator(model_path=args.model) as pose:
            print(f"Opened source {source!r} at resolution {cap.resolution[0]}x{cap.resolution[1]}")
            if args.no_window:
                print("To exit cleanly: press Ctrl+C.")
            else:
                print("To exit cleanly: press 'q' or Esc in the window (or Ctrl+C).")

            start = time.monotonic()
            count = 0
            for frame in cap.frames():
                count += 1
                timestamp_ms = int((time.monotonic() - start) * 1000)
                result = pose.detect(frame, timestamp_ms)

                # Redact faces before drawing, so the skeleton stays visible on top.
                if blurrer is not None and not args.no_window:
                    blurrer.blur(frame, result)

                have_pose = bool(result.pose_landmarks)
                if have_pose:
                    angles = joint_angles(result.pose_world_landmarks[0])
                    if args.no_window:
                        readout = "  ".join(
                            f"{k}={v:.0f}" for k, v in angles.items() if v == v
                        )
                        print(f"frame {count}: {readout}")
                    else:
                        draw_pose(frame, result.pose_landmarks[0])
                        draw_angles(frame, angles)
                elif args.no_window:
                    print(f"frame {count}: no pose detected")

                if not args.no_window:
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                        break
                if args.max_frames is not None and count >= args.max_frames:
                    break
            print(f"Processed {count} frame(s).")
    except CaptureError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        cv2.destroyAllWindows()
        if blurrer is not None:
            blurrer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
