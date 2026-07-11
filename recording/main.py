"""Record a webcam session to an HDF5 file for offline analysis.

Runs the capture -> pose -> face-blur pipeline and writes the face-blurred video
(as JPEG frames) plus the raw pose landmarks to a self-contained ``.h5``. Derived
signals (angles, fps, pose-present) are recomputed later via
``recording.reader.Recording``.

Run::

    pixi run record                          # -> recording.hdf5
    python -m recording.main --output s.h5 --video s.mp4   # also write a parallel mp4

Press Ctrl+C to stop.
"""

import argparse
import time

from face_blur.factory import BLUR_METHODS, build_blurrer
from pose_estimation.estimator import PoseEstimator
from video_capture.capture import CaptureError, VideoCapture

from .recorder import HDF5Recorder


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
        "--output", "-o", default="recording.hdf5", help="Output HDF5 path (default: recording.hdf5)."
    )
    parser.add_argument(
        "--video", default=None, help="Also write a parallel mp4 (mp4v) to this path."
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=75,
        help="JPEG quality (1-100) for the stored video frames. Default: 75.",
    )
    parser.add_argument(
        "--blur-faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redact detected faces for privacy before recording. On by default.",
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
            print(f"Recording to {args.output}. Press Ctrl+C to stop.")
            recorder = HDF5Recorder(
                args.output,
                jpeg_quality=args.jpeg_quality,
                video_path=args.video,
                model_name=args.model or "pose_landmarker_lite",
                blur_style=args.blur_style if args.blur_faces else None,
                faces_blurred=args.blur_faces,
            )
            try:
                start = time.monotonic()
                count = 0
                for frame in cap.frames():
                    count += 1
                    timestamp_ms = int((time.monotonic() - start) * 1000)
                    result = pose.detect(frame, timestamp_ms)

                    # Redact faces before recording so stored video is never raw.
                    if blurrer is not None:
                        blurrer.blur(frame, result)
                    recorder.append(frame, timestamp_ms, result)

                    if args.max_frames is not None and count >= args.max_frames:
                        break
                print(f"Recorded {count} frame(s) to {args.output}.")
            finally:
                recorder.close()
    except CaptureError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        if blurrer is not None:
            blurrer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
