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

import argparse
import time

from face_blur.factory import BLUR_METHODS, build_blurrer
from pose_estimation.estimator import PoseEstimator
from recording.recorder import HDF5Recorder
from video_capture.capture import CaptureError, VideoCapture

from .viewer import PoseRerunLogger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (default: 0, the built-in webcam) or a video file/URL.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720).")
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
    parser.add_argument(
        "--memory-limit",
        default="75%",
        help="Viewer in-memory store cap; oldest data is dropped past it "
        "(e.g. '2GB', '50%%'). Default: 75%% of system RAM.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality (1-100) for the logged video frames. Default: 75.",
    )
    parser.add_argument(
        "--blur-faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redact detected faces for privacy before logging. On by default.",
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
    parser.add_argument(
        "--record", default=None, help="Also write an HDF5 recording to this path (offline analysis)."
    )
    parser.add_argument(
        "--record-video", default=None, help="With --record, also write a parallel mp4 to this path."
    )
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
    recorder = None
    try:
        logger = PoseRerunLogger(
            spawn=not args.no_spawn,
            save_path=args.save,
            memory_limit=args.memory_limit,
            jpeg_quality=args.jpeg_quality,
        )
        with VideoCapture(source, width=args.width, height=args.height) as cap, \
                PoseEstimator(model_path=args.model) as pose:
            print(f"Opened source {source!r} at resolution {cap.resolution[0]}x{cap.resolution[1]}")
            if args.save:
                print(f"Recording to {args.save}. Press Ctrl+C to stop.")
            else:
                print("Streaming to the Rerun viewer. Press Ctrl+C to stop.")
            if args.record:
                print(f"Writing HDF5 recording to {args.record}.")
                recorder = HDF5Recorder(
                    args.record,
                    jpeg_quality=args.jpeg_quality,
                    video_path=args.record_video,
                    model_name=args.model or "pose_landmarker_lite",
                    blur_style=args.blur_style if args.blur_faces else None,
                    faces_blurred=args.blur_faces,
                )

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

                # Redact faces before logging so recordings never hold raw faces.
                if blurrer is not None:
                    blurrer.blur(frame, result)
                logger.log_frame(count, now - start, fps, frame, result)
                if recorder is not None:
                    recorder.append(frame, timestamp_ms, result)

                if args.max_frames is not None and count >= args.max_frames:
                    break
            print(f"Processed {count} frame(s).")
    except CaptureError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        if recorder is not None:
            recorder.close()
        if blurrer is not None:
            blurrer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
