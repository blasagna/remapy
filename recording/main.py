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

from adafruit_feather_sense import open_feather
from face_blur.factory import BLUR_METHODS, build_blurrer
from motor_metrics import derive
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
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720).")
    parser.add_argument(
        "--fps",
        type=float,
        default=derive.FS,
        help=(
            "Requested capture rate (default: %(default)s, the motor_metrics grid). "
            "Advisory — many webcams ignore it; a warning is printed when the device "
            "reports a different rate, because capturing above the grid means the "
            "metrics chain decimates without anti-aliasing."
        ),
    )
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
    parser.add_argument(
        "--feather",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Also record Feather Sense sensor data (USB serial or BLE, see "
        "--feather-transport) into the same .h5 (under /feather). Default: auto "
        "(use it if detected); --feather requires it (error if absent); "
        "--no-feather disables the probe.",
    )
    parser.add_argument(
        "--feather-transport",
        choices=["serial", "ble"],
        default="serial",
        help="How to reach the Feather Sense: serial (USB, default) or ble.",
    )
    parser.add_argument(
        "--feather-port",
        default=None,
        help="Serial port of the Feather Sense (serial transport; default: auto-detect).",
    )
    parser.add_argument(
        "--feather-address",
        default=None,
        help="BLE address of the Feather Sense (ble transport; default: scan for the name).",
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
    feather = None
    if args.feather is not False:  # None (auto) or True (required)
        feather = open_feather(
            args.feather_transport, port=args.feather_port, address=args.feather_address
        )
        if feather is None and args.feather is True:
            print("Error: Feather Sense requested (--feather) but not detected.")
            return 1
        print(
            f"Feather Sense ({args.feather_transport}) detected on {feather.port}; recording sensor data."
            if feather is not None
            else "Feather Sense not detected; continuing without sensor data."
        )
    try:
        with VideoCapture(source, width=args.width, height=args.height, fps=args.fps) as cap, \
                PoseEstimator(model_path=args.model) as pose:
            print(f"Opened source {source!r} at resolution {cap.resolution[0]}x{cap.resolution[1]}")
            fps_note = cap.fps_warning()
            if fps_note:
                print(fps_note)
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

                    # Drain any Feather Sense samples into /feather/<stream>.
                    if feather is not None:
                        for rec in feather.poll():
                            recorder.append_sensor(
                                rec.name, rec.timestamp_ms, rec.values, rec.fields
                            )

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
        if feather is not None:
            feather.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
