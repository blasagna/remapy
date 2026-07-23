"""Replay a stored HDF5 recording into Rerun — as a ``.rrd`` file or the viewer.

The live pipeline (``rerun_viewer.main``) logs to Rerun *while* capturing. This
module does the same from an already-recorded ``recording.reader.Recording``, so
a past session can be reviewed with the skeleton overlay and joint-angle plots
rather than as a plain mp4 (``recording.export``).

Everything the file happens to carry is replayed: video + 2D skeleton + angles
always, the Feather Sense plots when ``/feather`` streams are present, and the
labeled time segments when ``/annotations`` is. The pose result is rebuilt from
the stored landmark rows and handed to the *unmodified* ``PoseRerunLogger``, and
the archived JPEG blobs are logged verbatim (no decode/re-encode round trip), so
what you see is what was recorded.

**Sensor alignment caveat:** the Feather Sense timestamps are the *device* clock
(ms since board boot) and the offset between it and the recording clock is never
stored. Sensor time is therefore anchored so the first sample sits at the
recording's ``t=0``: relative sensor timing is exact, but video↔sensor alignment
is nominal.

CLI::

    python -m rerun_viewer.replay session.h5 out.rrd   # write a .rrd
    python -m rerun_viewer.replay session.h5           # spawn the viewer
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recording.reader import Recording

from .viewer import LAYOUTS, PoseRerunLogger

# Streams the reader derives on read (see recording.reader.Recording.feather).
# The logger re-derives them from the raw accel exactly as it does live, so
# replaying them would log each one twice.
_DERIVED_STREAMS = ("gravity", "linear_accel")


def _pose_result(norm_row: np.ndarray, world_row: np.ndarray, present: bool):
    """Rebuild the duck-typed PoseLandmarkerResult the logger reads.

    Empty landmark lists on a frame with no pose, which is the logger's existing
    "clear the stale skeleton" path.
    """
    if not present:
        return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    return SimpleNamespace(
        pose_landmarks=[[SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in norm_row]],
        pose_world_landmarks=[
            [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in world_row]
        ],
    )


def _sensor_records(feather: dict) -> list:
    """Flatten the recorded Feather streams into timestamp-ordered SensorRecords.

    Only the raw streams (``derived=False``) are replayed; see
    ``_DERIVED_STREAMS``. The streams are asynchronous and multi-rate, so they
    are merged into one time-ordered list, matching what ``poll()`` yields live.
    """
    records = []
    for name, stream in feather.items():
        if name in _DERIVED_STREAMS or getattr(stream, "derived", False):
            continue
        if name == "error":
            for ts, source, message in zip(stream.timestamps_ms, stream.source, stream.message):
                records.append(
                    SimpleNamespace(
                        name=name,
                        timestamp_ms=int(ts),
                        values=(str(source), str(message)),
                        fields=stream.fields,
                    )
                )
            continue
        for ts, values in zip(stream.timestamps_ms, stream.values):
            records.append(
                SimpleNamespace(
                    name=name,
                    timestamp_ms=int(ts),
                    values=tuple(float(v) for v in values),
                    fields=stream.fields,
                )
            )
    records.sort(key=lambda r: r.timestamp_ms)
    return records


def replay_recording(
    h5_path: Path | str,
    save_path: Path | str | None = None,
    spawn: bool = True,
    layout: str = "split",
    memory_limit: str = "75%",
    max_frames: int | None = None,
) -> int:
    """Log a recording to Rerun. Returns the number of frames logged."""
    with Recording(h5_path) as rec:
        n = len(rec)
        if max_frames is not None:
            n = min(n, max_frames)
        if n == 0:
            raise ValueError(f"Recording {str(h5_path)!r} has no frames to replay.")

        height = int(rec.metadata["image_height"])
        width = int(rec.metadata["image_width"])
        logger = PoseRerunLogger(
            application_id="remapy recording",
            spawn=spawn,
            save_path=str(save_path) if save_path is not None else None,
            memory_limit=memory_limit,
            layout=layout,
            feather=bool(rec.feather),
            annotations=bool(rec.annotations),
        )

        ts = rec.timestamps_ms
        t0 = float(ts[0])
        fps = rec.fps()  # length N-1: the rate *into* each frame after the first
        present = rec.pose_present
        for i in range(n):
            logger.log_frame(
                i + 1,
                (float(ts[i]) - t0) / 1000.0,
                float(fps[i - 1]) if i > 0 else 0.0,
                None,
                _pose_result(rec.landmarks_norm[i], rec.landmarks_world[i], bool(present[i])),
                jpeg_bytes=rec.frame_jpeg(i),
                image_size=(height, width),
            )

        if rec.annotations:
            logger.log_annotations(rec.annotations, ts)
        records = _sensor_records(rec.feather)
        if records:
            # elapsed_s=0.0 anchors the first sample at the recording's t=0; the
            # true offset between the device and recording clocks is not stored.
            logger.log_sensors(records, 0.0)
        return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5", help="Input recording .h5 file.")
    parser.add_argument(
        "rrd",
        nargs="?",
        default=None,
        help="Output .rrd path. Omit to spawn the Rerun viewer instead.",
    )
    parser.add_argument(
        "--spawn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Spawn the viewer. Default: on when no output .rrd is given, off when one is.",
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUTS,
        default="split",
        help="split (default) = camera/pose and line plots side by side; tabs = separate tabs.",
    )
    parser.add_argument(
        "--memory-limit",
        default="75%",
        help="Viewer in-memory store cap (e.g. '2GB', '50%%'). Default: 75%% of system RAM. "
        "Ignored when writing a .rrd.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Replay only the first N frames."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spawn = (args.rrd is None) if args.spawn is None else args.spawn
    try:
        n = replay_recording(
            args.h5,
            save_path=args.rrd,
            spawn=spawn,
            layout=args.layout,
            memory_limit=args.memory_limit,
            max_frames=args.max_frames,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    if args.rrd:
        print(f"Logged {n} frame(s) to {args.rrd}. Open with: rerun {args.rrd}")
    else:
        print(f"Logged {n} frame(s) to the Rerun viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
