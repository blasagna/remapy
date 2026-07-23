"""Scrub an existing recording and label time segments (annotations).

Opens a recording ``.h5`` written by ``recording.recorder.HDF5Recorder`` in an
OpenCV window, lets you scrub through frames, mark in/out points, and attach a
text label to the segment between them. Segments are stored as an interval table
in the same file (see ``recording.annotations``); edits are saved immediately.

Run::

    pixi run annotate recording.hdf5

Keys (also printed with '?'):

    , / .   step back / forward one frame
    < / >   jump back / forward ~1 second
    space   play / pause
    i       mark the in-point at the playhead
    o       mark the out-point -> type a label in the terminal
    x       delete the annotation nearest the playhead
    p       toggle the pose skeleton overlay
    ?       print this help
    q / Esc quit
"""

import argparse

import cv2
import numpy as np

from pose_estimation.draw import draw_skeleton
from recording.annotations import AnnotationStore
from recording.reader import Recording

WINDOW_NAME = "remapy annotate"
QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc

_STRIP_H = 50  # pixels of timeline strip drawn at the bottom of the frame
_MAX_LANES = 4  # overlapping-segment rows before colors wrap
_MAX_READOUT = 4  # active-label lines drawn before the rest are summarized
_LEGEND_CHARS = 30  # label text is truncated to this in the legend row
_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Stable-ish BGR palette; a label keeps its color for the session via _label_color.
_PALETTE = [
    (0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 200),
    (0, 128, 255), (200, 255, 0), (255, 0, 0), (128, 0, 255),
]

HELP_TEXT = __doc__[__doc__.index("Keys"):]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5", help="Recording .h5 file to annotate (edited in place).")
    return parser.parse_args(argv)


def _jump_frames(rec: Recording) -> int:
    """Frames spanning ~1 second, from the median fps (default 30 if unknown)."""
    fps = rec.fps()
    fps = fps[~np.isnan(fps)] if fps.size else fps
    median = float(np.median(fps)) if fps.size else 30.0
    return max(1, int(round(median)))


def _label_color(label: str, colors: dict[str, tuple]) -> tuple:
    if label not in colors:
        colors[label] = _PALETTE[len(colors) % len(_PALETTE)]
    return colors[label]


def _put_text(frame, text, org, scale, color, thickness=1) -> None:
    """Draw text with a black outline so it stays readable over arbitrary footage."""
    cv2.putText(frame, text, org, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _spans_playhead(ann, playhead_ms: int) -> bool:
    """Whether ``ann`` covers ``playhead_ms``. One definition, used by readout+delete."""
    return ann.start_ms <= playhead_ms <= ann.end_ms


def _active_labels(annotations, playhead_ms: int) -> list:
    """Annotations covering the playhead, earliest first. Several may overlap."""
    return sorted(
        (a for a in annotations if _spans_playhead(a, playhead_ms)),
        key=lambda a: a.start_ms,
    )


def _draw_readout(frame, annotations, playhead_ms, colors) -> None:
    """Name the segment(s) under the playhead, each in that label's strip color."""
    active = _active_labels(annotations, playhead_ms)
    y = 52
    for ann in active[:_MAX_READOUT]:
        _put_text(frame, f"> {ann.label}", (10, y), 0.7,
                  _label_color(ann.label, colors), 2)
        y += 26
    extra = len(active) - _MAX_READOUT
    if extra > 0:
        _put_text(frame, f"  +{extra} more", (10, y), 0.6, (200, 200, 200))


def _draw_legend(frame, annotations, colors, strip_top: int, w: int) -> None:
    """A swatch->label row above the strip, so every span color has a name."""
    labels = list(dict.fromkeys(a.label for a in annotations))  # distinct, ordered
    if not labels:
        return
    y = strip_top - 8
    x = 10
    for i, label in enumerate(labels):
        text = _truncate(label, _LEGEND_CHARS)
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.45, 1)
        need = 14 + tw + 12
        if x + need > w - 10:
            _put_text(frame, f"+{len(labels) - i}", (x, y), 0.45, (200, 200, 200))
            return
        cv2.rectangle(frame, (x, y - 9), (x + 9, y), _label_color(label, colors), -1)
        _put_text(frame, text, (x + 14, y), 0.45, (235, 235, 235))
        x += need


def _assign_lanes(annotations) -> dict[int, int]:
    """Greedy interval scheduling: map each annotation index -> a lane (row) number."""
    lane_index: dict[int, int] = {}
    lane_end: list[int] = []  # lane_end[k] = end_ms of the last segment placed in lane k
    for ann in sorted(annotations, key=lambda a: a.start_ms):
        placed = False
        for k, end in enumerate(lane_end):
            if end <= ann.start_ms:
                lane_end[k] = ann.end_ms
                lane_index[ann.index] = k
                placed = True
                break
        if not placed:
            lane_index[ann.index] = len(lane_end)
            lane_end.append(ann.end_ms)
    return lane_index


def _render(rec, frame_idx, in_point_ms, annotations, playing, colors,
            show_pose=True, pose_present=None):
    """Draw one displayed frame: the video plus the timeline strip and status text."""
    frame = rec.frame(frame_idx).copy()
    h, w = frame.shape[:2]
    strip_top = h - _STRIP_H

    # Skeleton goes on before the strip is painted, or limbs in the bottom _STRIP_H
    # pixels would draw across the timeline instead of under it.
    if show_pose and (pose_present is None or pose_present[frame_idx]):
        draw_skeleton(frame, rec.landmarks_norm[frame_idx], rec.pose_connections,
                      visibility=rec.visibility[frame_idx])

    t0 = int(rec.timestamps_ms[0])
    t1 = int(rec.timestamps_ms[-1])
    span = max(1, t1 - t0)

    def x_of(ms: int) -> int:
        return int(np.clip((ms - t0) / span, 0.0, 1.0) * (w - 1))

    # Strip background.
    cv2.rectangle(frame, (0, strip_top), (w, h), (30, 30, 30), -1)

    # Segments, one lane per overlapping group.
    lanes = _assign_lanes(annotations)
    n_lanes = min(_MAX_LANES, max(1, (max(lanes.values()) + 1) if lanes else 1))
    lane_h = (_STRIP_H - 6) // n_lanes
    for ann in annotations:
        lane = lanes[ann.index] % _MAX_LANES
        y0 = strip_top + 3 + lane * lane_h
        y1 = y0 + lane_h - 1
        color = _label_color(ann.label, colors)
        cv2.rectangle(frame, (x_of(ann.start_ms), y0), (x_of(ann.end_ms), y1), color, -1)

    # Pending in-point marker (yellow) and playhead (red).
    playhead_ms = int(rec.timestamps_ms[frame_idx])
    if in_point_ms is not None:
        ix = x_of(in_point_ms)
        cv2.line(frame, (ix, strip_top), (ix, h), (0, 255, 255), 2)
    px = x_of(playhead_ms)
    cv2.line(frame, (px, strip_top), (px, h), (0, 0, 255), 2)

    # Status text (top-left), then the label(s) under the playhead beneath it.
    state = "PLAY" if playing else "PAUSE"
    pending = "  in-point set" if in_point_ms is not None else ""
    pose_off = "" if show_pose else "  pose off"
    text = f"{frame_idx + 1}/{len(rec)}  {playhead_ms} ms  [{state}]{pending}{pose_off}"
    _put_text(frame, text, (10, 24), 0.6, (255, 255, 255))
    _draw_readout(frame, annotations, playhead_ms, colors)
    _draw_legend(frame, annotations, colors, strip_top, w)
    return frame


def _apply_step(key: int, frame_idx: int, jump: int, n: int) -> int:
    delta = {ord(","): -1, ord("."): 1, ord("<"): -jump, ord(">"): jump}[key]
    return int(np.clip(frame_idx + delta, 0, n - 1))


def _mark_out_and_prompt(store, rec, frame, frame_idx, in_point_ms) -> None:
    """Prompt (in the terminal) for a label for [in_point, playhead] and store it."""
    if in_point_ms is None:
        print("Mark an in-point first (press 'i').")
        return
    out_ms = int(rec.timestamps_ms[frame_idx])
    start_ms, end_ms = sorted((in_point_ms, out_ms))
    # imshow only paints after a waitKey; show the hint before the blocking input().
    hint = frame.copy()
    cv2.putText(hint, "Enter label in terminal...", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, hint)
    cv2.waitKey(1)
    print(f"Segment {start_ms}-{end_ms} ms.")
    label = input("Label (blank to cancel): ").strip()
    if not label:
        print("Cancelled.")
        return
    idx = store.add(label, start_ms, end_ms)
    print(f"Saved annotation #{idx}: {label!r} [{start_ms}, {end_ms}] ms.")


def _delete_nearest(store, rec, frame_idx) -> None:
    segs = store.list()
    if not segs:
        print("No annotations to delete.")
        return
    playhead = int(rec.timestamps_ms[frame_idx])

    def dist(a):
        if _spans_playhead(a, playhead):
            return 0
        return min(abs(playhead - a.start_ms), abs(playhead - a.end_ms))

    nearest = min(segs, key=dist)
    store.delete(nearest.index)
    print(f"Deleted annotation #{nearest.index}: {nearest.label!r} "
          f"[{nearest.start_ms}, {nearest.end_ms}] ms.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # h5py locking: opening "r" before "r+" on the same path in one process raises.
    # AnnotationStore ("r+") MUST be opened before Recording ("r"). The reverse
    # order crashes; this order is verified safe (see tests/test_annotations.py::
    # AnnotationStoreTests.test_rw_then_ro_handle_coexist).
    try:
        store = AnnotationStore(args.h5)
    except OSError as exc:
        print(f"Error: {exc}")
        return 1

    rec = None
    try:
        rec = Recording(args.h5)
        if len(rec) == 0:
            print("Recording has no frames to annotate.")
            return 1

        print(HELP_TEXT)
        frame_idx = 0
        playing = False
        in_point_ms = None
        jump = _jump_frames(rec)
        frame_delay_ms = max(1, int(round(1000.0 / jump)))
        colors: dict[str, tuple] = {}
        show_pose = True
        # pose_present is a property that re-slices the whole world array on each
        # access; cache it once rather than paying that per rendered frame.
        pose_present = rec.pose_present

        while True:
            frame = _render(rec, frame_idx, in_point_ms, store.list(), playing, colors,
                            show_pose, pose_present)
            cv2.imshow(WINDOW_NAME, frame)
            # waitKey(0) blocks (the paused state); waitKey(delay) advances while playing.
            key = cv2.waitKey(frame_delay_ms if playing else 0) & 0xFF

            if key in QUIT_KEYS:
                break
            elif key == ord(" "):
                playing = not playing
            elif key in (ord(","), ord("."), ord("<"), ord(">")):
                playing = False
                frame_idx = _apply_step(key, frame_idx, jump, len(rec))
            elif key == ord("i"):
                in_point_ms = int(rec.timestamps_ms[frame_idx])
                print(f"In-point at {in_point_ms} ms (frame {frame_idx + 1}).")
            elif key == ord("o"):
                _mark_out_and_prompt(store, rec, frame, frame_idx, in_point_ms)
                in_point_ms = None
            elif key == ord("x"):
                _delete_nearest(store, rec, frame_idx)
            elif key == ord("p"):
                show_pose = not show_pose
            elif key == ord("?"):
                print(HELP_TEXT)

            if playing:
                if frame_idx >= len(rec) - 1:
                    playing = False  # auto-pause at the end
                else:
                    frame_idx += 1
    except KeyboardInterrupt:
        print("\nInterrupted; shutting down.")
    finally:
        cv2.destroyAllWindows()
        if rec is not None:
            rec.close()
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
