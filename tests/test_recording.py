"""Tests for the :mod:`recording` package (recorder / reader / export).

HDF5 is real (written to a temp file) since it is fast and self-contained. The
mp4 ``cv2.VideoWriter`` is mocked (H.264 is unavailable in this OpenCV build and
we don't want to depend on any codec). MediaPipe results are duck-typed fakes.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from recording.export import export_mp4
from recording.reader import Recording
from recording.recorder import LANDMARK_NAMES, NUM_LANDMARKS, HDF5Recorder
from tests.fakes import FakeVideoWriter, make_landmarks, pose_result, solid_frame


def _write_empty_recording(path):
    """Write a valid recording file with datasets present but zero rows."""
    import h5py

    from recording.recorder import POSE_CONNECTIONS
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamps_ms", (0,), maxshape=(None,), dtype="int64")
        f.create_dataset("video/jpeg", (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.uint8))
        for name in ("landmarks_norm", "landmarks_world"):
            f.create_dataset(f"pose/{name}", (0, NUM_LANDMARKS, 3),
                             maxshape=(None, NUM_LANDMARKS, 3), dtype="float32")
        for name in ("visibility", "presence"):
            f.create_dataset(f"pose/{name}", (0, NUM_LANDMARKS),
                             maxshape=(None, NUM_LANDMARKS), dtype="float32")
        f.create_dataset("meta/landmark_names",
                         data=np.array(LANDMARK_NAMES, dtype=h5py.string_dtype()))
        f.create_dataset("meta/pose_connections", data=np.array(POSE_CONNECTIONS, dtype="int32"))
        f.attrs["image_width"] = 64
        f.attrs["image_height"] = 48


def _write_recording(path, *, frames, timestamps, poses, faces_blurred=True):
    """Write a recording; ``poses[i]`` is a landmark list or ``None`` (no pose)."""
    with HDF5Recorder(path, faces_blurred=faces_blurred, blur_style="box") as rec:
        for frame, ts, lms in zip(frames, timestamps, poses):
            if lms is None:
                result = pose_result()
            else:
                result = pose_result(poses_norm=[lms], poses_world=[lms])
            rec.append(frame, ts, result)


class RecorderTests(unittest.TestCase):
    def test_writes_frames_and_metadata(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "rec.h5"
            frames = [solid_frame(value=v) for v in (50, 100, 150)]
            _write_recording(
                path, frames=frames, timestamps=[0, 33, 66],
                poses=[make_landmarks(), None, make_landmarks()],
            )
            import h5py
            with h5py.File(path, "r") as f:
                self.assertEqual(f.attrs["num_frames"], 3)
                self.assertEqual(f.attrs["num_landmarks"], NUM_LANDMARKS)
                self.assertTrue(bool(f.attrs["faces_blurred"]))
                self.assertEqual(f.attrs["blur_style"], "box")
                self.assertEqual(f["pose/landmarks_world"].shape, (3, NUM_LANDMARKS, 3))
                self.assertEqual(len(f["video/jpeg"]), 3)
                # No-pose frame is stored as NaN.
                self.assertTrue(np.all(np.isnan(f["pose/landmarks_world"][1])))
                # Pose frame is finite.
                self.assertFalse(np.any(np.isnan(f["pose/landmarks_world"][0])))

    def test_landmark_names_length(self):
        self.assertEqual(len(LANDMARK_NAMES), NUM_LANDMARKS)

    def test_append_after_close_raises(self):
        with TemporaryDirectory() as d:
            rec = HDF5Recorder(Path(d) / "rec.h5")
            rec.append(solid_frame(), 0, pose_result())
            rec.close()
            with self.assertRaises(RuntimeError):
                rec.append(solid_frame(), 1, pose_result())

    def test_empty_recording_has_no_num_frames_attr(self):
        # Datasets are created lazily on first append; an unused recorder is valid.
        with TemporaryDirectory() as d:
            path = Path(d) / "rec.h5"
            with HDF5Recorder(path):
                pass
            import h5py
            with h5py.File(path, "r") as f:
                self.assertNotIn("num_frames", f.attrs)

    def test_parallel_video_writer_used(self):
        created = {}

        def make_writer(vpath, fourcc, fps, size):
            writer = FakeVideoWriter(vpath, fourcc, fps, size)
            created["writer"] = writer
            return writer

        with TemporaryDirectory() as d, \
                mock.patch("cv2.VideoWriter", side_effect=make_writer):
            path = Path(d) / "rec.h5"
            with HDF5Recorder(path, video_path=Path(d) / "out.mp4") as rec:
                rec.append(solid_frame(), 0, pose_result())
                rec.append(solid_frame(), 33, pose_result())
        self.assertEqual(len(created["writer"].frames), 2)
        self.assertTrue(created["writer"].released)


class ReaderTests(unittest.TestCase):
    def _make(self, d):
        path = Path(d) / "rec.h5"
        frames = [solid_frame(h=48, w=64, value=v) for v in (60, 120, 180)]
        _write_recording(
            path, frames=frames, timestamps=[0, 40, 80],
            poses=[make_landmarks(), None, make_landmarks()],
        )
        return path

    def test_len_and_pose_present(self):
        with TemporaryDirectory() as d:
            with Recording(self._make(d)) as rec:
                self.assertEqual(len(rec), 3)
                np.testing.assert_array_equal(rec.pose_present, [True, False, True])

    def test_fps_from_timestamps(self):
        with TemporaryDirectory() as d:
            with Recording(self._make(d)) as rec:
                fps = rec.fps()
                # dt = 40ms -> 25 fps between each consecutive pair.
                np.testing.assert_allclose(fps, [25.0, 25.0])

    def test_frame_decodes_to_bgr(self):
        with TemporaryDirectory() as d:
            with Recording(self._make(d)) as rec:
                frame = rec.frame(0)
                self.assertEqual(frame.shape, (48, 64, 3))
                self.assertEqual(frame.dtype, np.uint8)

    def test_frames_iterates_all(self):
        with TemporaryDirectory() as d:
            with Recording(self._make(d)) as rec:
                self.assertEqual(len(list(rec.frames())), 3)

    def test_angles_dataframe_nan_on_missing_pose(self):
        with TemporaryDirectory() as d:
            with Recording(self._make(d)) as rec:
                df = rec.angles()
        self.assertEqual(len(df), 3)
        # Frame 1 had no pose -> all-NaN row.
        self.assertTrue(df.iloc[1].isna().all())
        # Frames 0 and 2 have finite angles.
        self.assertFalse(df.iloc[0].isna().all())

    def test_metadata_and_landmark_names(self):
        with TemporaryDirectory() as d:
            with Recording(self._make(d)) as rec:
                self.assertEqual(len(rec.landmark_names), NUM_LANDMARKS)
                self.assertEqual(rec.metadata["blur_style"], "box")
                self.assertEqual(rec.pose_connections.shape[1], 2)


class ExportTests(unittest.TestCase):
    def _make(self, d, timestamps=(0, 33, 66)):
        path = Path(d) / "rec.h5"
        frames = [solid_frame() for _ in timestamps]
        _write_recording(
            path, frames=frames, timestamps=list(timestamps),
            poses=[None] * len(timestamps),
        )
        return path

    def test_writes_every_frame(self):
        writers = []

        def make_writer(vpath, fourcc, fps, size):
            w = FakeVideoWriter(vpath, fourcc, fps, size)
            writers.append(w)
            return w

        with TemporaryDirectory() as d, \
                mock.patch("cv2.VideoWriter", side_effect=make_writer):
            out = export_mp4(self._make(d), Path(d) / "out.mp4")
            self.assertEqual(out, Path(d) / "out.mp4")
        self.assertEqual(len(writers[0].frames), 3)
        self.assertTrue(writers[0].released)

    def test_default_fps_from_median_timestamps(self):
        writers = []

        def make_writer(vpath, fourcc, fps, size):
            w = FakeVideoWriter(vpath, fourcc, fps, size)
            writers.append(w)
            return w

        with TemporaryDirectory() as d, \
                mock.patch("cv2.VideoWriter", side_effect=make_writer):
            # Even 33ms spacing -> ~30 fps.
            export_mp4(self._make(d, timestamps=(0, 33, 66, 99)), Path(d) / "out.mp4")
        self.assertEqual(writers[0].fps, 30.0)

    def test_explicit_fps_overrides(self):
        writers = []

        def make_writer(vpath, fourcc, fps, size):
            w = FakeVideoWriter(vpath, fourcc, fps, size)
            writers.append(w)
            return w

        with TemporaryDirectory() as d, \
                mock.patch("cv2.VideoWriter", side_effect=make_writer):
            export_mp4(self._make(d), Path(d) / "out.mp4", fps=12.0)
        self.assertEqual(writers[0].fps, 12.0)

    def test_empty_recording_raises(self):
        # A readable-but-zero-frame recording: datasets exist, but have no rows.
        with TemporaryDirectory() as d:
            path = Path(d) / "empty.h5"
            _write_empty_recording(path)
            with mock.patch("cv2.VideoWriter", side_effect=AssertionError("unused")):
                with self.assertRaises(ValueError):
                    export_mp4(path, Path(d) / "out.mp4")

    def test_writer_open_failure_raises(self):
        with TemporaryDirectory() as d, \
                mock.patch("cv2.VideoWriter", return_value=FakeVideoWriter(
                    "x", 0, 30, (64, 48), opened=False)):
            with self.assertRaises(RuntimeError):
                export_mp4(self._make(d), Path(d) / "out.mp4")


if __name__ == "__main__":
    unittest.main()
