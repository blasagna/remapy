"""Tests for :mod:`rerun_viewer.viewer`.

The ``rerun`` SDK is mocked entirely (``rerun_viewer.viewer.rr``), so nothing is
logged to a real recording, spawned, or saved. We assert on the calls made.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from tests.fakes import make_landmarks, pose_result, solid_frame


class InitTests(unittest.TestCase):
    def test_spawn_by_default_forwards_memory_limit(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            PoseRerunLogger(spawn=True, memory_limit="50%")
        rr.init.assert_called_once()
        rr.spawn.assert_called_once_with(memory_limit="50%")
        rr.save.assert_not_called()

    def test_save_path_uses_save_not_spawn(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            PoseRerunLogger(save_path="out.rrd", spawn=True)
        rr.save.assert_called_once_with("out.rrd")
        rr.spawn.assert_not_called()

    def test_no_spawn_no_save_is_headless(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            PoseRerunLogger(spawn=False)
        rr.spawn.assert_not_called()
        rr.save.assert_not_called()


class LogFrameTests(unittest.TestCase):
    def _logger(self, rr):
        from rerun_viewer.viewer import PoseRerunLogger
        return PoseRerunLogger(spawn=False)

    def test_logs_image_and_fps(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            logger.log_frame(0, 0.0, 30.0, solid_frame(), pose_result())
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("video/image", logged)
        self.assertIn("metrics/fps", logged)

    def test_no_pose_clears_skeletons(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            logger.log_frame(1, 0.1, 30.0, solid_frame(), pose_result())
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("video/image/skeleton", logged)
        self.assertIn("pose3d", logged)
        # No skeleton keypoints / angles are logged when there's no pose.
        self.assertNotIn("video/image/keypoints", logged)

    def test_pose_present_logs_skeleton_and_angles(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            lms = make_landmarks()
            result = pose_result(poses_norm=[lms], poses_world=[lms])
            logger.log_frame(2, 0.2, 29.0, solid_frame(), result)
        logged = [c.args[0] for c in rr.log.call_args_list]
        self.assertIn("video/image/skeleton", logged)
        self.assertIn("video/image/keypoints", logged)
        self.assertTrue(any(name.startswith("metrics/angles/") for name in logged))

    def test_sets_both_timelines(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            logger = self._logger(rr)
            logger.log_frame(5, 1.5, 30.0, solid_frame(), pose_result())
        self.assertEqual(rr.set_time.call_count, 2)


class LogFrameJpegTests(unittest.TestCase):
    """The replay path hands over the archived JPEG instead of a BGR frame."""

    def test_logs_stored_bytes_without_reencoding(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr, \
                mock.patch("rerun_viewer.viewer.cv2.imencode") as imencode:
            from rerun_viewer.viewer import PoseRerunLogger
            logger = PoseRerunLogger(spawn=False)
            logger.log_frame(1, 0.0, 30.0, None, pose_result(), jpeg_bytes=b"JPEGBYTES")
        imencode.assert_not_called()
        image = [c for c in rr.log.call_args_list if c.args[0] == "video/image"]
        self.assertEqual(len(image), 1)
        rr.EncodedImage.assert_called_once_with(contents=b"JPEGBYTES", media_type="image/jpeg")

    def test_image_size_scales_the_2d_skeleton_without_a_frame(self):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            logger = PoseRerunLogger(spawn=False)
            lms = make_landmarks()
            logger.log_frame(
                1, 0.0, 30.0, None,
                pose_result(poses_norm=[lms], poses_world=[lms]),
                jpeg_bytes=b"J", image_size=(48, 64),
            )
        pts = rr.Points2D.call_args.args[0]
        # make_landmarks spreads x/y over 0..1, so the last point sits at (w, h).
        np.testing.assert_allclose(pts[-1], [64.0, 48.0])


def _origins(blueprint):
    """Every view origin in a blueprint, walking the container tree."""
    found = []

    def walk(node):
        origin = getattr(node, "origin", None)
        if origin is not None:
            found.append(str(origin))
        for child in getattr(node, "contents", ()) or ():
            walk(child)

    walk(blueprint.root_container)
    return found


class BlueprintTests(unittest.TestCase):
    def test_annotations_view_only_when_requested(self):
        from rerun_viewer.viewer import _build_blueprint
        for layout in ("split", "tabs"):
            with self.subTest(layout=layout):
                self.assertNotIn("annotations", _origins(_build_blueprint(layout)))
                self.assertIn(
                    "annotations", _origins(_build_blueprint(layout, annotations=True))
                )
                # Feather + annotations coexist as separate tabs.
                both = _origins(_build_blueprint(layout, feather=True, annotations=True))
                self.assertIn("annotations", both)
                self.assertIn("feather/accel", both)


class LogAnnotationsTests(unittest.TestCase):
    def test_logs_label_at_start_and_clear_at_end(self):
        from recording.annotations import Annotation
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            logger = PoseRerunLogger(spawn=False)
            logger.log_annotations(
                [Annotation(0, "sit_hold;arms=free", 100, 300)],
                np.array([0, 100, 200, 300, 400]),
            )
        paths = [c.args[0] for c in rr.log.call_args_list]
        self.assertEqual(paths, ["annotations/sit_hold;arms=free"] * 2)
        rr.TextLog.assert_called_once_with("sit_hold;arms=free")
        rr.Clear.assert_called_once_with(recursive=True)

    def test_no_frames_is_a_noop(self):
        from recording.annotations import Annotation
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.viewer import PoseRerunLogger
            logger = PoseRerunLogger(spawn=False)
            logger.log_annotations([Annotation(0, "x", 0, 1)], np.array([]))
        rr.log.assert_not_called()


class ReplayRecordingTests(unittest.TestCase):
    """End-to-end: a real HDF5 recording replayed into a mocked Rerun."""

    def _recording(self, d, *, poses, annotate=None, feather=False):
        from recording.annotations import AnnotationStore
        from recording.recorder import HDF5Recorder

        path = Path(d) / "rec.h5"
        with HDF5Recorder(path, faces_blurred=True, blur_style="box") as rec:
            for i, lms in enumerate(poses):
                result = pose_result() if lms is None else pose_result([lms], [lms])
                rec.append(solid_frame(value=50 + i), i * 33, result)
                if feather:
                    rec.append_sensor("accel", 1000 + i * 10, (0.0, 0.0, 9.8), ["x", "y", "z"])
        if annotate:
            with AnnotationStore(path) as store:
                for label, start, end in annotate:
                    store.add(label, start, end)
        return path

    def _replay(self, path, **kwargs):
        with mock.patch("rerun_viewer.viewer.rr") as rr:
            from rerun_viewer.replay import replay_recording
            n = replay_recording(path, spawn=False, **kwargs)
        return n, rr

    def test_logs_one_image_per_frame_and_clears_untracked(self):
        with TemporaryDirectory() as d:
            path = self._recording(d, poses=[make_landmarks(), None, make_landmarks()])
            n, rr = self._replay(path)
        self.assertEqual(n, 3)
        paths = [c.args[0] for c in rr.log.call_args_list]
        self.assertEqual(paths.count("video/image"), 3)
        # One clear pair for the single untracked frame.
        self.assertEqual(paths.count("pose3d"), 1)
        self.assertEqual(paths.count("video/image/keypoints"), 2)

    def test_max_frames_bounds_the_replay(self):
        with TemporaryDirectory() as d:
            path = self._recording(d, poses=[make_landmarks()] * 4)
            n, rr = self._replay(path, max_frames=2)
        self.assertEqual(n, 2)
        paths = [c.args[0] for c in rr.log.call_args_list]
        self.assertEqual(paths.count("video/image"), 2)

    def test_empty_recording_raises(self):
        import h5py

        from rerun_viewer.replay import replay_recording
        with TemporaryDirectory() as d:
            path = self._recording(d, poses=[make_landmarks()])
            empty = Path(d) / "empty.h5"
            with h5py.File(path, "r") as src, h5py.File(empty, "w") as dst:
                src.copy("meta", dst)
                for k in src.attrs:
                    dst.attrs[k] = src.attrs[k]
                dst.create_dataset("timestamps_ms", (0,), dtype="int64")
                dst.create_dataset("video/jpeg", (0,), dtype=h5py.vlen_dtype(np.uint8))
                for name in ("landmarks_norm", "landmarks_world"):
                    dst.create_dataset(f"pose/{name}", (0, 33, 3), dtype="float32")
                for name in ("visibility", "presence"):
                    dst.create_dataset(f"pose/{name}", (0, 33), dtype="float32")
            with mock.patch("rerun_viewer.viewer.rr"):
                with self.assertRaises(ValueError):
                    replay_recording(empty, spawn=False)

    def test_annotations_are_replayed(self):
        with TemporaryDirectory() as d:
            path = self._recording(
                d, poses=[make_landmarks()] * 3, annotate=[("sit_hold", 0, 66)]
            )
            _, rr = self._replay(path)
        paths = [c.args[0] for c in rr.log.call_args_list]
        self.assertEqual(paths.count("annotations/sit_hold"), 2)  # label + clear

    def test_feather_streams_logged_once_with_derived_motion(self):
        with TemporaryDirectory() as d:
            path = self._recording(d, poses=[make_landmarks()] * 3, feather=True)
            _, rr = self._replay(path)
        paths = [c.args[0] for c in rr.log.call_args_list]
        # 3 raw accel samples, each on 3 axes...
        self.assertEqual(paths.count("feather/accel/x"), 3)
        # ...and the logger derives gravity/linear_accel from them exactly once
        # each (the reader also derives them; replaying those would double up).
        self.assertEqual(paths.count("feather/gravity/x"), 3)
        self.assertEqual(paths.count("feather/linear_accel/x"), 3)

    def test_blueprint_reflects_what_the_file_carries(self):
        with TemporaryDirectory() as d:
            plain = self._recording(d, poses=[make_landmarks()])
            _, rr = self._replay(plain)
            self.assertNotIn("annotations", _origins(rr.send_blueprint.call_args.args[0]))

            rich = self._recording(
                d, poses=[make_landmarks()], annotate=[("sit_hold", 0, 1)], feather=True
            )
            _, rr = self._replay(rich)
            origins = _origins(rr.send_blueprint.call_args.args[0])
            self.assertIn("annotations", origins)
            self.assertIn("feather/accel", origins)


if __name__ == "__main__":
    unittest.main()
