"""Tests for :mod:`recording.annotations` and :attr:`recording.reader.Recording.annotations`.

Real HDF5 is written to a temp file (fast, self-contained) exactly like
``tests/test_recording.py``; nothing is mocked. Base recordings are built via
``HDF5Recorder`` + the duck-typed fakes.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py

from recording.annotations import LABEL_DS, AnnotationStore
from recording.reader import Recording
from recording.recorder import HDF5Recorder
from tests.fakes import make_landmarks, pose_result, solid_frame


def _write_base_recording(path, n=3):
    """Write a minimal valid recording (no annotations) to annotate later."""
    with HDF5Recorder(path) as rec:
        for i in range(n):
            result = pose_result(poses_norm=[make_landmarks()], poses_world=[make_landmarks()])
            rec.append(solid_frame(), i * 33, result)


class AnnotationStoreTests(unittest.TestCase):
    def _recording(self, d, n=3):
        path = Path(d) / "rec.h5"
        _write_base_recording(path, n=n)
        return path

    def test_group_absent_until_first_add(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with h5py.File(path, "r") as f:
                self.assertNotIn("annotations", f)
            with AnnotationStore(path) as store:
                self.assertEqual(store.list(), [])
                store.add("walking", 0, 66)
            with h5py.File(path, "r") as f:
                self.assertIn(LABEL_DS, f)

    def test_add_list_round_trip_with_overlap(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                store.add("walking", 0, 100)
                store.add("occluded", 50, 150)  # overlaps the first -> both are kept
                segs = store.list()
            self.assertEqual(len(segs), 2)
            self.assertEqual([s.label for s in segs], ["walking", "occluded"])
            self.assertEqual((segs[0].start_ms, segs[0].end_ms), (0, 100))
            self.assertEqual((segs[1].start_ms, segs[1].end_ms), (50, 150))

    def test_delete_tombstones_without_reordering(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                i0 = store.add("a", 0, 10)
                i1 = store.add("b", 20, 30)
                store.delete(i0)
                segs = store.list()
            self.assertEqual([s.index for s in segs], [i1])
            self.assertEqual(segs[0].label, "b")

    def test_delete_out_of_range_raises_indexerror(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                store.add("a", 0, 10)
                with self.assertRaises(IndexError):
                    store.delete(5)

    def test_delete_before_any_add_raises_indexerror(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                with self.assertRaises(IndexError):
                    store.delete(0)

    def test_double_delete_raises_keyerror(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                idx = store.add("a", 0, 10)
                store.delete(idx)
                with self.assertRaises(KeyError):
                    store.delete(idx)

    def test_end_before_start_raises_valueerror(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                with self.assertRaises(ValueError):
                    store.add("a", 100, 50)

    def test_reopen_persists_across_close(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            with AnnotationStore(path) as store:
                store.add("walking", 0, 66)
            with AnnotationStore(path) as store:
                segs = store.list()
            self.assertEqual(len(segs), 1)
            self.assertEqual(segs[0].label, "walking")

    def test_open_missing_file_raises(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(OSError):
                AnnotationStore(Path(d) / "does_not_exist.h5")

    def test_add_after_close_raises_runtimeerror(self):
        with TemporaryDirectory() as d:
            path = self._recording(d)
            store = AnnotationStore(path)
            store.close()
            with self.assertRaises(RuntimeError):
                store.add("a", 0, 10)

    def test_rw_then_ro_handle_coexist(self):
        # Regression pin for the h5py locking constraint the CLI depends on:
        # AnnotationStore ("r+") opened BEFORE Recording ("r") is safe, and the
        # reader sees rows written through the store. (The reverse order raises.)
        with TemporaryDirectory() as d:
            path = self._recording(d)
            store = AnnotationStore(path)
            store.add("walking", 0, 66)
            rec = Recording(path)  # opens "r" while store's "r+" is still open
            try:
                self.assertEqual(len(rec.annotations), 1)
                self.assertEqual(rec.annotations[0].label, "walking")
            finally:
                rec.close()
                store.close()


class RecordingAnnotationsTests(unittest.TestCase):
    def test_reads_back_non_deleted(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "rec.h5"
            _write_base_recording(path)
            with AnnotationStore(path) as store:
                store.add("walking", 0, 50)
                dead = store.add("oops", 60, 70)
                store.delete(dead)
            with Recording(path) as rec:
                self.assertEqual([a.label for a in rec.annotations], ["walking"])

    def test_empty_on_file_without_group(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "rec.h5"
            _write_base_recording(path)
            with Recording(path) as rec:
                self.assertEqual(rec.annotations, [])


if __name__ == "__main__":
    unittest.main()
