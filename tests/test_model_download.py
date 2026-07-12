"""Tests for the ``ensure_model`` download helpers.

``face_blur.model`` and ``pose_estimation.model`` share the same download/cache
logic; both are exercised here with ``urllib`` mocked so nothing is fetched.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from face_blur import model as face_model
from pose_estimation import model as pose_model


class EnsureModelTests(unittest.TestCase):
    """Run the same suite against both modules."""

    modules = (("face_blur.model", face_model), ("pose_estimation.model", pose_model))

    def setUp(self):
        # ensure_model prints progress on download; keep test output clean.
        printer = mock.patch("builtins.print")
        printer.start()
        self.addCleanup(printer.stop)

    def test_existing_file_is_returned_without_download(self):
        for name, module in self.modules:
            with self.subTest(module=name), TemporaryDirectory() as d:
                path = Path(d) / "model.bin"
                path.write_bytes(b"already here")
                with mock.patch(f"{name}.urllib.request.urlretrieve") as fetch:
                    result = module.ensure_model(path)
                self.assertEqual(result, path)
                fetch.assert_not_called()

    def test_missing_file_is_downloaded(self):
        for name, module in self.modules:
            with self.subTest(module=name), TemporaryDirectory() as d:
                path = Path(d) / "sub" / "model.bin"  # parent dir does not exist yet

                def fake_fetch(url, tmp):
                    Path(tmp).write_bytes(b"downloaded")

                with mock.patch(
                    f"{name}.urllib.request.urlretrieve", side_effect=fake_fetch
                ) as fetch:
                    result = module.ensure_model(path)
                fetch.assert_called_once()
                self.assertTrue(path.exists())
                self.assertEqual(path.read_bytes(), b"downloaded")
                # The temporary .part file is cleaned up.
                self.assertFalse(path.with_suffix(path.suffix + ".part").exists())

    def test_default_url_passed_when_downloading(self):
        for name, module in self.modules:
            with self.subTest(module=name), TemporaryDirectory() as d:
                path = Path(d) / "model.bin"

                def fake_fetch(url, tmp):
                    Path(tmp).write_bytes(b"x")

                with mock.patch(
                    f"{name}.urllib.request.urlretrieve", side_effect=fake_fetch
                ) as fetch:
                    module.ensure_model(path)
                url_arg = fetch.call_args.args[0]
                self.assertTrue(url_arg.startswith("https://"))

    def test_failed_download_cleans_up_partfile(self):
        for name, module in self.modules:
            with self.subTest(module=name), TemporaryDirectory() as d:
                path = Path(d) / "model.bin"
                with mock.patch(
                    f"{name}.urllib.request.urlretrieve",
                    side_effect=OSError("network down"),
                ):
                    with self.assertRaises(OSError):
                        module.ensure_model(path)
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(path.suffix + ".part").exists())

    def test_string_path_accepted(self):
        for name, module in self.modules:
            with self.subTest(module=name), TemporaryDirectory() as d:
                path = Path(d) / "model.bin"
                path.write_bytes(b"here")
                # A str (not Path) must be accepted and normalized.
                result = module.ensure_model(str(path))
                self.assertEqual(result, path)


if __name__ == "__main__":
    unittest.main()
