"""Locate (and, if needed, download) the MediaPipe face detector model.

The MediaPipe Tasks API needs a ``.tflite`` model file that is *not* shipped with
the pip package. We default to the "blaze_face short-range" detector — fast and
well suited to a webcam at conversational distance — and cache it under
``face_blur/models/``.
"""

import urllib.request
from pathlib import Path

# Float16 blaze_face short-range detector from Google's model hub.
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "blaze_face_short_range.tflite"


def ensure_model(path: Path | str | None = None, url: str = FACE_MODEL_URL) -> Path:
    """Return a path to the model file, downloading it on first use.

    Parameters
    ----------
    path:
        Where the model lives / should be cached. Defaults to
        :data:`DEFAULT_MODEL_PATH`.
    url:
        Source URL to download from if the file is missing.
    """
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face model from {url}\n  -> {path}")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    print("Model download complete.")
    return path
