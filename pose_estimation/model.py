"""Locate (and, if needed, download) the MediaPipe pose landmarker model bundle.

The MediaPipe Tasks API needs a ``.task`` model file that is *not* shipped with
the pip package. We default to the "lite" bundle — the fastest tier, well suited
to a live webcam — and cache it under ``pose_estimation/models/``.
"""

import urllib.request
from pathlib import Path

# Float16 "lite" pose landmarker bundle from Google's model hub.
LITE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "pose_landmarker_lite.task"


def ensure_model(path: Path | str | None = None, url: str = LITE_MODEL_URL) -> Path:
    """Return a path to the model bundle, downloading it on first use.

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
    print(f"Downloading pose model from {url}\n  -> {path}")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    print("Model download complete.")
    return path
