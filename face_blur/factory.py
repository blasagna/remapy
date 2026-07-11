"""Select a face-redaction backend from a CLI ``--blur-method`` choice.

Keeps the detector-vs-pose branch in one place so every entry point wires it up
identically. Both returned objects share the same
``blur(frame, pose_result) / close()`` context-manager surface, so callers treat
them interchangeably.
"""

from pathlib import Path
from typing import Union

from .blur import FaceBlurrer
from .hybrid import HybridFaceBlurrer
from .pose_blur import PoseFaceBlurrer

BLUR_METHODS = ("detector", "pose", "hybrid")

Blurrer = Union[FaceBlurrer, PoseFaceBlurrer, HybridFaceBlurrer]


def build_blurrer(
    method: str,
    *,
    style: str = "box",
    model_path: Path | str | None = None,
) -> Blurrer:
    """Construct the redaction backend named by ``method``.

    ``"detector"`` runs the standalone ``FaceDetector``; ``"pose"`` redacts from
    the pose result the pipeline already computes (``model_path`` is unused there,
    since it reuses the pose model's keypoints); ``"hybrid"`` uses the pose
    keypoints when a pose is present and falls back to the detector otherwise.
    """
    if method == "pose":
        return PoseFaceBlurrer(style=style)
    if method == "detector":
        return FaceBlurrer(model_path=model_path, style=style)
    if method == "hybrid":
        return HybridFaceBlurrer(style=style, model_path=model_path)
    raise ValueError(f"blur method must be one of {BLUR_METHODS}, got {method!r}")
