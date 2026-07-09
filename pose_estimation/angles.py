"""Joint-angle helpers computed from pose *world* landmarks (metric, in meters).

MediaPipe does not output joint angles, but they are easy to derive: the angle
at a joint is the angle between the two bone vectors that meet there.
"""

import numpy as np
from mediapipe.tasks.python.vision import PoseLandmark as L

# Joints of interest as (a, joint, c) landmark triplets. The angle is measured
# at ``joint``, between segments joint->a and joint->c.
JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow": (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
    "right_elbow": (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
    "left_shoulder": (L.LEFT_ELBOW, L.LEFT_SHOULDER, L.LEFT_HIP),
    "right_shoulder": (L.RIGHT_ELBOW, L.RIGHT_SHOULDER, L.RIGHT_HIP),
    "left_knee": (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
    "right_knee": (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
    "left_hip": (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE),
    "right_hip": (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE),
}


def angle_between(a: np.ndarray, joint: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at ``joint`` between vectors joint->a and joint->c."""
    ba = a - joint
    bc = c - joint
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cos = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _to_xyz(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def joint_angles(world_landmarks) -> dict[str, float]:
    """Compute all :data:`JOINT_TRIPLETS` angles for one pose's world landmarks.

    ``world_landmarks`` is a single pose's list of landmarks
    (``result.pose_world_landmarks[i]``). Returns degrees per joint.
    """
    angles: dict[str, float] = {}
    for name, (i_a, i_joint, i_c) in JOINT_TRIPLETS.items():
        angles[name] = angle_between(
            _to_xyz(world_landmarks[i_a]),
            _to_xyz(world_landmarks[i_joint]),
            _to_xyz(world_landmarks[i_c]),
        )
    return angles
