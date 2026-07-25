package dev.remapy.metrics

/**
 * MediaPipe pose landmark indices, and the groups a metric might gate on.
 *
 * Port of the `PoseLandmark` constants that `motor_metrics/quality.py`,
 * `signals.py`, `crawl.py` and `pose_estimation/angles.py` import. On the Python side that
 * import drags the whole MediaPipe wheel into modules that make no framework call — the
 * enum is used purely as integers. Here they are integers, so the kernel has no dependency
 * on the pose library at all and the app module can swap detectors without touching the maths.
 *
 * `ConstantsTest` checks every group against the exported `PoseLandmark` values, so a
 * transcription slip fails the build rather than quietly measuring the wrong joint.
 */
object Landmarks {
    const val COUNT: Int = 33

    const val NOSE = 0
    const val LEFT_SHOULDER = 11
    const val RIGHT_SHOULDER = 12
    const val LEFT_ELBOW = 13
    const val RIGHT_ELBOW = 14
    const val LEFT_WRIST = 15
    const val RIGHT_WRIST = 16
    const val LEFT_HIP = 23
    const val RIGHT_HIP = 24
    const val LEFT_KNEE = 25
    const val RIGHT_KNEE = 26
    const val LEFT_ANKLE = 27
    const val RIGHT_ANKLE = 28

    /**
     * Groups a metric gates on. **Gate on what you read, not on all 33** — requiring the
     * ankles for a sitting-sway metric would throw away good trials.
     */
    val TORSO = intArrayOf(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    val ARMS = intArrayOf(LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
    val LEGS = intArrayOf(LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)
    val WRISTS = intArrayOf(LEFT_WRIST, RIGHT_WRIST)
    val KNEES = intArrayOf(LEFT_KNEE, RIGHT_KNEE)

    /** The face landmarks a pose-derived head box is built from (`face_blur/pose_blur.py`). */
    val FACE = IntArray(11) { it }

    /**
     * The skeleton edge list — 35 pairs, in MediaPipe's own order.
     *
     * Data rather than drawing logic, which is why it lives here beside the indices and not in the
     * overlay: `recording/recorder.py` persists this same list into `meta/pose_connections`
     * precisely so a consumer can draw a pose without pulling in MediaPipe for a 35-pair constant,
     * and `pose_estimation/draw.py` takes it as a parameter for the same reason.
     *
     * `ConstantsTest` pins it against the exported Python list, so a mistyped pair fails the build
     * rather than drawing a bone between the wrong two joints.
     */
    val POSE_CONNECTIONS: List<Pair<Int, Int>> = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 7, 0 to 4, 4 to 5, 5 to 6, 6 to 8, 9 to 10,
        11 to 12, 11 to 13, 13 to 15, 15 to 17, 15 to 19, 15 to 21, 17 to 19,
        12 to 14, 14 to 16, 16 to 18, 16 to 20, 16 to 22, 18 to 20,
        11 to 23, 12 to 24, 23 to 24,
        23 to 25, 24 to 26, 25 to 27, 26 to 28, 27 to 29, 28 to 30, 29 to 31, 30 to 32,
        27 to 31, 28 to 32,
    )

    /**
     * Joints of interest as `(a, joint, c)` triplets. The angle is measured at `joint`,
     * between the segments joint->a and joint->c.
     */
    val JOINT_TRIPLETS: Map<String, Triple<Int, Int, Int>> = linkedMapOf(
        "left_elbow" to Triple(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        "right_elbow" to Triple(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        "left_shoulder" to Triple(LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
        "right_shoulder" to Triple(RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
        "left_knee" to Triple(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        "right_knee" to Triple(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        "left_hip" to Triple(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
        "right_hip" to Triple(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    )
}
