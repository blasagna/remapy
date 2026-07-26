package dev.remapy.app

import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import dev.remapy.metrics.Landmarks
import dev.remapy.metrics.PoseFrame

/**
 * The MediaPipe boundary: one detector result -> one [PoseFrame].
 *
 * The Kotlin counterpart of `recording/recorder.py`'s `landmark_rows`, and it exists for the same
 * reason that one is a shared function rather than an inline block: **the full-NaN convention is
 * load-bearing downstream.** `PoseFrames.posePresent` and all of `Quality` key off it — a NaN
 * landmark-0 x implies the whole row is NaN — so a second conversion that filled zeros instead, or
 * wrote NaN coordinates but left visibility at 0.0, would produce frames that read as *tracked
 * while carrying nothing*.
 *
 * This is the only file in the app that knows both MediaPipe's types and the kernel's. Keep it
 * that way: it is what lets `:metrics` be tested on a desktop JDK with no detector present.
 */
object LandmarkRows {

    /**
     * Convert a result to a frame, or [PoseFrame.noPose] when nothing was detected.
     *
     * Only the first pose is read. The pipeline runs with `numPoses = 1` throughout — the metrics
     * describe *a* child, and picking one of several detected people is a question this code is
     * not equipped to answer.
     */
    fun from(result: PoseLandmarkerResult): PoseFrame {
        val normPoses = result.landmarks()
        val worldPoses = result.worldLandmarks()
        if (normPoses.isEmpty() || worldPoses.isEmpty()) return PoseFrame.noPose()

        val norm = normPoses[0]
        val world = worldPoses[0]
        if (norm.size < Landmarks.COUNT || world.size < Landmarks.COUNT) return PoseFrame.noPose()

        val normRow = FloatArray(Landmarks.COUNT * 3)
        val worldRow = FloatArray(Landmarks.COUNT * 3)
        val visibility = FloatArray(Landmarks.COUNT)
        val presence = FloatArray(Landmarks.COUNT)

        for (i in 0 until Landmarks.COUNT) {
            val n = norm[i]
            normRow[i * 3] = n.x()
            normRow[i * 3 + 1] = n.y()
            normRow[i * 3 + 2] = n.z()

            val w = world[i]
            worldRow[i * 3] = w.x()
            worldRow[i * 3 + 1] = w.y()
            worldRow[i * 3 + 2] = w.z()

            // Both are Optional on the Java API. Absent is *not* zero — zero means "detected and
            // untrustworthy", which would silently gate out a perfectly good frame. Treat a
            // missing score as fully trusted and let the coordinates speak for themselves.
            visibility[i] = n.visibility().orElse(1.0f)
            presence[i] = n.presence().orElse(1.0f)
        }

        return PoseFrame(worldRow, normRow, visibility, presence)
    }
}
