package dev.remapy.metrics

/**
 * Frame-quality gating.
 *
 * Port of `motor_metrics/quality.py`, and load-bearing for the same reason: **MediaPipe does
 * not drop an occluded landmark, it extrapolates one and marks it low-visibility.** A frame
 * where Remy's arm is under his body still yields coordinates, and they are invented.
 * [PoseFrames.posePresent] does not catch this — it is a whole-row NaN check and is correct
 * as-is; the extrapolated frames sail straight through it.
 *
 * Hence: every metric gates on the landmarks it actually reads, and every metric reports its
 * [coverage] next to its numbers. A sway figure computed over 40 % of a window is not a sway
 * figure, and the operator must be able to see that without reading the code.
 */
object Quality {

    /**
     * Thresholds a landmark must clear to be trusted on a given frame.
     *
     * The 0.5 defaults are MediaPipe's own detection/presence default and are a starting
     * point, not a validated value. Whatever you pick, keep it fixed across sessions you
     * intend to compare — and note that a phone's camera is not the laptop webcam these were
     * eyeballed on, so they are worth re-checking against real footage before trusting a
     * cross-device trend.
     */
    data class Gate(val minVisibility: Double = 0.5, val minPresence: Double = 0.5)

    /**
     * Frames with a pose where **every** landmark in [indices] is trusted.
     *
     * NaN visibility/presence (the no-pose rows) compare false and so are excluded
     * automatically; [PoseFrames.posePresent] is ANDed in anyway to keep the intent explicit.
     */
    fun landmarksOk(frames: PoseFrames, indices: IntArray, gate: Gate = Gate()): BooleanArray =
        BooleanArray(frames.size) { i ->
            if (!frames.posePresent(i)) {
                false
            } else {
                indices.all { lm ->
                    frames.visibility(i, lm) >= gate.minVisibility &&
                        frames.presence(i, lm) >= gate.minPresence
                }
            }
        }

    /**
     * Fraction of frames in `[start, stop)` that pass [mask].
     *
     * An empty span returns `0.0`, **not NaN**: coverage exists to be threshold-checked, and
     * NaN compares false against every threshold, so an empty window would silently *pass*
     * the check it was meant to fail.
     */
    fun coverage(mask: BooleanArray, start: Int, stop: Int): Double {
        if (stop <= start) return 0.0
        var passing = 0
        for (i in start until stop) if (mask[i]) passing++
        return passing.toDouble() / (stop - start)
    }

    /**
     * Longest contiguous `true` run of [mask] within `[start, stop)`, as absolute indices.
     *
     * Returns `(start, start)` (length 0) when nothing passes, and the **first** longest run
     * when several tie, matching `np.argmax`.
     *
     * This is what a *held* duration means: an 8-second sit that dropped tracking in the
     * middle is not 8 seconds of measured sitting, and stitching the two halves together
     * would invent the transition between them.
     */
    fun longestRun(mask: BooleanArray, start: Int, stop: Int): Span {
        if (stop <= start) return Span(start, start)
        var bestStart = start
        var bestLength = 0
        var runStart = -1
        for (i in start until stop) {
            if (mask[i]) {
                if (runStart < 0) runStart = i
                val length = i - runStart + 1
                if (length > bestLength) {
                    bestLength = length
                    bestStart = runStart
                }
            } else {
                runStart = -1
            }
        }
        return if (bestLength == 0) Span(start, start) else Span(bestStart, bestStart + bestLength)
    }

    /** Seconds spanned by `[start, stop)`; 0.0 for fewer than two frames. */
    fun spanSeconds(frames: PoseFrames, start: Int, stop: Int): Double {
        if (stop - start < 2) return 0.0
        return (frames.timestampMs(stop - 1) - frames.timestampMs(start)) / 1000.0
    }
}
