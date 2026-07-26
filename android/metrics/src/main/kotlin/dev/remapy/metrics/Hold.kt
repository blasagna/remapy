package dev.remapy.metrics

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Static-posture holds: sitting arms-free (GMFM dim B) and supported standing (dim D).
 *
 * Port of `motor_metrics/hold.py`. One function serves both — a supported stand and an
 * unsupported sit are the same measurement problem, how long and how steadily, and the label
 * carries the difference.
 *
 * **What replaces the ordinal score.** The GMFM sitting items are a ladder of duration
 * thresholds (3 s, 5 s, 20 s, 60 s) — a continuous variable someone binned. So the headline
 * is the uncensored duration, and underneath it the **sway** of the trunk over the pelvis:
 * the sub-clinical signal that should move while the bin number sits still.
 *
 * **Two traps this closes**, both restated here because they are easy to lose in a port:
 * [HoldMetrics.pathLengthM] grows with trial length, so a *worse* 20-second hold beats a
 * *better* 8-second one on it — [HoldMetrics.meanVelocityMps] is the duration-free form. And
 * [HoldMetrics.ellipseAreaM2] reads ~0 for one-axis rocking, because a line encloses no area,
 * so it must be read next to the ML/AP split and never alone.
 *
 * Nothing here infers loss-of-posture from a trunk-angle threshold: picking that threshold
 * would be inventing a clinical criterion and burying it in a constant. Offline that job
 * belongs to the annotator; live there is no annotator, which is why the live path windows
 * on a clock instead and reports no `duration_s` at all.
 */
object Hold {

    /** Chi-square with 2 dof at p=0.95 — the standard posturography 95 % sway ellipse. */
    private const val CHI2_95_2DOF = 5.991

    /**
     * Metrics for one hold segment. Distances in meters, angles in degrees, times in seconds.
     *
     * Every measurement field is NaN rather than an exception for segments that are empty,
     * untracked, or shorter than the smoothing window. That is not defensive coding — a
     * mis-marked two-frame annotation is a normal thing to find in a session, and inside a
     * capture loop an unexpected throw costs the whole session rather than one table row.
     */
    data class HoldMetrics(
        /** The annotated trial: a human's call on when the hold began and ended. */
        val durationS: Double,
        /** Longest continuously-trusted run; a **data-quality** figure, not a hold. */
        val trackedS: Double,
        /** Fraction of the trial with trusted torso landmarks. */
        val coverage: Double,
        val nFrames: Int,
        /** Confounded with duration — compare only at equal `windowS`. */
        val pathLengthM: Double,
        /** `pathLength / trackedS`; the duration-free form. */
        val meanVelocityMps: Double,
        /** ~0 for one-axis rocking. Read beside the ML/AP split, never alone. */
        val ellipseAreaM2: Double,
        /** Radial RMS excursion from the mean posture. */
        val rmsM: Double,
        /** Medio-lateral: in the image plane, measured well. */
        val swayMlRmsM: Double,
        /** Antero-posterior: inferred depth, markedly noisier. */
        val swayApRmsM: Double,
        /** Unsigned lean from `up`; 0 = upright. */
        val trunkAngleMeanDeg: Double,
        val trunkAngleSdDeg: Double,
        val trunkAngleRangeDeg: Double,
        /** QC DIAGNOSTIC ONLY — see [handsLowFrac]. */
        val handsLowFrac: Double,
        /** Which vertical was used; `world_y` assumes a level camera. */
        val upSource: String,
    )

    /**
     * Metrics for one `sit_hold` / `stand_hold` segment.
     *
     * Sway is computed over the longest continuously-tracked run inside the trial, **never
     * across a tracking gap** — bridging one would invent the movement that happened inside
     * it. Pass [windowS] to truncate that run to a common prefix so trials of unequal length
     * can be compared on [HoldMetrics.pathLengthM].
     */
    fun holdMetrics(
        frames: PoseFrames,
        seg: Span,
        up: DoubleArray = Signals.WORLD_UP,
        gate: Quality.Gate = Quality.Gate(),
        windowS: Double? = null,
    ): HoldMetrics {
        val ok = Quality.landmarksOk(frames, Landmarks.TORSO, gate)
        val cov = Quality.coverage(ok, seg.start, seg.stop)
        val durationS = Quality.spanSeconds(frames, seg.start, seg.stop)

        var run = Quality.longestRun(ok, seg.start, seg.stop)
        if (windowS != null && run.stop > run.start) {
            run = Span(run.start, clipToWindow(frames, run, windowS))
        }
        val trackedS = Quality.spanSeconds(frames, run.start, run.stop)

        val sway = sway(frames, run, up, trackedS)
        val angle = trunkAngleStats(frames, run, up)

        return HoldMetrics(
            durationS = durationS,
            trackedS = trackedS,
            coverage = cov,
            nFrames = seg.nFrames,
            pathLengthM = sway.pathLengthM,
            meanVelocityMps = sway.meanVelocityMps,
            ellipseAreaM2 = sway.ellipseAreaM2,
            rmsM = sway.rmsM,
            swayMlRmsM = sway.swayMlRmsM,
            swayApRmsM = sway.swayApRmsM,
            trunkAngleMeanDeg = angle[0],
            trunkAngleSdDeg = angle[1],
            trunkAngleRangeDeg = angle[2],
            handsLowFrac = handsLowFrac(frames, run, up, gate),
            upSource = upSource(up),
        )
    }

    /** `world_y` when [up] is the default vertical, `custom` otherwise. */
    fun upSource(up: DoubleArray): String {
        // np.allclose's defaults: rtol 1e-5, atol 1e-8.
        val same = up.indices.all {
            abs(up[it] - Signals.WORLD_UP[it]) <= 1e-8 + 1e-5 * abs(Signals.WORLD_UP[it])
        }
        return if (same) "world_y" else "custom"
    }

    /**
     * 95 % confidence ellipse area of a 2D cloud, in m^2.
     *
     * `chi2(2, 0.95) * pi * sqrt(l1 * l2)` over the covariance eigenvalues — the standard
     * posturography figure. A perfectly collinear cloud has a zero eigenvalue and so zero
     * area, which is correct: it encloses no region.
     */
    fun swayEllipseArea(points: Matrix): Double {
        if (points.rows < 3 || !points.allFinite()) return Double.NaN
        val c = points.centred()
        val n = points.rows
        // np.cov's default normalisation is N - 1, not N.
        var sxx = 0.0
        var sxy = 0.0
        var syy = 0.0
        for (i in 0 until n) {
            sxx += c[i, 0] * c[i, 0]
            sxy += c[i, 0] * c[i, 1]
            syy += c[i, 1] * c[i, 1]
        }
        val denom = (n - 1).toDouble()
        val a = sxx / denom
        val b = sxy / denom
        val d = syy / denom
        // Symmetric 2x2 eigenvalues, ascending, as np.linalg.eigvalsh returns them.
        val mean = (a + d) / 2.0
        val radius = sqrt(((a - d) / 2.0) * ((a - d) / 2.0) + b * b)
        // Clamp: a degenerate cloud can produce a tiny negative eigenvalue from rounding.
        val low = (mean - radius).coerceAtLeast(0.0)
        val high = (mean + radius).coerceAtLeast(0.0)
        return CHI2_95_2DOF * PI * sqrt(low * high)
    }

    private class Sway(
        val pathLengthM: Double = Double.NaN,
        val meanVelocityMps: Double = Double.NaN,
        val ellipseAreaM2: Double = Double.NaN,
        val rmsM: Double = Double.NaN,
        val swayMlRmsM: Double = Double.NaN,
        val swayApRmsM: Double = Double.NaN,
    )

    /** Sway of the trunk-over-pelvis path, on the pinned filter chain. */
    private fun sway(frames: PoseFrames, run: Span, up: DoubleArray, trackedS: Double): Sway {
        if (run.nFrames < 2) return Sway()

        val horizontal = Signals.projectHorizontal(Signals.trunkVectors(frames, run), up)
        val (_, uniform) = Derive.resampleUniform(frames.timestampsMs(run.start, run.stop), horizontal)
        if (uniform.rows == 0) return Sway()
        val smoothed = Derive.smooth(uniform) // NaN when the run is shorter than the window
        if (!smoothed.allFinite()) return Sway()

        val centred = smoothed.centred()
        val length = pathLength(smoothed)
        var radial = 0.0
        var ml = 0.0
        var ap = 0.0
        for (i in 0 until centred.rows) {
            radial += centred[i, 0] * centred[i, 0] + centred[i, 1] * centred[i, 1]
            ml += centred[i, 0] * centred[i, 0]
            ap += centred[i, 1] * centred[i, 1]
        }
        val n = centred.rows
        return Sway(
            pathLengthM = length,
            meanVelocityMps = if (trackedS > 0) length / trackedS else Double.NaN,
            ellipseAreaM2 = swayEllipseArea(smoothed),
            rmsM = sqrt(radial / n),
            swayMlRmsM = sqrt(ml / n),
            swayApRmsM = sqrt(ap / n),
        )
    }

    /** Returns `[mean, sd, range]` of the trunk lean over the run, NaN-filtered. */
    private fun trunkAngleStats(frames: PoseFrames, run: Span, up: DoubleArray): DoubleArray {
        val nan = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN)
        if (run.nFrames == 0) return nan
        val finite = Signals.trunkFromVertical(frames, run, up).filter { it.isFinite() }.toDoubleArray()
        if (finite.isEmpty()) return nan
        return doubleArrayOf(finite.average(), std(finite), finite.max() - finite.min())
    }

    /**
     * Fraction of frames with either wrist below the pelvis, along `up`.
     *
     * **Not an arms-free detector.** Whether a hand bears weight is a force question and there
     * is no force sensor here: a hand can rest low without bearing weight, and can bear weight
     * without being low. It is a hint that an `arms=free` label may be wrong, nothing more.
     *
     * Gated on the wrists separately from the torso, so a trial stays measurable when the
     * hands are out of frame — this returns NaN there instead of poisoning the whole row.
     */
    private fun handsLowFrac(
        frames: PoseFrames,
        run: Span,
        up: DoubleArray,
        gate: Quality.Gate,
    ): Double {
        if (run.stop <= run.start) return Double.NaN
        val wristsOk = Quality.landmarksOk(frames, Landmarks.WRISTS, gate)
        var considered = 0
        var low = 0
        val upUnit = Signals.unit(up)
        for (i in run.start until run.stop) {
            if (!wristsOk[i]) continue
            considered++
            val row = frames.worldRow(i)
            val anyLow = Landmarks.WRISTS.any { wrist ->
                val p = row.point(wrist)
                // Height above the pelvis (the world frame's origin) along `up`.
                p[0] * upUnit[0] + p[1] * upUnit[1] + p[2] * upUnit[2] < 0.0
            }
            if (anyLow) low++
        }
        if (considered == 0) return Double.NaN
        return low.toDouble() / considered
    }

    /** Shrink a run to the first [windowS] seconds of it. */
    private fun clipToWindow(frames: PoseFrames, run: Span, windowS: Double): Int {
        val cutoff = frames.timestampMs(run.start) + windowS * 1000.0
        // searchsorted(..., side="right"): first index whose timestamp exceeds the cutoff.
        var i = run.start
        while (i < run.stop && frames.timestampMs(i) <= cutoff) i++
        return i
    }
}
