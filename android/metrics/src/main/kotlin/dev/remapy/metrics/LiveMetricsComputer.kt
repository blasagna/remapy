package dev.remapy.metrics

import kotlin.math.ceil

/**
 * Live motor metrics over a trailing window, for feedback during a session.
 *
 * Port of `motor_metrics/live.py`. It computes **no metric of its own** — it is a buffer, a
 * window and a dispatch, running the *same* [Hold.holdMetrics] / [Crawl.crawlMetrics] against
 * a rolling buffer instead of an annotator's marks. That is the design's load-bearing
 * property: there is exactly one implementation of the maths, so a live number and an offline
 * number cannot drift apart.
 *
 * **Cost is not the constraint.** Measured on the Python side against the 67 ms frame budget
 * that pose detection already dominates: a full recompute over a 5 s window is ~3 ms, over
 * 30 s ~8 ms. At [RECOMPUTE_EVERY] frames that is a couple of percent of the budget, so nothing
 * here needs an incremental algorithm and the window is simply recomputed whole. The margin got
 * wider when the grid moved to 15 Hz — a window of a given duration holds half the samples and
 * the budget doubled. A phone is slower than the laptop that was measured on, but not by the
 * order of magnitude that would change this conclusion — and if it ever is,
 * `LiveWindowBudgetTest` is where it shows up.
 *
 * **The constraint is that the last [LIVE_LAG] samples of any window are extrapolated.** See
 * [LIVE_LAG].
 *
 * **What is deliberately absent**, carried over verbatim because each omission is a refusal
 * rather than an oversight:
 * - **No movement detector**, hence no SPARC, submovement count or movement duration. They
 *   need a movement's onset and offset, and nothing here is entitled to decide where those
 *   are — the same refusal [Hold] makes about loss-of-posture.
 * - **No duration.** It is the annotator's marks by definition, and there is no annotator live.
 * - **No between-trial symmetry index** and no reciprocity/`phase_offset` for either girdle.
 */
class LiveMetricsComputer(
    val mode: String = HOLD,
    windowS: Double? = null,
    private val up: DoubleArray = Signals.WORLD_UP,
    private val gate: Quality.Gate = Quality.Gate(),
    private val minCoverage: Double = MIN_COVERAGE,
    recomputeEvery: Int = RECOMPUTE_EVERY,
) {

    companion object {
        const val HOLD = "hold"
        const val CRAWL = "crawl"

        /**
         * Samples to read back from the end of a window for an instantaneous value.
         *
         * **Not a tuning choice, and deliberately computed rather than written down**: it is
         * the half-width of the Savitzky-Golay fit, and at exactly this lag the live value
         * equals the offline one. The interior fit needs two samples either side, so the last
         * two of *any* window are fitted from one side only. Measured against the offline
         * whole-signal chain, as a function of how far back the value is read:
         *
         * ```
         * lag 0 (edge)   position RMSE 0.00223 m   velocity RMSE 0.0614 m/s
         * lag 1 ( 67 ms)               0.00134 m                 0.0287 m/s
         * lag 2 (133 ms)               0.000000                  0.000000
         * ```
         *
         * The test signal's velocity sd was 0.0679 m/s, so **the edge-extrapolated derivative's
         * error is 90 % of the signal's own spread** — essentially no information — while
         * reading two samples back reproduces the offline value *exactly*. 133 ms is
         * imperceptible as feedback and buys a number that is the same measurement the offline
         * table reports. Do not "simplify" this to 0.
         *
         * This was a literal `3` until [Derive.FS] moved to 15 Hz, at which point it became a
         * stale value that still compiled, still ran, and silently read twice as far into the
         * past as it needed to. Deriving it is what stops that happening again — and it is why
         * `ConstantsTest`'s half-width assertion is kept even though it now reads as a tautology.
         *
         * That lag governs the **instantaneous** readouts. The aggregates (sway RMS, mean
         * velocity, cadence) average over the whole window, so the two edge samples are 2 of
         * ~75 and dilute away; they are not trimmed, because trimming would only move the
         * edge, not remove it.
         */
        @JvmStatic
        val LIVE_LAG: Int = Derive.windowLength() / 2

        /**
         * Recompute the expensive window metrics every N frames (~3 Hz at a 15 Hz camera).
         *
         * Cheap quality figures are refreshed on **every** frame regardless — they are what
         * tells the operator whether the framing is usable, and they must not lag the video.
         */
        const val RECOMPUTE_EVERY: Int = 5

        /**
         * Below this fraction of trusted frames the readout is blanked rather than shown.
         *
         * A **display** convention, not a validated threshold, and deliberately not a metrics
         * constant: it decides what an operator is shown in the moment, and changing it cannot
         * change any recorded number. A stale figure left on screen while tracking has failed
         * is worse than no figure, because it reads as a measurement of the child.
         */
        const val MIN_COVERAGE: Double = 0.5

        /**
         * Default trailing window per mode, in seconds. A hold needs enough samples for a sway
         * statistic; a crawl needs several pull cycles before the period CV means anything, and
         * at a ~1 Hz cadence that is a longer window.
         */
        val MODE_WINDOW_S: Map<String, Double> = mapOf(HOLD to 5.0, CRAWL to 6.0)

        /**
         * Ring-buffer sizing assumption. The buffer is sized in *frames* but the window is
         * taken in *time*, so this only has to be an upper bound on the camera's rate for the
         * window to hold its full duration; being generous costs a few hundred KB.
         */
        private const val CAPACITY_FPS = 60.0

        /**
         * `(instantaneous, baseline)` trunk lean in degrees over [span].
         *
         * The instantaneous value is read [LIVE_LAG] samples back from the end of the smoothed
         * window, which is where it equals what the offline chain would report for that
         * instant; the baseline is the window's median. Returns `(NaN, NaN)` when the window is
         * too short to smooth or the angle is unusable, rather than throwing.
         */
        fun trunkAngleNow(
            frames: PoseFrames,
            span: Span,
            up: DoubleArray = Signals.WORLD_UP,
        ): Pair<Double, Double> {
            val nan = Double.NaN to Double.NaN
            if (span.nFrames < 2) return nan

            val angles = Signals.trunkFromVertical(frames, span, up)
            if (!angles.all { it.isFinite() }) return nan
            val (_, uniform) = Derive.resampleUniform(frames.timestampsMs(span.start, span.stop), angles)
            if (uniform.isEmpty()) return nan
            val smoothed = Derive.smooth(uniform)
            if (!smoothed.all { it.isFinite() } || smoothed.size <= LIVE_LAG) return nan
            return smoothed[smoothed.size - 1 - LIVE_LAG] to median(smoothed)
        }
    }

    val windowS: Double = windowS ?: (
        MODE_WINDOW_S[mode] ?: throw IllegalArgumentException(
            "mode must be one of ${MODE_WINDOW_S.keys}, got '$mode'."
        )
        )

    private val recomputeEvery = maxOf(1, recomputeEvery)

    /** Sized for the window plus the smoothing tail, so a full window is always available. */
    val window = LiveWindow(ceil(this.windowS * CAPACITY_FPS).toInt() + 2 * LIVE_LAG + 2)

    private var last: LiveMetrics? = null
    private var pushes = 0

    init {
        require(MODE_WINDOW_S.containsKey(mode)) {
            "mode must be one of ${MODE_WINDOW_S.keys}, got '$mode'."
        }
    }

    /**
     * Add one frame and return the current readout.
     *
     * Called once per camera frame, and **always returns something renderable** so a caller has
     * a value every frame. Never throws on a cold, partial or fully-untracked buffer — it
     * returns a blanked readout instead. That discipline matters more live than offline: this
     * runs inside the capture loop, where an exception costs the session rather than one row of
     * a table.
     */
    fun push(timestampMs: Long, frame: PoseFrame): LiveMetrics {
        window.push(timestampMs, frame)
        pushes++

        val span = window.windowSpan(windowS)
        val ok = Quality.landmarksOk(window, Landmarks.TORSO, gate)
        val cov = Quality.coverage(ok, span.start, span.stop)
        val trackedS = trackedSeconds(ok, span)
        val upSource = if (mode == CRAWL) "n/a" else Hold.upSource(up)
        val blank = LiveMetrics.blank(mode, windowS, span.nFrames, cov, trackedS, upSource)

        if (cov < minCoverage) {
            // Blank rather than reuse: the last good value describes a moment that has passed,
            // and on screen it is indistinguishable from a current measurement.
            last = null
            return blank
        }
        val previous = last
        if (previous != null && pushes % recomputeEvery != 0) {
            // Reuse the measurements, but keep the freshly computed quality figures — coverage
            // is what tells the operator the readout is still trustworthy.
            return previous.copy(liveNFrames = span.nFrames, liveCoverage = cov, liveTrackedS = trackedS)
        }

        val computed = compute(span, blank)
        last = computed
        return computed
    }

    /**
     * Dispatch to the offline metric for this mode, degrading to [blank] rather than throwing.
     *
     * The catch-all is the belt to the metrics' braces. [Hold] and [Crawl] are written not to
     * throw on degenerate input and their tests pin that; this is here because inside a capture
     * loop an *unexpected* exception costs the whole session.
     */
    private fun compute(span: Span, blank: LiveMetrics): LiveMetrics {
        if (span.nFrames < 2) return blank
        val out = try {
            if (mode == HOLD) {
                val m = Hold.holdMetrics(window, span, up, gate)
                val (angleNow, baseline) = trunkAngleNow(window, span, up)
                blank.copy(
                    liveSwayRmsM = m.rmsM,
                    liveSwayMlRmsM = m.swayMlRmsM,
                    liveSwayApRmsM = m.swayApRmsM,
                    liveSwayVelocityMps = m.meanVelocityMps,
                    liveTrunkAngleDeg = angleNow,
                    liveTrunkAngleBaselineDeg = baseline,
                    liveTrunkAngleDeltaDeg = angleNow - baseline,
                    liveUpSource = m.upSource,
                )
            } else {
                val m = Crawl.crawlMetrics(window, span, gate)
                blank.copy(
                    liveCadenceCpm = m.cadenceCpm,
                    liveCadenceCpmLeft = m.cadenceCpmLeft,
                    liveCadenceCpmRight = m.cadenceCpmRight,
                    liveNCyclesLeft = m.nCyclesLeft,
                    liveNCyclesRight = m.nCyclesRight,
                    liveCyclePeriodCv = m.cyclePeriodCv,
                    liveLegCadenceCpm = m.legCadenceCpm,
                    liveLegCadenceCpmLeft = m.legCadenceCpmLeft,
                    liveLegCadenceCpmRight = m.legCadenceCpmRight,
                    liveLegNCyclesLeft = m.legNCyclesLeft,
                    liveLegNCyclesRight = m.legNCyclesRight,
                    liveLegCyclePeriodCv = m.legCyclePeriodCv,
                    liveLegAmplitudeSymmetry = m.legAmplitudeSymmetry,
                )
            }
        } catch (_: RuntimeException) {
            return blank
        }
        return out.copy(liveValid = true)
    }

    /**
     * Longest continuously-trusted stretch inside [span], in seconds.
     *
     * A data-quality figure, exactly as offline: it says how much of the window can be measured
     * without bridging a dropout, and is **not** a claim that the child was doing anything for
     * that long.
     */
    private fun trackedSeconds(ok: BooleanArray, span: Span): Double {
        val run = Quality.longestRun(ok, span.start, span.stop)
        if (run.nFrames < 2) return 0.0
        return (window.timestampMs(run.stop - 1) - window.timestampMs(run.start)) / 1000.0
    }
}
