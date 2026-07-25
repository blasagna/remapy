package dev.remapy.metrics

/**
 * Prone belly-crawl (GMFM dim C): cadence, cycle variability, per-limb symmetry.
 *
 * Port of `motor_metrics/crawl.py`. Two things from that module's docstring govern what this
 * file is allowed to claim:
 *
 * **The GMFM item asks for distance — "creeps forward 1.8 m" — and this cannot measure it.**
 * The world frame is hip-centered, so the pelvis cannot travel in it by construction, and
 * there is no second camera, no depth sensor and no IMU to recover it from. What remains is
 * [CrawlMetrics.speedNormPerS], in **image widths per second, never metres**, comparable only
 * within a session at fixed framing. Converting it honestly needs a scale reference this
 * pipeline does not have — a floor fiducial or a second camera, not cleverer maths.
 *
 * **What it measures instead is the pattern, which is the better metric anyway** and the one
 * likelier to move before the ordinal score does. Both limb girdles are measured — arms
 * (wrists, the unprefixed fields) *and* legs (knees, the `leg*` fields) — because the
 * developmental signal is not always in the arms: Remy's arms often move together while he
 * drives with the legs and favours one repeatedly, so reading only the wrists would miss
 * exactly the asymmetry that is changing. Each girdle gates independently, since legs leave
 * frame in prone far more than arms and one occluded knee must not cost the arm cadence.
 *
 * **Reciprocity (`phase_offset`) is deliberately not ported.** It needs `scipy.signal.hilbert`,
 * and the live path excludes it anyway: Hilbert's edge effects peak at exactly a short
 * trailing window's edges, so alternating-vs-together is an offline number for both girdles.
 * The fields are kept in [CrawlMetrics] as NaN rather than dropped, so the shape still matches
 * the Python dataclass and adding them later is a fill-in rather than a signature change.
 * Note what this costs: the "bunny haul vs mature crawl" axis is not available on the phone.
 */
object Crawl {

    /** Each girdle needs the torso for the body axis, plus its own limb pair. */
    val ARM_LANDMARKS = Landmarks.TORSO + Landmarks.WRISTS
    val LEG_LANDMARKS = Landmarks.TORSO + Landmarks.KNEES

    /**
     * A peak counts as a pull cycle if it stands this far above the surrounding trough, as a
     * fraction of the signal's full range.
     */
    const val CYCLE_PROMINENCE_FRAC: Double = 0.20

    /**
     * ...but a *relative* gate alone is scale-invariant, and that is a trap.
     *
     * Normalized against its own range, pure landmark jitter on a motionless arm looks exactly
     * like a crawl, and a still child reports a confident, entirely fictional cadence —
     * measured at **57 cycles from pure noise**. So a signal must also clear an absolute
     * excursion in meters before it is treated as movement at all. A real belly-crawl arm pull
     * travels 10–20 cm along the body axis and wrist landmark noise is on the order of 1 cm,
     * so 2 cm sits well clear of both.
     *
     * A convention, not a validated constant. Check it against real trials, and keep it fixed
     * across sessions you intend to compare. `CrawlTest` pins that removing it resurrects the
     * fictional cadence.
     */
    const val MIN_CYCLE_EXCURSION_M: Double = 0.02

    enum class Side { LEFT, RIGHT }

    enum class Marker(val left: Int, val right: Int) {
        WRIST(Landmarks.LEFT_WRIST, Landmarks.RIGHT_WRIST),
        KNEE(Landmarks.LEFT_KNEE, Landmarks.RIGHT_KNEE),
        ;

        fun index(side: Side): Int = if (side == Side.LEFT) left else right
    }

    /**
     * One belly-crawl trial.
     *
     * **Unprefixed fields are the arms** (wrists — their long-standing meaning); the `leg*`
     * fields are the same measurements off the knees. An arms-together / legs-favouring-one-side
     * pattern shows as a nonzero [legAmplitudeSymmetry] next to even arm cycle counts.
     */
    data class CrawlMetrics(
        val durationS: Double,
        /** Arms (torso + wrists); the trial's headline quality figure. */
        val trackedS: Double,
        val coverage: Double,
        val nFrames: Int,
        /** Pooled across sides. */
        val cadenceCpm: Double,
        val cadenceCpmLeft: Double,
        val cadenceCpmRight: Double,
        val nCyclesLeft: Int,
        val nCyclesRight: Int,
        val cyclePeriodSdS: Double,
        /** sd/mean; dimensionless, so comparable across cadences. */
        val cyclePeriodCv: Double,
        /** Always NaN in this port — needs Hilbert. See the class docstring. */
        val phaseOffset: Double,
        /** Always NaN in this port — needs Hilbert. */
        val phaseOffsetCircularSd: Double,
        /** 0 = both arms working equally; sign gives the side. */
        val amplitudeSymmetry: Double,
        /** IMAGE WIDTHS per second. Not metres. Within-session only. */
        val speedNormPerS: Double,
        /** Legs get their own quality figures — a leg dropout must not read as an arm one. */
        val legCoverage: Double,
        val legTrackedS: Double,
        val legCadenceCpm: Double,
        val legCadenceCpmLeft: Double,
        val legCadenceCpmRight: Double,
        val legNCyclesLeft: Int,
        val legNCyclesRight: Int,
        val legCyclePeriodSdS: Double,
        val legCyclePeriodCv: Double,
        /** Always NaN in this port — needs Hilbert. */
        val legPhaseOffset: Double,
        /** Always NaN in this port — needs Hilbert. */
        val legPhaseOffsetCircularSd: Double,
        /** "Favours one leg": 0 = even, sign gives the side. Remy's signal. */
        val legAmplitudeSymmetry: Double,
    )

    /**
     * The limb's position along the body's long axis, per frame, in meters.
     *
     * `dot(limb - midHip, trunkUnit)`: how far the wrist (or knee) is toward the head, measured
     * relative to the pelvis and to the body's own axis. **In prone there is no useful
     * vertical**, so the trunk vector — not gravity — is the axis a crawl cycle oscillates
     * along. This is why crawl is the camera-robust mode: it reads no `up` at all.
     *
     * NaN where the pose or the trunk axis is unusable.
     */
    fun limbSignal(frames: PoseFrames, span: Span, side: Side, marker: Marker): DoubleArray {
        if (span.nFrames <= 0) return DoubleArray(0)
        val index = marker.index(side)
        return DoubleArray(span.nFrames) { i ->
            val row = frames.worldRow(span.start + i)
            val trunk = Signals.trunkVector(row)
            val length = Signals.norm(trunk)
            if (!(length > 1e-9)) {
                Double.NaN
            } else {
                val hips = Signals.mid(row, Landmarks.LEFT_HIP, Landmarks.RIGHT_HIP)
                val limb = row.point(index)
                var acc = 0.0
                for (c in 0 until 3) acc += (limb[c] - hips[c]) * (trunk[c] / length)
                acc
            }
        }
    }

    /**
     * Indices of pull-cycle peaks in a limb signal (which is in meters).
     *
     * Two gates, and **both are needed**. The signal must travel at least [minExcursion]
     * overall — otherwise it is a still arm and there are no cycles to find, however peaky its
     * jitter looks — and each peak must then clear [CYCLE_PROMINENCE_FRAC] of the signal's
     * range. The absolute gate is the one that stops noise being normalized up into a
     * plausible cadence.
     */
    fun cycles(signal: DoubleArray, minExcursion: Double = MIN_CYCLE_EXCURSION_M): IntArray {
        if (signal.size < 3 || !signal.all { it.isFinite() }) return IntArray(0)
        val span = signal.max() - signal.min()
        if (span < minExcursion) return IntArray(0)
        return PeakFind.findPeaks(signal, CYCLE_PROMINENCE_FRAC * span)
    }

    /**
     * `2 * (L - R) / (L + R)` on the medians. 0 = symmetric, sign gives the side.
     *
     * Port of `motor_metrics/transition.py`'s `symmetry_index`, which crawl uses on the two
     * limbs' excursion ranges. NaN when either side is empty or the medians sum to zero (no
     * movement to compare).
     *
     * Note this is the *within-trial* left/right comparison. The between-trial one — grouped
     * by a label's `side=` — is an offline concept and has no live counterpart; do not conflate
     * them just because they share a formula.
     */
    fun symmetryIndex(left: DoubleArray, right: DoubleArray): Double {
        val l = left.filter { it.isFinite() }.toDoubleArray()
        val r = right.filter { it.isFinite() }.toDoubleArray()
        if (l.isEmpty() || r.isEmpty()) return Double.NaN
        val lMed = median(l)
        val rMed = median(r)
        val total = lMed + rMed
        if (total == 0.0) return Double.NaN
        return 2.0 * (lMed - rMed) / total
    }

    /**
     * Metrics for one `crawl` segment, for **both** limb girdles.
     *
     * Each girdle is computed over the longest run where the torso and both of its limbs are
     * trusted, and gated independently. Returns NaN fields rather than throwing for trials
     * that are empty, untracked, or too short.
     */
    fun crawlMetrics(
        frames: PoseFrames,
        seg: Span,
        gate: Quality.Gate = Quality.Gate(),
    ): CrawlMetrics {
        val arm = girdle(frames, seg, ARM_LANDMARKS, Marker.WRIST, gate)
        val leg = girdle(frames, seg, LEG_LANDMARKS, Marker.KNEE, gate)

        return CrawlMetrics(
            durationS = Quality.spanSeconds(frames, seg.start, seg.stop),
            trackedS = arm.trackedS,
            coverage = arm.coverage,
            nFrames = seg.nFrames,
            cadenceCpm = arm.cadenceCpm,
            cadenceCpmLeft = arm.cadenceCpmLeft,
            cadenceCpmRight = arm.cadenceCpmRight,
            nCyclesLeft = arm.nCyclesLeft,
            nCyclesRight = arm.nCyclesRight,
            cyclePeriodSdS = arm.cyclePeriodSdS,
            cyclePeriodCv = arm.cyclePeriodCv,
            phaseOffset = Double.NaN,
            phaseOffsetCircularSd = Double.NaN,
            amplitudeSymmetry = arm.amplitudeSymmetry,
            speedNormPerS = speedNorm(frames, arm.run, arm.trackedS),
            legCoverage = leg.coverage,
            legTrackedS = leg.trackedS,
            legCadenceCpm = leg.cadenceCpm,
            legCadenceCpmLeft = leg.cadenceCpmLeft,
            legCadenceCpmRight = leg.cadenceCpmRight,
            legNCyclesLeft = leg.nCyclesLeft,
            legNCyclesRight = leg.nCyclesRight,
            legCyclePeriodSdS = leg.cyclePeriodSdS,
            legCyclePeriodCv = leg.cyclePeriodCv,
            legPhaseOffset = Double.NaN,
            legPhaseOffsetCircularSd = Double.NaN,
            legAmplitudeSymmetry = leg.amplitudeSymmetry,
        )
    }

    private class Girdle(
        val coverage: Double,
        val trackedS: Double,
        val run: Span,
        val cadenceCpm: Double,
        val cadenceCpmLeft: Double,
        val cadenceCpmRight: Double,
        val nCyclesLeft: Int,
        val nCyclesRight: Int,
        val cyclePeriodSdS: Double,
        val cyclePeriodCv: Double,
        val amplitudeSymmetry: Double,
    )

    private fun girdle(
        frames: PoseFrames,
        seg: Span,
        landmarks: IntArray,
        marker: Marker,
        gate: Quality.Gate,
    ): Girdle {
        val ok = Quality.landmarksOk(frames, landmarks, gate)
        val cov = Quality.coverage(ok, seg.start, seg.stop)
        val run = Quality.longestRun(ok, seg.start, seg.stop)
        val trackedS = Quality.spanSeconds(frames, run.start, run.stop)

        val left = prepared(frames, run, Side.LEFT, marker)
        val right = prepared(frames, run, Side.RIGHT, marker)
        val peaksL = cycles(left)
        val peaksR = cycles(right)
        val periods = periodStats(peaksL, peaksR)

        return Girdle(
            coverage = cov,
            trackedS = trackedS,
            run = run,
            cadenceCpm = cadence(peaksL.size + peaksR.size, 2 * trackedS),
            cadenceCpmLeft = cadence(peaksL.size, trackedS),
            cadenceCpmRight = cadence(peaksR.size, trackedS),
            nCyclesLeft = peaksL.size,
            nCyclesRight = peaksR.size,
            cyclePeriodSdS = periods[0],
            cyclePeriodCv = periods[1],
            amplitudeSymmetry = amplitudeSymmetry(left, right),
        )
    }

    /** A limb signal on the uniform grid and through the pinned smoothing chain. */
    private fun prepared(frames: PoseFrames, run: Span, side: Side, marker: Marker): DoubleArray {
        val raw = limbSignal(frames, run, side, marker)
        if (raw.size < 2 || !raw.all { it.isFinite() }) return DoubleArray(0)
        val (_, uniform) = Derive.resampleUniform(frames.timestampsMs(run.start, run.stop), raw)
        if (uniform.isEmpty()) return DoubleArray(0)
        val out = Derive.smooth(uniform)
        return if (out.all { it.isFinite() }) out else DoubleArray(0)
    }

    private fun cadence(nCycles: Int, seconds: Double): Double {
        if (seconds <= 0 || nCycles == 0) return Double.NaN
        return nCycles.toDouble() / seconds * 60.0
    }

    /** Returns `[sd, cv]` of the cycle period, pooled across sides. */
    private fun periodStats(peaksL: IntArray, peaksR: IntArray): DoubleArray {
        val periods = ArrayList<Double>()
        for (peaks in listOf(peaksL, peaksR)) {
            for (i in 1 until peaks.size) periods.add((peaks[i] - peaks[i - 1]) / Derive.FS)
        }
        if (periods.size < 2) return doubleArrayOf(Double.NaN, Double.NaN)
        val values = periods.toDoubleArray()
        val mean = values.average()
        val sd = std(values)
        return doubleArrayOf(sd, if (mean > 0) sd / mean else Double.NaN)
    }

    /** Whether one limb travels further than the other, on their excursion ranges. */
    private fun amplitudeSymmetry(left: DoubleArray, right: DoubleArray): Double {
        if (left.isEmpty() || right.isEmpty()) return Double.NaN
        return symmetryIndex(doubleArrayOf(ptp(left)), doubleArrayOf(ptp(right)))
    }

    /** Pelvis travel in IMAGE FRACTIONS per second. See the class docstring. */
    private fun speedNorm(frames: PoseFrames, run: Span, trackedS: Double): Double {
        if (run.nFrames < 2 || trackedS <= 0) return Double.NaN
        val pelvis = Signals.comNorm(frames, run)
        if (!pelvis.allFinite()) return Double.NaN
        val (_, uniform) = Derive.resampleUniform(frames.timestampsMs(run.start, run.stop), pelvis)
        if (uniform.rows == 0) return Double.NaN
        val smoothed = Derive.smooth(uniform)
        if (!smoothed.allFinite()) return Double.NaN
        return pathLength(smoothed) / trackedS
    }
}
