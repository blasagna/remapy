package dev.remapy.metrics

import kotlin.math.PI
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The measurement that justifies [LiveMetricsComputer.LIVE_LAG], reproduced in Kotlin.
 *
 * This is the test most likely to catch a subtly wrong Savitzky-Golay edge, and the one whose
 * failure would matter most: it is the reason a live instantaneous readout can be presented as
 * the *same measurement* the offline table reports, rather than a live approximation of it.
 *
 * Everything here follows from one fact — the interior Savitzky-Golay fit needs
 * `windowLength() / 2` samples either side, so that many at the end of **any** window are
 * fitted from one side only. Reading that many samples back steps off the extrapolated tail
 * entirely. Nothing here hardcodes the number; it follows [LiveMetricsComputer.LIVE_LAG].
 */
class LiveLagIdentityTest {

    private val fs = Derive.FS
    private val lag = LiveMetricsComputer.LIVE_LAG

    /** A 5 s trailing window, in frames — the hold mode's window, derived rather than counted. */
    private val WINDOW_FRAMES = (5.0 * fs).toInt()

    /** Scales the unit-ish hash noise to a ~0.004 m sd, matching the Python measurement. */
    private val NOISE_SD_SCALE = 0.004 / (1.0 / sqrt(12.0))

    /**
     * A sway-like signal: slow fundamental plus **broadband** landmark noise.
     *
     * The noise is broadband deliberately, and generated from a hash rather than an RNG so both
     * languages can see the same character without sharing a generator. An earlier version used
     * a single 5 Hz tone, and that was a poor stand-in for a specific reason: a Savitzky-Golay
     * edge error is made almost entirely of the high-frequency content the fit is extrapolating,
     * so with one tone the measured error is a function of where that tone happens to sit
     * relative to the window rather than of the edge being unsafe to read. It did not bite at
     * 30 Hz / 0.233 s (ratio 1.40). At 15 Hz / 0.333 s the wider window attenuates 5 Hz far
     * enough that the ratio fell to 0.38 — and it would have fallen to 0.18 at the 7-sample
     * alternative — while the identity at [LiveMetricsComputer.LIVE_LAG] stayed exactly zero
     * throughout. Real landmark noise is broadband; `tests/test_live.py` uses
     * `rng.normal(0, 0.004)` for this same measurement, and this reproduces that.
     */
    private fun signal(n: Int): DoubleArray = DoubleArray(n) {
        val t = it / fs
        0.03 * sin(2 * PI * 0.4 * t) + NOISE_SD_SCALE * whiteNoise(it)
    }

    /**
     * Deterministic white noise in `[-0.5, 0.5)`. A hash, not a PRNG: it needs to be
     * reproducible and flat across the band, and it does not need to be good.
     */
    private fun whiteNoise(i: Int): Double {
        var h = i * 374761393 + 668265263
        h = h xor (h ushr 13)
        h *= 1274126177
        h = h xor (h ushr 16)
        return (h and 0xFFFF) / 65536.0 - 0.5
    }

    /**
     * **The identity.** A trailing window's value at `-(LIVE_LAG + 1)` equals the offline
     * whole-signal value at that same sample — not approximately, but to floating-point noise.
     */
    @Test
    fun `a trailing window reproduces the offline value exactly at LIVE_LAG`() {
        val n = 400
        val x = signal(n)
        val offline = Derive.smooth(x)

        for (end in 200..n step 17) {
            val windowStart = end - WINDOW_FRAMES
            val live = Derive.smooth(x.sliceArray(windowStart until end))
            val liveValue = live[live.size - 1 - lag]
            val offlineValue = offline[end - 1 - lag]
            assertClose(offlineValue, liveValue, 1e-12, "position at end=$end")
        }
    }

    /** ...and the same for the derivative, which is what the velocity readout uses. */
    @Test
    fun `the trailing-window derivative is exact at LIVE_LAG too`() {
        val n = 400
        val x = signal(n)
        val offline = Derive.smooth(x, deriv = 1)

        for (end in 200..n step 17) {
            val live = Derive.smooth(x.sliceArray(end - WINDOW_FRAMES until end), deriv = 1)
            assertClose(offline[end - 1 - lag], live[live.size - 1 - lag], 1e-12, "velocity at end=$end")
        }
    }

    /**
     * **The other half, and the reason nobody may "simplify" the lag to zero.**
     *
     * At the window edge the Savitzky-Golay fit is one-sided and the derivative it extrapolates
     * is essentially all error — the Python docstring measures RMSE 0.0614 m/s against a signal
     * whose own velocity sd was 0.0679 m/s. This asserts the shape of that result: the edge
     * error is a large fraction of the signal's own variation, while the error at
     * [LiveMetricsComputer.LIVE_LAG] is zero.
     */
    @Test
    fun `the edge-extrapolated derivative is essentially all error`() {
        val n = 600
        val x = signal(n)
        val offline = Derive.smooth(x, deriv = 1)

        val edgeErrors = ArrayList<Double>()
        val laggedErrors = ArrayList<Double>()
        for (end in 200..n step 3) {
            val live = Derive.smooth(x.sliceArray(end - WINDOW_FRAMES until end), deriv = 1)
            edgeErrors.add(live[live.size - 1] - offline[end - 1])
            laggedErrors.add(live[live.size - 1 - lag] - offline[end - 1 - lag])
        }

        val signalSd = std(offline.sliceArray(100 until n - 100))
        val edgeRmse = rmse(edgeErrors)
        val laggedRmse = rmse(laggedErrors)

        assertTrue(laggedRmse < 1e-12, "lag-$lag error must be zero, was $laggedRmse")
        assertTrue(
            edgeRmse > 0.5 * signalSd,
            "edge RMSE $edgeRmse should rival the signal's own sd $signalSd — if it no longer " +
                "does, re-derive LIVE_LAG rather than trusting the edge",
        )
    }

    /** The lag is the fit's half-width. Stated here so the two can never be tuned apart. */
    @Test
    fun `reading fewer than LIVE_LAG samples back is not exact`() {
        val n = 400
        val x = signal(n)
        val offline = Derive.smooth(x, deriv = 1)
        for (shorterLag in 0 until lag) {
            var worst = 0.0
            for (end in 200..n step 17) {
                val live = Derive.smooth(x.sliceArray(end - WINDOW_FRAMES until end), deriv = 1)
                val delta = kotlin.math.abs(
                    live[live.size - 1 - shorterLag] - offline[end - 1 - shorterLag]
                )
                if (delta > worst) worst = delta
            }
            assertTrue(worst > 1e-9, "lag $shorterLag unexpectedly exact (worst $worst)")
        }
    }

    /**
     * The whole chain, against Python: `trunkAngleNow` over a live buffer at sampled frames.
     *
     * The tests above pin the property; this pins that *this port's* live path actually
     * delivers it on the same data the Python one did.
     */
    @Test
    fun `trunk angle now matches Python over a replayed live buffer`() {
        val case = Goldens.group("live_lag_identity").asJsonObject
        val world = case["world"].points()
        val ts = case["timestamps_ms"].asJsonArray.let { a -> LongArray(a.size()) { a[it].asLong } }
        val ones = Array(world.size) { FloatArray(Landmarks.COUNT) { 1f } }
        val frames = poseFrames(world, ones)

        val window = LiveWindow(4096)
        val expected = case["samples"].asJsonArray.associate { s ->
            s.asJsonObject["frame"].asInt to s.asJsonObject
        }
        var checked = 0
        for (i in world.indices) {
            window.push(ts[i], frames[i])
            val golden = expected[i] ?: continue
            val (now, baseline) = LiveMetricsComputer.trunkAngleNow(window, window.windowSpan(5.0))
            assertClose(golden["trunk_angle_now"].num(), now, 1e-9, "trunkAngleNow[$i]")
            assertClose(golden["baseline"].num(), baseline, 1e-9, "baseline[$i]")
            checked++
        }
        assertTrue(checked > 0, "no sampled frames were compared")
    }

    private fun rmse(values: List<Double>): Double =
        sqrt(values.sumOf { it * it } / values.size)
}
