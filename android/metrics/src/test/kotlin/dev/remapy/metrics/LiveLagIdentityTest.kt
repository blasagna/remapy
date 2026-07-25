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
 * Everything here follows from one fact — `Derive.windowLength()` is 7, so the interior fit
 * needs three samples either side and the last three of **any** window are fitted from one
 * side only. Reading three samples back steps off the extrapolated tail entirely.
 */
class LiveLagIdentityTest {

    private val fs = Derive.FS
    private val lag = LiveMetricsComputer.LIVE_LAG

    /** A sway-like signal: slow fundamental plus a little high-frequency landmark noise. */
    private fun signal(n: Int): DoubleArray = DoubleArray(n) {
        val t = it / fs
        0.03 * sin(2 * PI * 0.4 * t) + 0.002 * sin(2 * PI * 5.0 * t + 0.7)
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
            val windowStart = end - 150 // a 5 s window at 30 Hz
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
            val live = Derive.smooth(x.sliceArray(end - 150 until end), deriv = 1)
            assertClose(offline[end - 1 - lag], live[live.size - 1 - lag], 1e-12, "velocity at end=$end")
        }
    }

    /**
     * **The other half, and the reason nobody may "simplify" the lag to zero.**
     *
     * At the window edge the Savitzky-Golay fit is one-sided and the derivative it extrapolates
     * is essentially all error — the Python docstring measures RMSE 0.0757 m/s against a signal
     * whose own velocity sd was 0.0695 m/s. This asserts the shape of that result: the edge
     * error is a large fraction of the signal's own variation, while the lag-3 error is zero.
     */
    @Test
    fun `the edge-extrapolated derivative is essentially all error`() {
        val n = 600
        val x = signal(n)
        val offline = Derive.smooth(x, deriv = 1)

        val edgeErrors = ArrayList<Double>()
        val laggedErrors = ArrayList<Double>()
        for (end in 200..n step 3) {
            val live = Derive.smooth(x.sliceArray(end - 150 until end), deriv = 1)
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
                val live = Derive.smooth(x.sliceArray(end - 150 until end), deriv = 1)
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
