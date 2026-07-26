package dev.remapy.metrics

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The filter chain, against `motor_metrics/derive.py`.
 *
 * Two kinds of test here and both are needed. The golden checks pin *scipy's* behaviour,
 * including the `mode="interp"` edge fit that nothing about the formula would suggest. The
 * closed-form checks pin the *intent* — a ramp differentiates to its slope, the gain rolls
 * off the way the module docstring says it does — so when the two disagree there is
 * something to arbitrate with.
 */
class DeriveTest {

    @Test
    fun `window length matches scipy's odd-and-greater-than-poly rule`() {
        for (case in Goldens.group("window_length").asJsonArray) {
            val c = case.asJsonObject
            assertEquals(
                c["expected"].asInt,
                Derive.windowLength(c["fs"].num(), c["window_s"].num(), c["poly"].asInt),
                "fs=${c["fs"]} window_s=${c["window_s"]} poly=${c["poly"]}",
            )
        }
    }

    @Test
    fun `savgol matches scipy including the interp edges`() {
        for ((name, case) in Goldens.casesByName("savgol")) {
            val input = case["input"].matrix()
            val expected = case["expected"].matrix()
            val actual = Derive.smooth(input, deriv = case["deriv"].asInt)
            assertClose(expected, actual, 1e-9, "savgol[$name]")
        }
    }

    /**
     * The edges specifically, stated as its own test so a failure names the cause.
     *
     * If this fails while the interior passes, the port is padding or reflecting the signal
     * instead of fitting a polynomial to the end block — and `LIVE_LAG`'s guarantee that a
     * live readout equals the offline one goes with it.
     */
    @Test
    fun `savgol edge samples are a polynomial fit, not a padded convolution`() {
        val case = Goldens.casesByName("savgol")["noisy_deriv1"]!!
        val expected = case["expected"].matrix().column(0)
        val actual = Derive.smooth(case["input"].matrix(), deriv = 1).column(0)
        val half = Derive.windowLength() / 2
        for (i in 0 until half) {
            assertClose(expected[i], actual[i], 1e-9, "leading edge[$i]")
            val tail = expected.size - 1 - i
            assertClose(expected[tail], actual[tail], 1e-9, "trailing edge[$tail]")
        }
    }

    @Test
    fun `resample lands on a uniform grid regardless of timestamp jitter`() {
        for ((name, case) in Goldens.casesByName("resample_uniform")) {
            val tMs = case["t_ms"].doubles()
            val x = case["x"].matrix()
            val (tS, out) = Derive.resampleUniform(tMs, x)
            assertClose(case["t_s"].doubles(), tS, 1e-12, "resample[$name].t_s")
            assertClose(case["expected"].matrix(), out, 1e-9, "resample[$name]")
        }
    }

    @Test
    fun `smooth returns all-NaN below the window rather than throwing`() {
        val short = Matrix.ofColumn(DoubleArray(Derive.windowLength() - 1) { it.toDouble() })
        val out = Derive.smooth(short)
        assertEquals(short.rows, out.rows)
        assertTrue(out.data.all { it.isNaN() }, "expected every value NaN, got ${out.data.toList()}")
    }

    @Test
    fun `smooth of an empty signal is empty`() {
        assertEquals(0, Derive.smooth(Matrix(0, 2)).rows)
    }

    /** A quadratic is in the fit's span, so a POLY=2 filter must reproduce it exactly. */
    @Test
    fun `quadratic passes through the filter unchanged`() {
        val n = 40
        val x = DoubleArray(n) { 0.5 * it * it - 3.0 * it + 7.0 }
        val out = Derive.smooth(x)
        for (i in 0 until n) assertClose(x[i], out[i], 1e-8, "quadratic[$i]")
    }

    /** ...and its derivative must come out as the analytic one, edges included. */
    @Test
    fun `derivative of a quadratic is exact everywhere including the edges`() {
        val n = 40
        val fs = Derive.FS
        val x = DoubleArray(n) { val t = it / fs; 0.5 * t * t - 3.0 * t + 7.0 }
        val out = Derive.smooth(x, deriv = 1)
        for (i in 0 until n) {
            val t = i / fs
            assertClose(t - 3.0, out[i], 1e-7, "d/dt quadratic[$i]")
        }
    }

    /**
     * The measured derivative gain, from `derive.py`'s docstring table.
     *
     * A *bias*, identical for every trial through the same chain, which is what lets it
     * cancel in within-child comparisons — and why it has to be the same bias on Android as
     * on the laptop, not merely a small one.
     */
    @Test
    fun `derivative gain rolls off with frequency exactly as documented`() {
        for (case in Goldens.group("derivative_gain").asJsonArray) {
            val c = case.asJsonObject
            val freq = c["freq_hz"].num()
            val n = 600
            val fs = Derive.FS
            val signal = DoubleArray(n) { sin(2 * PI * freq * it / fs) }
            val analytic = DoubleArray(n) { 2 * PI * freq * cos(2 * PI * freq * it / fs) }
            val measured = Derive.smooth(signal, deriv = 1)
            val interior = 20 until n - 20
            val gain = std(measured.sliceArray(interior)) / std(analytic.sliceArray(interior))
            assertClose(c["gain"].num(), gain, 1e-9, "gain at ${freq} Hz")
        }
    }
}
