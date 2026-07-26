package dev.remapy.metrics

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max

/**
 * Resampling, smoothing and differentiation for landmark trajectories.
 *
 * Port of `motor_metrics/derive.py`. Read that module's docstring first — it carries the
 * reasoning, the measured derivative-gain table, and the reason [FS], [WINDOW_S] and [POLY]
 * are constants rather than parameters. The short version: two numbers are comparable only
 * through an identical filter chain, and these get compared across months of sessions.
 *
 * **This file is the one with real risk in it.** Everything else in the kernel is arithmetic
 * that either matches or obviously does not. Savitzky-Golay has an *edge* convention, and
 * `LIVE_LAG` — the whole reason a live readout is trustworthy — is defined by it. scipy's
 * default `mode="interp"` does not pad the signal: it fits one polynomial of order [POLY] to
 * the first `windowLength` samples and evaluates it at each of the first `halfWindow`
 * positions, and likewise at the tail. A port that mirrors, reflects or zero-pads instead
 * will agree on every interior sample and disagree on exactly the `windowLength / 2` that the
 * live instantaneous readout is taken from. `SavgolTest` and `LiveLagIdentityTest` pin it.
 */
object Derive {
    /**
     * Hz, the uniform grid every derived quantity is computed on.
     *
     * 15 because that is what the capture hardware sustains — the phone measured 15-20 fps
     * and both it and the desktop CLIs now request 15. Resampling *up* to a grid faster than
     * the camera delivers adds no information; it manufactures samples that are exact linear
     * combinations of their neighbours, and the filter then averages over samples that are
     * partly each other.
     */
    const val FS: Double = 15.0

    /**
     * Savitzky-Golay window, in seconds. **Paired with [FS]** — see [windowLength].
     */
    const val WINDOW_S: Double = 0.35

    /** Savitzky-Golay polynomial order. */
    const val POLY: Int = 2

    /**
     * Savitzky-Golay window in samples: odd, and greater than [POLY], as scipy requires.
     *
     * `max(int(window_s * fs), poly + 1)` in Python truncates toward zero; [windowLength]
     * only ever sees positive input, so [floor] matches.
     *
     * **The degenerate case is real and silent.** `poly + 1` samples fit a polynomial of order
     * [POLY] *exactly*, so a window that small is an interpolation rather than a fit: `deriv=0`
     * returns its input untouched and `deriv=1` collapses to a plain central difference, with
     * nothing raised and nothing NaN. At [POLY] = 2 that is `w == 3`, which `windowS = 0.25`
     * produces at any `fs` in [12, 20) — exactly where a 15 Hz grid sits. [WINDOW_S] is
     * therefore paired with [FS] and the two move together; `ConstantsTest` pins the pair
     * against the Python side's own value.
     */
    fun windowLength(fs: Double = FS, windowS: Double = WINDOW_S, poly: Int = POLY): Int {
        val w = max(floor(windowS * fs).toInt(), poly + 1)
        return if (w % 2 == 0) w + 1 else w
    }

    /**
     * Linearly resample [x] from timestamps [tMs] onto a uniform [fs] grid.
     *
     * Returns the grid in seconds from the first sample, and the resampled columns. Returns
     * empty arrays when fewer than two samples are given, or when the span is shorter than
     * one grid step — a caller asking for the derivative of a single point should get an
     * empty answer, not an exception.
     *
     * NaN inputs propagate to their neighbouring intervals rather than being dropped.
     * Silently interpolating across missing frames would fabricate the very movement the
     * tracker failed to see, which is the same refusal `longestRun` makes.
     */
    fun resampleUniform(tMs: DoubleArray, x: Matrix, fs: Double = FS): Pair<DoubleArray, Matrix> {
        val cols = x.cols
        if (tMs.size < 2 || x.rows < 2) return DoubleArray(0) to Matrix(0, cols)

        val tS = DoubleArray(tMs.size) { (tMs[it] - tMs[0]) / 1000.0 }
        val span = tS[tS.size - 1]
        val count = floor(span * fs).toInt() + 1
        if (count < 2) return DoubleArray(0) to Matrix(0, cols)

        val grid = DoubleArray(count) { it / fs }
        val out = Matrix(count, cols)
        for (c in 0 until cols) {
            for (i in 0 until count) out[i, c] = interp(grid[i], tS, x, c)
        }
        return grid to out
    }

    /** [resampleUniform] for a single column. */
    fun resampleUniform(tMs: DoubleArray, x: DoubleArray, fs: Double = FS): Pair<DoubleArray, DoubleArray> {
        val (t, m) = resampleUniform(tMs, Matrix.ofColumn(x), fs)
        return t to m.column(0)
    }

    /**
     * `np.interp` for one query point: clamped at both ends, NaN-propagating.
     *
     * numpy clamps outside the sample range to the endpoint values rather than
     * extrapolating. The grid here never exceeds the range by construction, but matching
     * the semantics costs nothing and removes a way for the two implementations to differ
     * only on a rounding edge.
     */
    private fun interp(at: Double, xs: DoubleArray, values: Matrix, col: Int): Double {
        val n = xs.size
        if (at <= xs[0]) return values[0, col]
        if (at >= xs[n - 1]) return values[n - 1, col]

        var lo = 0
        var hi = n - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) ushr 1
            if (xs[mid] <= at) lo = mid else hi = mid
        }
        val x0 = xs[lo]
        val x1 = xs[hi]
        val y0 = values[lo, col]
        val y1 = values[hi, col]
        if (x1 == x0) return y0
        return y0 + (y1 - y0) * (at - x0) / (x1 - x0)
    }

    /**
     * Savitzky-Golay filter (or its [deriv]-th derivative) along axis 0.
     *
     * Assumes [x] is already on a uniform [fs] grid — run it through [resampleUniform]
     * first. With `deriv = 1` the result is in units per second.
     *
     * Returns an all-NaN array of the input's shape when the input is shorter than the
     * window, rather than throwing as scipy's `savgol_filter` does. Short segments are
     * ordinary: a two-second sit is a real trial and a two-frame one is a mis-marked
     * annotation, and neither should take down a report that has 40 other rows in it.
     */
    fun smooth(
        x: Matrix,
        fs: Double = FS,
        windowS: Double = WINDOW_S,
        poly: Int = POLY,
        deriv: Int = 0,
    ): Matrix {
        val w = windowLength(fs, windowS, poly)
        if (x.rows < w) return Matrix.filled(x.rows, x.cols, Double.NaN)

        val delta = 1.0 / fs
        val half = w / 2
        val interior = savgolCoefficients(w, poly, deriv, delta, half.toDouble())
        val out = Matrix(x.rows, x.cols)

        for (c in 0 until x.cols) {
            val col = x.column(c)
            // Interior: a plain FIR convolution with the centred coefficients.
            for (i in half until x.rows - half) {
                var acc = 0.0
                for (k in 0 until w) acc += interior[k] * col[i - half + k]
                out[i, c] = acc
            }
            // Edges: scipy's mode="interp" fits ONE polynomial to the first (last) `w`
            // samples and evaluates it at each edge position, rather than extending the
            // signal. Equivalent to a per-position coefficient vector over that same block.
            for (i in 0 until half) {
                val coef = savgolCoefficients(w, poly, deriv, delta, i.toDouble())
                var acc = 0.0
                for (k in 0 until w) acc += coef[k] * col[k]
                out[i, c] = acc
            }
            for (i in 0 until half) {
                val pos = w - 1 - i
                val coef = savgolCoefficients(w, poly, deriv, delta, pos.toDouble())
                var acc = 0.0
                for (k in 0 until w) acc += coef[k] * col[x.rows - w + k]
                out[x.rows - 1 - i, c] = acc
            }
        }
        return out
    }

    /** [smooth] for a single column. */
    fun smooth(
        x: DoubleArray,
        fs: Double = FS,
        windowS: Double = WINDOW_S,
        poly: Int = POLY,
        deriv: Int = 0,
    ): DoubleArray = smooth(Matrix.ofColumn(x), fs, windowS, poly, deriv).column(0)

    /**
     * Savitzky-Golay coefficients: the [deriv]-th derivative of the least-squares
     * polynomial fit over a [windowLength]-sample block, evaluated at sample [pos].
     *
     * `pos = windowLength / 2` gives scipy's centred `savgol_coeffs`; other values give the
     * edge coefficients `mode="interp"` uses. Solved as a small normal-equation system —
     * the window is 5 samples at the shipped constants, so a general least-squares routine
     * would be more machinery than the problem has.
     */
    fun savgolCoefficients(
        windowLength: Int,
        poly: Int,
        deriv: Int,
        delta: Double,
        pos: Double,
    ): DoubleArray {
        require(windowLength > poly) { "windowLength must exceed poly, got $windowLength and $poly." }
        if (deriv > poly) return DoubleArray(windowLength)

        val order = poly + 1
        // Vandermonde on the sample offsets: A[i][j] = (i - pos)^j.
        val a = Array(windowLength) { i ->
            DoubleArray(order) { j -> powInt(i - pos, j) }
        }
        // Normal equations (A^T A) c = A^T e_deriv, where the right-hand side selects the
        // `deriv`-th Taylor coefficient of the fitted polynomial.
        val ata = Array(order) { r -> DoubleArray(order) { c ->
            var acc = 0.0
            for (i in 0 until windowLength) acc += a[i][r] * a[i][c]
            acc
        } }
        val rhs = DoubleArray(order)
        rhs[deriv] = factorial(deriv) / powInt(delta, deriv)

        val alpha = solveSymmetric(ata, rhs)
        // c_k = sum_j alpha_j * A[k][j]: the fitted derivative as a linear functional of the
        // samples, which is exactly the FIR kernel.
        return DoubleArray(windowLength) { k ->
            var acc = 0.0
            for (j in 0 until order) acc += alpha[j] * a[k][j]
            acc
        }
    }

    /** Gaussian elimination with partial pivoting. Systems here are (poly + 1) square. */
    private fun solveSymmetric(matrix: Array<DoubleArray>, rhs: DoubleArray): DoubleArray {
        val n = rhs.size
        val m = Array(n) { r -> DoubleArray(n + 1) { c -> if (c < n) matrix[r][c] else rhs[r] } }
        for (col in 0 until n) {
            var pivot = col
            for (r in col + 1 until n) if (abs(m[r][col]) > abs(m[pivot][col])) pivot = r
            val tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp
            val p = m[col][col]
            if (p == 0.0) continue
            for (r in 0 until n) {
                if (r == col) continue
                val factor = m[r][col] / p
                if (factor == 0.0) continue
                for (c in col..n) m[r][c] -= factor * m[col][c]
            }
        }
        return DoubleArray(n) { if (m[it][it] == 0.0) 0.0 else m[it][n] / m[it][it] }
    }

    private fun powInt(base: Double, exponent: Int): Double {
        var acc = 1.0
        repeat(exponent) { acc *= base }
        return acc
    }

    private fun factorial(n: Int): Double {
        var acc = 1.0
        for (i in 2..n) acc *= i
        return acc
    }

    /**
     * Velocity of a path: the uniform time grid, the per-axis velocity, and the scalar speed.
     *
     * All in units per second. Speed is NaN throughout when the segment is shorter than the
     * smoothing window.
     */
    fun velocity(tMs: DoubleArray, p: Matrix, fs: Double = FS): Triple<DoubleArray, Matrix, DoubleArray> {
        val (tS, uniform) = resampleUniform(tMs, p, fs)
        if (tS.isEmpty()) return Triple(DoubleArray(0), Matrix(0, p.cols), DoubleArray(0))
        val v = smooth(uniform, fs, deriv = 1)
        val speed = DoubleArray(v.rows) { i ->
            var acc = 0.0
            for (c in 0 until v.cols) acc += v[i, c] * v[i, c]
            kotlin.math.sqrt(acc)
        }
        return Triple(tS, v, speed)
    }
}
