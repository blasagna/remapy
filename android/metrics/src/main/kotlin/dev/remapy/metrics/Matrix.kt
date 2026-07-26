package dev.remapy.metrics

import kotlin.math.sqrt

/**
 * A dense row-major `(rows, cols)` block of doubles.
 *
 * Deliberately not a linear-algebra library. The shapes this kernel handles are fixed and
 * tiny — landmark frames are `(N, 33, 3)` and every derived signal is `(N, 1)` or `(N, 2)` —
 * so the whole numeric surface is indexing, a covariance, and a 2x2 eigenvalue. Pulling in a
 * matrix dependency to express that would be more coupling than arithmetic, and it would put
 * a third implementation of `mean` between this port and the numpy it has to agree with.
 *
 * NaN is a *value* here, not an error: a frame with no pose is a full-NaN row by the
 * `landmark_rows` convention, and every statistic below propagates it rather than skipping
 * it. Callers that want NaN-tolerance ask for it explicitly.
 */
class Matrix(val rows: Int, val cols: Int, val data: DoubleArray = DoubleArray(rows * cols)) {

    init {
        require(data.size == rows * cols) { "data is ${data.size}, expected ${rows * cols}." }
    }

    operator fun get(row: Int, col: Int): Double = data[row * cols + col]

    operator fun set(row: Int, col: Int, value: Double) {
        data[row * cols + col] = value
    }

    fun column(col: Int): DoubleArray = DoubleArray(rows) { this[it, col] }

    fun row(row: Int): DoubleArray = DoubleArray(cols) { this[row, it] }

    /** Rows `[start, stop)`, copied. */
    fun slice(start: Int, stop: Int): Matrix {
        val n = (stop - start).coerceAtLeast(0)
        val out = Matrix(n, cols)
        for (i in 0 until n) for (c in 0 until cols) out[i, c] = this[start + i, c]
        return out
    }

    fun columnMeans(): DoubleArray = DoubleArray(cols) { c ->
        var acc = 0.0
        for (i in 0 until rows) acc += this[i, c]
        acc / rows
    }

    /** This matrix with each column's mean subtracted. */
    fun centred(): Matrix {
        val means = columnMeans()
        val out = Matrix(rows, cols)
        for (i in 0 until rows) for (c in 0 until cols) out[i, c] = this[i, c] - means[c]
        return out
    }

    fun allFinite(): Boolean = data.all { it.isFinite() }

    fun isEmpty(): Boolean = rows == 0

    companion object {
        fun ofColumn(values: DoubleArray): Matrix = Matrix(values.size, 1, values.copyOf())

        fun filled(rows: Int, cols: Int, value: Double): Matrix =
            Matrix(rows, cols, DoubleArray(rows * cols) { value })

        fun ofRows(rows: List<DoubleArray>): Matrix {
            if (rows.isEmpty()) return Matrix(0, 0)
            val out = Matrix(rows.size, rows[0].size)
            rows.forEachIndexed { i, r -> r.forEachIndexed { c, v -> out[i, c] = v } }
            return out
        }
    }
}

/** Total distance travelled along an `(N, K)` path. NaN for a path shorter than 2 or non-finite. */
fun pathLength(points: Matrix): Double {
    if (points.rows < 2 || !points.allFinite()) return Double.NaN
    var total = 0.0
    for (i in 1 until points.rows) {
        var sq = 0.0
        for (c in 0 until points.cols) {
            val d = points[i, c] - points[i - 1, c]
            sq += d * d
        }
        total += sqrt(sq)
    }
    return total
}

/** Population standard deviation, matching `np.std`'s default `ddof=0`. */
fun std(values: DoubleArray): Double {
    if (values.isEmpty()) return Double.NaN
    val mean = values.average()
    var acc = 0.0
    for (v in values) acc += (v - mean) * (v - mean)
    return sqrt(acc / values.size)
}

/** Median, matching `np.median` — the mean of the two middle values for an even count. */
fun median(values: DoubleArray): Double {
    if (values.isEmpty()) return Double.NaN
    val sorted = values.sortedArray()
    val mid = sorted.size / 2
    return if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2.0
}

/** Peak-to-peak range, matching `np.ptp`. NaN-propagating, as numpy's is. */
fun ptp(values: DoubleArray): Double {
    if (values.isEmpty()) return Double.NaN
    var lo = values[0]
    var hi = values[0]
    for (v in values) {
        if (v.isNaN()) return Double.NaN
        if (v < lo) lo = v
        if (v > hi) hi = v
    }
    return hi - lo
}
