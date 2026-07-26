package dev.remapy.metrics

import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Reader for `goldens.json`, the cross-language fixtures `pixi run export-fixtures` writes.
 *
 * The file is the Python kernel's actual behaviour, not a transcription of it. Everything in
 * this package that claims agreement with `motor_metrics` claims it against this file.
 *
 * Non-finite doubles arrive as the string tokens `"nan"` / `"inf"` / `"-inf"`, because bare
 * `NaN` is not valid JSON. That round trip is not incidental: NaN is what every metric here
 * returns instead of throwing, so a fixture format that dropped it would silently stop
 * testing the single most load-bearing convention in the package.
 */
object Goldens {
    val root: JsonObject by lazy {
        val stream = Goldens::class.java.getResourceAsStream("/goldens.json")
            ?: error(
                "goldens.json is not on the test classpath. Generate it with " +
                    "`pixi run export-fixtures` from the repository root."
            )
        Gson().fromJson(stream.reader(), JsonObject::class.java)
    }

    fun group(name: String): JsonElement = root[name] ?: error("No golden group '$name'.")

    /** Cases stored as a list of objects, keyed by their `name` field. */
    fun casesByName(groupName: String): Map<String, JsonObject> =
        group(groupName).asJsonArray.associate { it.asJsonObject["name"].asString to it.asJsonObject }
}

fun JsonElement.num(): Double = when {
    isJsonPrimitive && asJsonPrimitive.isString -> when (val token = asString) {
        "nan" -> Double.NaN
        "inf" -> Double.POSITIVE_INFINITY
        "-inf" -> Double.NEGATIVE_INFINITY
        else -> error("Unexpected non-finite token '$token'.")
    }
    else -> asDouble
}

fun JsonElement.doubles(): DoubleArray = asJsonArray.let { a -> DoubleArray(a.size()) { a[it].num() } }

fun JsonElement.ints(): IntArray = asJsonArray.let { a -> IntArray(a.size()) { a[it].asInt } }

fun JsonElement.matrix(): Matrix {
    val outer = asJsonArray
    if (outer.size() == 0) return Matrix(0, 0)
    val first = outer[0]
    // A 1-D export is a column; a 2-D one carries its own width.
    if (!first.isJsonArray) return Matrix.ofColumn(doubles())
    val cols = first.asJsonArray.size()
    val out = Matrix(outer.size(), cols)
    for (i in 0 until outer.size()) {
        val row = outer[i].asJsonArray
        for (c in 0 until cols) out[i, c] = row[c].num()
    }
    return out
}

/**
 * Rebuild an `(N, 33, 3)` landmark array from the sparse export.
 *
 * Values are stored, and rebuilt, through `Float`. That is not a size optimisation — it is
 * how the two implementations stay bit-comparable. Landmarks are float32 in the `.h5` and in
 * `LiveWindow`, and both sides widen to double only to compute, so a port that kept them as
 * doubles end to end would drift from Python in the last few digits of every metric.
 */
fun JsonElement.points(): Array<FloatArray> {
    val obj = asJsonObject
    val n = obj["n"].asInt
    val out = Array(n) { FloatArray(Landmarks.COUNT * 3) }
    for ((key, value) in obj["landmarks"].asJsonObject.entrySet()) {
        val index = key.toInt()
        val frames = value.asJsonArray
        for (i in 0 until n) {
            val row = frames[i].asJsonArray
            for (c in 0 until 3) out[i][index * 3 + c] = row[c].num().toFloat()
        }
    }
    for (frame in obj["nan_frames"].asJsonArray) {
        out[frame.asInt].fill(Float.NaN)
    }
    return out
}

/** Rebuild an `(N, 33)` visibility/presence array from the sparse export. */
fun JsonElement.scores(): Array<FloatArray> {
    val obj = asJsonObject
    val n = obj["n"].asInt
    val fill = obj["fill"].num().toFloat()
    val out = Array(n) { FloatArray(Landmarks.COUNT) { fill } }
    for ((key, value) in obj["overrides"].asJsonObject.entrySet()) {
        val index = key.toInt()
        val column = value.asJsonArray
        for (i in 0 until n) out[i][index] = column[i].num().toFloat()
    }
    return out
}

/** Rebuild the frames + timestamps of a recording payload as a [FrameBuffer]. */
fun JsonObject.asFrames(): FrameBuffer {
    val ts = this["timestamps_ms"].asJsonArray.let { a -> LongArray(a.size()) { a[it].asLong } }
    val world = this["world"].points()
    val norm = this["norm"].points()
    val visibility = this["visibility"].scores()
    val presence = this["presence"].scores()
    return FrameBuffer(ts, world, norm, visibility, presence)
}

// --------------------------------------------------------------------------- //
// Assertions
// --------------------------------------------------------------------------- //

/**
 * Assert two doubles agree, treating NaN as a value that must match NaN.
 *
 * A port that returned 0.0 where Python returns NaN would pass any tolerance-only check
 * while breaking the blanking rule outright — on screen, 0.0 reads as a measurement of the
 * child and NaN reads as `--`. So NaN-ness is compared before magnitude, always.
 */
fun assertClose(expected: Double, actual: Double, tolerance: Double = 1e-9, message: String = "") {
    val label = if (message.isEmpty()) "" else "$message: "
    if (expected.isNaN() || actual.isNaN()) {
        assertEquals(expected.isNaN(), actual.isNaN(), "${label}expected $expected, got $actual")
        return
    }
    if (expected.isInfinite() || actual.isInfinite()) {
        assertEquals(expected, actual, "${label}expected $expected, got $actual")
        return
    }
    val scale = maxOf(1.0, kotlin.math.abs(expected))
    val delta = kotlin.math.abs(expected - actual)
    assertTrue(
        delta <= tolerance * scale,
        "${label}expected $expected, got $actual (delta $delta, tolerance ${tolerance * scale})",
    )
}

fun assertClose(expected: DoubleArray, actual: DoubleArray, tolerance: Double = 1e-9, message: String = "") {
    assertEquals(expected.size, actual.size, "$message: length")
    for (i in expected.indices) assertClose(expected[i], actual[i], tolerance, "$message[$i]")
}

fun assertClose(expected: Matrix, actual: Matrix, tolerance: Double = 1e-9, message: String = "") {
    assertEquals(expected.rows, actual.rows, "$message: rows")
    if (expected.rows > 0) assertEquals(expected.cols, actual.cols, "$message: cols")
    for (i in 0 until expected.rows) {
        for (c in 0 until expected.cols) {
            assertClose(expected[i, c], actual[i, c], tolerance, "$message[$i,$c]")
        }
    }
}
