package dev.remapy.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Frame gating, against `motor_metrics/quality.py`. */
class QualityTest {

    private val case = Goldens.group("quality").asJsonObject

    private fun frames(): PoseFrames {
        val world = case["world"].points()
        val ts = case["timestamps_ms"].asJsonArray.let { a -> LongArray(a.size()) { a[it].asLong } }
        return FrameBuffer(ts, world, world, case["visibility"].scores(), case["presence"].scores())
    }

    @Test
    fun `pose present is the whole-row NaN check`() {
        val expected = case["pose_present"].asJsonArray.map { it.asBoolean }
        val f = frames()
        for (i in expected.indices) assertEquals(expected[i], f.posePresent(i), "posePresent[$i]")
    }

    @Test
    fun `landmarks ok matches for both a torso and a wrist gate`() {
        val f = frames()
        for ((key, indices) in listOf(
            "landmarks_ok_torso" to Landmarks.TORSO,
            "landmarks_ok_wrists" to Landmarks.WRISTS,
        )) {
            val expected = case[key].asJsonArray.map { it.asBoolean }
            val actual = Quality.landmarksOk(f, indices)
            for (i in expected.indices) assertEquals(expected[i], actual[i], "$key[$i]")
        }
    }

    @Test
    fun `coverage and longest run match across spans, including an empty one`() {
        val ok = Quality.landmarksOk(frames(), Landmarks.TORSO)
        for (span in case["spans"].asJsonArray) {
            val s = span.asJsonObject
            val start = s["start"].asInt
            val stop = s["stop"].asInt
            assertClose(s["coverage"].num(), Quality.coverage(ok, start, stop), 1e-12, "coverage[$start,$stop)")
            val expected = s["longest_run"].ints()
            val actual = Quality.longestRun(ok, start, stop)
            assertEquals(expected[0], actual.start, "longestRun[$start,$stop).start")
            assertEquals(expected[1], actual.stop, "longestRun[$start,$stop).stop")
        }
    }

    /**
     * The trap this module exists for, restated where it is most tempting to conflate.
     *
     * MediaPipe *extrapolates* occluded landmarks rather than dropping them, so a frame can be
     * `posePresent` while carrying invented coordinates. If these two ever agree, something has
     * "helpfully" strengthened `posePresent` and every metric silently starts trusting
     * fabricated points.
     */
    @Test
    fun `an extrapolated frame is pose-present but not landmarks-ok`() {
        val world = Array(1) { FloatArray(Landmarks.COUNT * 3) { 0.1f } }
        val low = Array(1) { FloatArray(Landmarks.COUNT) { 0.2f } }
        val f = FrameBuffer(longArrayOf(0), world, world, low, low)
        assertTrue(f.posePresent(0), "a low-visibility frame still carries a pose")
        assertFalse(Quality.landmarksOk(f, Landmarks.TORSO)[0], "...but must not be trusted")
    }

    /**
     * Coverage of an empty span is 0.0 and **not** NaN.
     *
     * NaN compares false against every threshold, so an empty trial checked with
     * `coverage < 0.5` would silently *pass* the check it was meant to fail.
     */
    @Test
    fun `an empty span has zero coverage, not NaN`() {
        val coverage = Quality.coverage(BooleanArray(10) { true }, 5, 5)
        assertEquals(0.0, coverage)
        assertTrue(coverage < LiveMetricsComputer.MIN_COVERAGE, "must fail a threshold check")
    }

    /**
     * A run is never stitched across a dropout: that would invent the movement inside it.
     */
    @Test
    fun `longest run never bridges a gap`() {
        //                 0     1     2     3      4     5     6     7     8
        val mask = booleanArrayOf(true, true, true, false, true, true, true, true, true)
        val run = Quality.longestRun(mask, 0, mask.size)
        assertEquals(4, run.start)
        assertEquals(9, run.stop)
        assertEquals(5, run.nFrames, "must be the later run of 5, not the stitched 8")
    }

    /** Ties go to the first run, matching `np.argmax`. */
    @Test
    fun `the first of two equal-length runs wins`() {
        val mask = booleanArrayOf(true, true, false, true, true)
        val run = Quality.longestRun(mask, 0, mask.size)
        assertEquals(0, run.start)
        assertEquals(2, run.stop)
    }

    @Test
    fun `a span with nothing passing has a zero-length run at its start`() {
        val run = Quality.longestRun(BooleanArray(10), 3, 8)
        assertEquals(3, run.start)
        assertEquals(3, run.stop)
        assertEquals(0, run.nFrames)
    }
}
