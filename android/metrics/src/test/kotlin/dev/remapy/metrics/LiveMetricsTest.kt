package dev.remapy.metrics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The live path, against `motor_metrics/live.py`.
 *
 * The golden cases are whole **push sequences**, compared frame by frame, because the
 * interesting behaviour is stateful: the [LiveMetricsComputer.RECOMPUTE_EVERY] reuse, the
 * blanking when coverage falls and the recompute when it recovers, and the ring buffer
 * wrapping once the window fills. None of that shows in a single snapshot.
 */
class LiveMetricsTest {

    private fun replay(case: com.google.gson.JsonObject): Pair<LiveMetricsComputer, Map<Int, LiveMetrics>> {
        val world = case["world"].points()
        val visibility = case["visibility"].scores()
        val ts = case["timestamps_ms"].asJsonArray.let { a -> LongArray(a.size()) { a[it].asLong } }
        val frames = poseFrames(world, visibility)
        val windowS = case["window_s"].let { if (it.isJsonNull) null else it.num() }
        val computer = LiveMetricsComputer(case["mode"].asString, windowS = windowS)

        val readouts = LinkedHashMap<Int, LiveMetrics>()
        for (i in world.indices) readouts[i] = computer.push(ts[i], frames[i])
        return computer to readouts
    }

    @Test
    fun `live push sequences match Python frame by frame`() {
        for ((name, case) in Goldens.casesByName("live_push")) {
            val (_, readouts) = replay(case)
            for (entry in case["readouts"].asJsonArray) {
                val e = entry.asJsonObject
                val frame = e["frame"].asInt
                assertMetricsMatch(
                    e["metrics"].asJsonObject,
                    readouts.getValue(frame).toMap(),
                    1e-9,
                    label = "live[$name]@$frame",
                )
            }
        }
    }

    /**
     * **Blanking.** Below `MIN_COVERAGE` the readout goes NaN rather than going stale.
     *
     * A number left on screen while tracking has failed reads as a measurement of the child,
     * which is worse than showing nothing. And when coverage recovers, the value must come back
     * *recomputed* — never the pre-dropout value resurrected.
     */
    @Test
    fun `a dropout blanks the readout and recovery recomputes it`() {
        val case = Goldens.casesByName("live_push").getValue("hold_dropout")
        val (_, readouts) = replay(case)

        // Every invalid readout carries NaN, whatever made it invalid. There are two causes and
        // they are worth keeping distinct: low coverage (the blanking rule) and a window with
        // fewer than two frames in it (cold start, where coverage is a perfectly healthy 1.0).
        val invalid = readouts.values.filter { !it.liveValid }
        for (m in invalid) {
            assertTrue(m.liveSwayRmsM.isNaN(), "a blanked readout must carry NaN, not a stale value")
            assertTrue(m.liveTrunkAngleDeltaDeg.isNaN())
        }

        val blankedByCoverage = invalid.filter { it.liveCoverage < LiveMetricsComputer.MIN_COVERAGE }
        assertTrue(blankedByCoverage.size > 20, "the fixture must actually exercise blanking")
        // The dropout is long enough that coverage genuinely collapses, not merely dips.
        assertTrue(
            blankedByCoverage.minOf { it.liveCoverage } < 0.4,
            "coverage should collapse well below the threshold, not hover at it",
        )

        // ...and coverage climbs back to a valid readout by the end of the run.
        val last = readouts.getValue(readouts.keys.max())
        assertTrue(last.liveValid, "coverage should have recovered")
        assertFalse(last.liveSwayRmsM.isNaN(), "a recovered readout must carry a number")
    }

    /**
     * Quality figures refresh **every** frame even when the measurements are reused.
     *
     * Coverage is what tells the operator the readout is still trustworthy; a coverage number
     * that lagged the video by up to `RECOMPUTE_EVERY` frames would be reassuring at exactly
     * the wrong moment.
     */
    @Test
    fun `coverage refreshes every frame while measurements are reused`() {
        val case = Goldens.casesByName("live_push").getValue("hold_extrapolated")
        val (_, readouts) = replay(case)
        // Frames where visibility drops are the ones that matter: consecutive readouts must show
        // coverage moving even between recomputes.
        val changes = readouts.keys.sorted().zipWithNext()
            .count { (a, b) -> readouts.getValue(a).liveCoverage != readouts.getValue(b).liveCoverage }
        assertTrue(
            changes > readouts.size / LiveMetricsComputer.RECOMPUTE_EVERY,
            "coverage changed only $changes times; it must not be gated by RECOMPUTE_EVERY",
        )
    }

    /** The ring buffer must stay ordered and time-bounded after wrapping many times. */
    @Test
    fun `the window keeps its duration after the ring wraps`() {
        val case = Goldens.casesByName("live_push").getValue("hold_ring_wrap")
        val (computer, readouts) = replay(case)
        val windowS = computer.windowS
        val span = computer.window.windowSpan(windowS)

        // Timestamps must still be ascending oldest-to-newest across the wrap point.
        for (i in 1 until computer.window.size) {
            assertTrue(
                computer.window.timestampMs(i) > computer.window.timestampMs(i - 1),
                "ring order broke at $i",
            )
        }
        val spanned = (computer.window.timestampMs(span.stop - 1) - computer.window.timestampMs(span.start)) / 1000.0
        assertTrue(spanned <= windowS + 1e-9, "window spanned ${spanned}s, longer than ${windowS}s")
        assertTrue(spanned > windowS - 0.1, "window spanned only ${spanned}s")
        assertTrue(readouts.values.last().liveValid)
    }

    /** Crawl reads no vertical at all, which is what makes it the camera-robust mode. */
    @Test
    fun `crawl mode reports no up source and hold mode does`() {
        val crawl = LiveMetricsComputer(LiveMetricsComputer.CRAWL)
        val hold = LiveMetricsComputer(LiveMetricsComputer.HOLD)
        assertEquals("n/a", crawl.push(0, PoseFrame.noPose()).liveUpSource)
        assertEquals("world_y", hold.push(0, PoseFrame.noPose()).liveUpSource)
    }

    /**
     * A cold, empty or fully-untracked buffer must degrade, never throw.
     *
     * This runs inside the capture loop: an exception here costs the session, not a table row.
     */
    @Test
    fun `a cold or untracked buffer returns a blanked readout rather than throwing`() {
        for (mode in LiveMetricsComputer.MODE_WINDOW_S.keys) {
            val computer = LiveMetricsComputer(mode)
            val first = computer.push(0, PoseFrame.noPose())
            assertFalse(first.liveValid, "$mode: a cold buffer cannot be valid")
            assertEquals(0.0, first.liveCoverage, "$mode: no pose means no coverage")
            repeat(200) { computer.push((it + 1) * 33L, PoseFrame.noPose()) }
            val later = computer.push(10_000, PoseFrame.noPose())
            assertFalse(later.liveValid, "$mode: still nothing tracked")
            assertTrue(later.liveSwayRmsM.isNaN())
        }
    }

    @Test
    fun `an unknown mode is rejected`() {
        val error = kotlin.runCatching { LiveMetricsComputer("wobble") }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException, "expected IllegalArgumentException, got $error")
    }

    /**
     * The window is selected by **time**, not by a frame count.
     *
     * This matters more on Android than on the laptop: MediaPipe's LIVE_STREAM mode drops
     * frames under load rather than queueing them, so a frame-count window would silently
     * become a shorter window exactly when the device is struggling.
     */
    @Test
    fun `the window is time-bounded even when frames arrive irregularly`() {
        val window = LiveWindow(500)
        // 10 Hz for the first half, 60 Hz for the second: a frame-count window would cover
        // wildly different durations at the two rates.
        var t = 0L
        repeat(100) { window.push(t, PoseFrame.noPose()); t += 100 }
        repeat(200) { window.push(t, PoseFrame.noPose()); t += 17 }

        val span = window.windowSpan(2.0)
        val spanned = (window.timestampMs(span.stop - 1) - window.timestampMs(span.start)) / 1000.0
        assertTrue(spanned in 1.9..2.0, "expected ~2 s, got ${spanned}s")
    }

    /**
     * A per-push budget. `live.py` measures ~2.9 ms for a full recompute over a 5 s window on a
     * laptop; this is a loose ceiling on a JVM, not a phone benchmark. Its job is to catch an
     * accidental quadratic — a port that recomputed per frame instead of per
     * [LiveMetricsComputer.RECOMPUTE_EVERY], say — not to certify device performance. Real
     * frame-budget evidence has to come from a device.
     */
    @Test
    fun `a full push sequence stays far inside the frame budget`() {
        val n = 900 // 30 s at 30 Hz
        val frames = Bodies.frames(Bodies.swayingTrunk(n, 0.03, 0.4))
        val poses = (0 until n).map {
            PoseFrame(frames.worldRow(it), frames.normRow(it), FloatArray(33) { 1f }, FloatArray(33) { 1f })
        }
        val computer = LiveMetricsComputer(LiveMetricsComputer.HOLD)
        repeat(2) { // warm the JIT; the first pass measures the compiler, not the code
            for (i in 0 until n) computer.push(frames.timestampMs(i), poses[i])
        }

        val fresh = LiveMetricsComputer(LiveMetricsComputer.HOLD)
        val startNs = System.nanoTime()
        for (i in 0 until n) fresh.push(frames.timestampMs(i), poses[i])
        val perPushMs = (System.nanoTime() - startNs) / 1e6 / n

        assertTrue(perPushMs < 5.0, "mean push took ${perPushMs} ms, well over the 33 ms frame budget's share")
    }
}
