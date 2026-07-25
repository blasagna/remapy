package dev.remapy.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import com.google.mediapipe.tasks.components.containers.Detection
import dev.remapy.metrics.Landmarks
import dev.remapy.metrics.PoseFrame

/**
 * Face redaction, ported from `face_blur/`.
 *
 * **The repo-wide invariant, restated because this is where it is enforced on Android:** redaction
 * is applied to the *image sink only*, always **after** detection has run on the raw frame. Pose
 * accuracy is therefore unaffected, and only redacted frames are ever shown.
 *
 * That invariant is the reason this app does not use CameraX's `PreviewView`. A `Preview` use case
 * hands the camera stream straight to a `SurfaceView` — the raw, unredacted feed would reach the
 * screen without passing through here at all. Rendering the analysed frames ourselves costs some
 * smoothness (display rate equals analysis rate) and buys the guarantee.
 *
 * `box` is **irreversible**; `mosaic` is only weak de-identification and is recoverable by ML
 * re-identification. Box is the default for that reason.
 */
object FaceRedaction {

    enum class Style { BOX, MOSAIC }

    /** Which source the head box comes from — `face_blur/factory.py`'s `BLUR_METHODS`. */
    enum class Method { DETECTOR, POSE, HYBRID }

    /** Padding fractions from `face_blur/pose_blur.py`. */
    private const val PAD = 0.3

    /** Extra headroom above: the pose keypoints stop at the eyebrows and miss forehead and hair. */
    private const val TOP_PAD = 0.9

    private const val MIN_VISIBILITY = 0.5f

    /** Padding applied to a detector box, from `face_blur/blur.py`. */
    private const val DETECTOR_PAD = 0.15

    private const val MOSAIC_BLOCKS = 12

    private val paint = Paint().apply { color = Color.BLACK }

    /**
     * Redact every face in [bitmap], in place.
     *
     * Returns whether anything was redacted, which the caller needs: with [Method.POSE] a frame
     * with no detected pose yields **no box at all**, and showing it unredacted would break the
     * invariant this object exists to keep. [Method.HYBRID] — the default, matching the Python
     * CLIs — closes that gap by falling back to the detector, which is why it is worth carrying a
     * second 230 KB model.
     */
    fun redact(
        bitmap: Bitmap,
        frame: PoseFrame?,
        detections: List<Detection>,
        method: Method,
        style: Style,
    ): Boolean {
        val boxes = when (method) {
            Method.POSE -> listOfNotNull(poseHeadBox(bitmap, frame))
            Method.DETECTOR -> detectorBoxes(bitmap, detections)
            // Pose keypoints when a body is tracked (more reliable at odd angles, in profile, and
            // for small faces), the detector otherwise — which covers close-up framing where the
            // pose model may not fire at all.
            Method.HYBRID -> listOfNotNull(poseHeadBox(bitmap, frame))
                .ifEmpty { detectorBoxes(bitmap, detections) }
        }
        for (box in boxes) redactRegion(bitmap, box, style)
        return boxes.isNotEmpty()
    }

    /** Blank the whole frame. The last resort when nothing located a face but one may be present. */
    fun redactAll(bitmap: Bitmap, style: Style) {
        redactRegion(bitmap, Rect(0, 0, bitmap.width, bitmap.height), style)
    }

    /**
     * Head box from the pose model's face keypoints (landmarks 0-10: nose, eyes, ears, mouth).
     *
     * No second model needed. Returns null when no pose is present or the face keypoints are below
     * the visibility gate — MediaPipe *extrapolates* occluded landmarks, so an invented face box
     * would redact the wrong part of the frame and leave the real face showing.
     */
    private fun poseHeadBox(bitmap: Bitmap, frame: PoseFrame?): Rect? {
        if (frame == null) return null
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var seen = 0
        for (i in Landmarks.FACE) {
            if (frame.visibility[i] < MIN_VISIBILITY) continue
            val x = frame.norm[i * 3]
            val y = frame.norm[i * 3 + 1]
            if (x.isNaN() || y.isNaN()) continue
            minX = minOf(minX, x)
            minY = minOf(minY, y)
            maxX = maxOf(maxX, x)
            maxY = maxOf(maxY, y)
            seen++
        }
        if (seen == 0) return null

        val w = bitmap.width
        val h = bitmap.height
        return paddedBounds(
            (minX * w).toInt(), (minY * h).toInt(), (maxX * w).toInt(), (maxY * h).toInt(),
            w, h, PAD, TOP_PAD,
        )
    }

    private fun detectorBoxes(bitmap: Bitmap, detections: List<Detection>): List<Rect> =
        detections.map { d ->
            val b = d.boundingBox()
            paddedBounds(
                b.left.toInt(), b.top.toInt(), b.right.toInt(), b.bottom.toInt(),
                bitmap.width, bitmap.height, DETECTOR_PAD, DETECTOR_PAD,
            )
        }

    /** Pad a box outward (with optional extra headroom) and clamp it to the frame. */
    private fun paddedBounds(
        x0: Int, y0: Int, x1: Int, y1: Int,
        width: Int, height: Int,
        pad: Double, topPad: Double,
    ): Rect {
        val boxW = (x1 - x0).coerceAtLeast(1)
        val boxH = (y1 - y0).coerceAtLeast(1)
        val dx = (boxW * pad).toInt()
        val dy = (boxH * pad).toInt()
        val dTop = (boxH * topPad).toInt()
        return Rect(
            (x0 - dx).coerceIn(0, width),
            (y0 - dTop).coerceIn(0, height),
            (x1 + dx).coerceIn(0, width),
            (y1 + dy).coerceIn(0, height),
        )
    }

    private fun redactRegion(bitmap: Bitmap, box: Rect, style: Style) {
        if (box.width() <= 0 || box.height() <= 0) return
        when (style) {
            Style.BOX -> Canvas(bitmap).drawRect(box, paint)
            Style.MOSAIC -> mosaic(bitmap, box)
        }
    }

    /** Downsample the region and blow it back up with nearest-neighbour, as `redact.py` does. */
    private fun mosaic(bitmap: Bitmap, box: Rect) {
        val w = box.width()
        val h = box.height()
        val smallW = (w / MOSAIC_BLOCKS).coerceAtLeast(1)
        val smallH = (h / MOSAIC_BLOCKS).coerceAtLeast(1)

        val region = Bitmap.createBitmap(bitmap, box.left, box.top, w, h)
        val small = region.scale(smallW, smallH, filter = true)
        val blocky = small.scale(w, h, filter = false)
        Canvas(bitmap).drawBitmap(blocky, box.left.toFloat(), box.top.toFloat(), null)

        region.recycle()
        small.recycle()
        blocky.recycle()
    }

    private fun Bitmap.scale(width: Int, height: Int, filter: Boolean): Bitmap =
        Bitmap.createScaledBitmap(this, width, height, filter)
}
