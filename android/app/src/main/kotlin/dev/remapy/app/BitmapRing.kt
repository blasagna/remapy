package dev.remapy.app

import android.graphics.Bitmap

/**
 * A tiny fixed ring of reusable display bitmaps.
 *
 * **Why this exists.** A 1280x720 ARGB_8888 bitmap is ~3.7 MB, and since Android 8 that lives on
 * the *native* heap. Allocating a fresh one per frame at 25-30 fps is ~100 MB/s of churn, which
 * outruns the GC: measured on the emulator, the app's native heap climbed 217 -> 358 MB in under
 * two minutes and kept going. A therapy session is minutes long, so "it settles eventually" is not
 * good enough — that is an out-of-memory kill partway through a trial.
 *
 * **Why a ring rather than one buffer.** The pipeline writes on the detector's callback thread
 * while Compose reads on the UI thread, so reusing a single bitmap would tear the frame being
 * drawn. Three buffers give the compositor two frames of slack behind a producer that is already
 * the slower of the two, which is ample; a single buffer would not be, and two leaves no margin
 * for a stalled draw.
 *
 * Not thread-safe by itself — [acquire] is only ever called from the detector callback.
 */
class BitmapRing(private val capacity: Int = 3) {

    private val buffers = arrayOfNulls<Bitmap>(capacity)
    private var next = 0

    /**
     * The next buffer in the ring, sized [width] x [height].
     *
     * Reallocates the slot if the frame size changed — which happens once, when the camera
     * delivers its first frame, and then never again unless the device rotates.
     */
    fun acquire(width: Int, height: Int): Bitmap {
        val i = next
        next = (next + 1) % capacity
        val existing = buffers[i]
        if (existing != null && !existing.isRecycled &&
            existing.width == width && existing.height == height
        ) {
            return existing
        }
        existing?.recycle()
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { buffers[i] = it }
    }

    fun release() {
        for (i in buffers.indices) {
            buffers[i]?.recycle()
            buffers[i] = null
        }
    }
}
