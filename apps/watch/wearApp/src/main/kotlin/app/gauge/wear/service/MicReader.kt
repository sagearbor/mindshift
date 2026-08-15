package app.gauge.wear.service

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.MicSource
import app.gauge.wear.control.NOOP_DIAG

/**
 * Real [MicSource]: reads one-second (16000-sample) windows of 16 kHz mono PCM16 audio off the
 * device mic. All decision-making (what a window *means*) lives in SentinelController — this
 * class only knows how to fill a fixed-size buffer.
 *
 * Not thread-safe by contract, same as [AudioRecord] itself: [start], [readWindow], and [release]
 * are meant to be called from a single thread (SentinelService's dedicated loop thread).
 *
 * Round-1 review fix (Important 3, P5-3): [diag] exists so a failed [pause]/[resume] leaves a
 * breadcrumb — both are still fail-soft (never throw), but a swallowed [resume] failure otherwise
 * means the very next [readWindow] silently returns `null`, which [app.gauge.wear.control.
 * SentinelController.tick] reads as MIC LOST and auto-disarms on with zero telemetry explaining
 * why. Defaults to [NOOP_DIAG] like every other diag-taking class in this app.
 */
class MicReader(private val diag: DiagLog = NOOP_DIAG) : MicSource {
    private var audioRecord: AudioRecord? = null

    // Round-2 review fix (Important 3, P5-3): warn-once state for [pause] failures — see [pause]'s
    // own KDoc for why this is bounded here (MicReader-internal) rather than as loop-thread state
    // in SentinelService, unlike Task 5's HrDemandPolicy warn-once flag. Reset on a SUCCESSFUL
    // [pause] (a distinct new failure stretch warns again) and in [release] (this instance's own
    // session boundary — always called at the end of an armed session, so the next arm starts
    // fresh).
    private var pauseFailureWarned = false

    /** Starts the underlying [AudioRecord]. Requires RECORD_AUDIO to already be granted. */
    @SuppressLint("MissingPermission")
    fun start() {
        val minBufferBytes = AudioRecord.getMinBufferSize(
            SAMPLE_RATE_HZ,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        // At least two full windows of headroom so a slow-draining consumer doesn't underrun.
        val bufferBytes = maxOf(minBufferBytes, WINDOW_SAMPLES * BYTES_PER_SAMPLE * 2)
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE_HZ,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferBytes,
        )
        try {
            check(record.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialize" }
            record.startRecording()
        } catch (e: Exception) {
            // record was constructed (a real native AudioRecord exists) but never made it into
            // `audioRecord`, so release() can't reach it — without this, a failed init (or a
            // startRecording() failure) leaks the native resource, and repeated ARM-while-mic-busy
            // retries would leak one AudioRecord per attempt.
            record.release()
            throw e
        }
        audioRecord = record
    }

    /**
     * Blocking-reads exactly one [WINDOW_SAMPLES]-sample window. [AudioRecord.read] can return
     * fewer samples than requested, so this loops until the window is full — and returns `null`
     * (mic gone) on any error code or on a `0`-sample read, since blocking reads returning nothing
     * only happens once the record has been stopped/released out from under us.
     */
    override fun readWindow(): ShortArray? {
        val record = audioRecord ?: return null
        val window = ShortArray(WINDOW_SAMPLES)
        var offset = 0
        while (offset < WINDOW_SAMPLES) {
            val n = record.read(window, offset, WINDOW_SAMPLES - offset)
            if (n <= 0) return null
            offset += n
        }
        return window
    }

    /** Idempotent: safe to call more than once, and safe to call even if [start] never ran. */
    fun release() {
        // Round-2 review fix: reset FIRST, unconditionally — this is the session boundary for
        // [pauseFailureWarned] regardless of whether there's a live `audioRecord` to actually
        // release (release() is documented idempotent/safe-to-call-with-nothing-to-release, and
        // the next arm's fresh session must be able to warn again on its own first pause failure).
        pauseFailureWarned = false
        val record = audioRecord ?: return
        audioRecord = null
        runCatching { record.stop() }
        record.release()
    }

    /**
     * P5-3: stops the underlying [AudioRecord] WITHOUT releasing it, so the platform stops
     * delivering (and powering) audio during a duty-cycle OFF phase while the object stays reusable.
     * Idempotent and fail-soft.
     *
     * HAZARD, read before touching the caller: [readWindow] returns `null` — which
     * [app.gauge.wear.control.SentinelController.tick] treats as MIC LOST and disarms on — for any
     * read that yields no samples, which is exactly what a stopped record does. Nothing may call
     * [readWindow] between [pause] and [resume]. [app.gauge.wear.service.SentinelService] guarantees
     * this by never invoking `tick()` on a non-capturing window at all, and by calling [resume]
     * unconditionally immediately before every capturing tick.
     *
     * Round-2 review fix (Important 3, P5-3): a [resume] failure is self-bounding — the very same
     * tick's [readWindow] then fails too, which auto-disarms and ends the session (at most one
     * warn ever gets logged before there's nothing left to fail). A [pause] failure has NO such
     * bound: this is called on every ON->OFF duty-cycle edge (~every 10s of quiet ARMED time) and
     * the duty-cycled branch returns *before* `tick()`, so nothing terminates the session — a
     * persistently-failing [AudioRecord.stop] would otherwise log ~6 warns/minute for a
     * potentially hours-long quiet ARMED stretch, evicting the 200-entry no-dedup telemetry ring
     * (this app's only on-device debugging channel) in well under an hour with nothing flushing in
     * the meantime. [pauseFailureWarned] closes that: warns once per failure stretch, suppresses
     * the rest, and says so in the one line it does log so a reader knows the true frequency.
     */
    fun pause() {
        val record = audioRecord ?: return
        runCatching { record.stop() }
            .onSuccess { pauseFailureWarned = false }
            .onFailure { e ->
                if (!pauseFailureWarned) {
                    pauseFailureWarned = true
                    diag.log(
                        "warn",
                        TAG,
                        "pause() failed: ${e.message} (further pause failures suppressed this session)",
                    )
                }
            }
    }

    /** Restarts a [pause]d record. Idempotent (checks `recordingState` first) and fail-soft — safe
     * to call before every capturing tick regardless of whether [pause] ever ran. A [resume]
     * failure is self-bounding (see [pause]'s own KDoc for the asymmetry) so it warns every time,
     * not once — there's at most one more window (this one) before the session ends anyway. */
    fun resume() {
        val record = audioRecord ?: return
        runCatching {
            if (record.recordingState != AudioRecord.RECORDSTATE_RECORDING) record.startRecording()
        }.onFailure { e -> diag.log("warn", TAG, "resume() failed: ${e.message}") }
    }

    companion object {
        const val SAMPLE_RATE_HZ = 16000
        const val WINDOW_SAMPLES = SAMPLE_RATE_HZ // exactly one second per window
        private const val BYTES_PER_SAMPLE = 2
        private const val TAG = "MicReader"
    }
}
