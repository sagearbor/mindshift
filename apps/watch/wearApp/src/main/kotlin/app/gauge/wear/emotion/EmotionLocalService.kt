package app.gauge.wear.emotion

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.BatteryManager
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import app.gauge.wear.BuildConfig
import app.gauge.wear.R
import app.gauge.wear.auth.AccountPrefs
import app.gauge.wear.control.DiagLog
import app.gauge.wear.haptics.HapticDirector
import app.gauge.wear.haptics.RealVibratorPort
import app.gauge.wear.net.ApiResult
import app.gauge.wear.net.WatchApiClient
import app.gauge.wear.prefs.GaugePrefs
import app.gauge.wear.telemetry.Telemetry
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * On-watch emotion mode (Phase 2, docs/plans/2026-11-10-emotion-on-watch-phase2.md): the TINY
 * Whisper-encoder model runs on the wrist, with no phone and no server in the loop. Off by default;
 * runs only while the Settings toggle ([GaugePrefs.emotionLocalEnabled]) is on.
 *
 * It owns its own AudioRecord on its own thread, deliberately separate from [app.gauge.wear.service.
 * SentinelService]: that loop is duty-cycled, one-second-windowed and shared by many consumers, and
 * a 6 s / 3 s hop pipeline would have to bend all of it. Two readers of the mic in one app is legal
 * on Android 10+; whether a given watch's audio HAL keeps both fed while the sentinel is armed is
 * UNVERIFIED (no hardware reachable when this was written) - the status line says "mic lost" if not.
 *
 * Privacy: audio never leaves the process or touches disk. Only a [WindowRow] of numbers is stored
 * ([WindowLog]) and, when signed in, uploaded for the phone's Emotion dev log.
 */
class EmotionLocalService : Service() {
    private var thread: Thread? = null
    private val stopFlag = AtomicBoolean(false)
    private var syncScope: CoroutineScope? = null
    private val syncing = AtomicBoolean(false)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!GaugePrefs.emotionLocalEnabled(this) || !hasMicPermission(this)) {
            LocalEmotionBus.update { it.copy(running = false, text = if (hasMicPermission(this)) "Off" else "Mic permission needed") }
            stopSelf()
            return START_NOT_STICKY
        }
        startForeground(NOTIFICATION_ID, notification("Listening for emotion on this watch"), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        if (thread?.isAlive != true) {
            stopFlag.set(false)
            syncScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
            thread = Thread({ loop() }, "EmotionLocalLoop").also { it.start() }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopFlag.set(true)
        thread?.join(2_000)
        thread = null
        syncScope?.cancel()
        syncScope = null
        LocalEmotionBus.update { it.copy(running = false, text = if (GaugePrefs.emotionLocalEnabled(this)) it.text else "Off") }
        super.onDestroy()
    }

    // ---- the loop (own thread) ------------------------------------------------------------------

    @SuppressLint("MissingPermission")
    private fun loop() {
        val diag = DiagLog { l, t, m -> runCatching { Telemetry.log(l, t, m) } }
        val ctx = applicationContext
        var found = findTinyModel(modelDirs(ctx))
        if (found is ModelLookup.Missing) {
            // No sideloaded file: fetch the pinned, sha256-verified copy from Hosting into the same folder.
            LocalEmotionBus.update { it.copy(running = true, text = "Downloading model...") }
            var lastPct = -1
            val r = ModelDownloader().download(modelDirs(ctx).first()) { bytes ->
                val pct = (bytes * 100 / 9_942_329L).toInt().coerceAtMost(99) / 5 * 5
                if (pct != lastPct) {
                    lastPct = pct
                    LocalEmotionBus.update { it.copy(text = "Downloading model... $pct%") }
                }
            }
            if (r is DownloadResult.Failed) {
                LocalEmotionBus.update { it.copy(running = false, text = "Model download failed: ${r.reason}. Toggle off/on to retry, or adb push.") }
                diag.log("warn", TAG, "model download failed: ${r.reason}")
                stopSelf()
                return
            }
            found = findTinyModel(modelDirs(ctx))
        }
        if (found is ModelLookup.Missing) {
            LocalEmotionBus.update { it.copy(running = false, text = "Model missing: ${found.reason}") }
            diag.log("warn", TAG, "model missing: ${found.reason}")
            stopSelf()
            return
        }
        val engine = try {
            OnnxTinyEngine.open((found as ModelLookup.Ready).file)
        } catch (t: Throwable) {
            LocalEmotionBus.update { it.copy(running = false, text = "Model failed to load: ${t.message}") }
            diag.log("error", TAG, "model load failed: ${t.message}")
            stopSelf()
            return
        }
        val log = WindowLog(File(ctx.filesDir, "emotion"))
        val haptics = HapticDirector(RealVibratorPort(ctx), diag = diag)
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val runner = LocalEmotionRunner(
            engine = engine,
            log = log,
            buzz = { haptics.playEmotionNudge() },
            batteryPercent = { runCatching { bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) }.getOrDefault(-1) },
        )

        val chunk = LocalEmotion.SAMPLE_RATE / 2 // 0.5 s per read
        val minBuf = AudioRecord.getMinBufferSize(LocalEmotion.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        // 4 s of headroom: an inference (tens of ms, hundreds on a slow watch) must never overrun the mic buffer.
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC, LocalEmotion.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, LocalEmotion.SAMPLE_RATE * 2 * 4),
        )
        try {
            check(record.state == AudioRecord.STATE_INITIALIZED) { "AudioRecord failed to initialize" }
            record.startRecording()
        } catch (t: Throwable) {
            record.release()
            engine.close()
            LocalEmotionBus.update { it.copy(running = false, text = "Mic unavailable: ${t.message}") }
            stopSelf()
            return
        }

        LocalEmotionBus.update { LocalEmotionStatus(running = true, text = "Listening", unsynced = log.unsent(Int.MAX_VALUE).size) }
        val buf = ShortArray(chunk)
        var lastTick = System.currentTimeMillis()
        var lastSync = lastTick
        var samplesSinceTick = 0
        try {
            while (!stopFlag.get()) {
                val n = record.read(buf, 0, chunk)
                if (n <= 0) {
                    LocalEmotionBus.update { it.copy(running = false, text = "Mic lost") }
                    diag.log("warn", TAG, "mic read returned $n; stopping")
                    break
                }
                runner.onPcm(buf, n)
                samplesSinceTick += n
                if (samplesSinceTick >= LocalEmotion.SAMPLE_RATE * LocalEmotion.HOP_S) {
                    samplesSinceTick = 0
                    val now = System.currentTimeMillis()
                    val tick = runner.tick(now)
                    lastTick = now
                    if (tick is LocalEmotionRunner.Tick.Failed) diag.log("warn", TAG, "classify failed: ${tick.error}")
                    publish(runner, tick, log)
                    if (now - lastSync >= SYNC_EVERY_MS) {
                        lastSync = now
                        syncAsync(log)
                    }
                }
            }
        } catch (t: Throwable) {
            diag.log("error", TAG, "loop failed: ${t.message}")
            LocalEmotionBus.update { it.copy(running = false, text = "Stopped: ${t.message}") }
        } finally {
            runCatching { record.stop() }
            record.release()
            engine.close()
            syncAsync(log)
        }
    }

    private fun publish(runner: LocalEmotionRunner, tick: LocalEmotionRunner.Tick, log: WindowLog) {
        LocalEmotionBus.update { s ->
            val last = (tick as? LocalEmotionRunner.Tick.Classified)?.row
            s.copy(
                windows = runner.windows,
                skippedSilent = runner.skippedSilent,
                nudges = runner.nudges,
                avgMs = runner.avgMs,
                maxMs = runner.maxMs,
                lastLabel = last?.let { LocalEmotion.LABELS[topClass(it.p.map { v -> v.toFloat() }.toFloatArray())] } ?: s.lastLabel,
                lastProb = last?.p?.maxOrNull() ?: s.lastProb,
                unsynced = log.unsent(Int.MAX_VALUE).size,
                text = if (tick is LocalEmotionRunner.Tick.Failed) "Classify error: ${tick.error}" else "Listening",
            )
        }
    }

    /** Upload unsent rows to the server queue the phone drains. Fail-soft; never blocks the mic loop. */
    private fun syncAsync(log: WindowLog) {
        val scope = syncScope ?: return
        if (!syncing.compareAndSet(false, true)) return
        scope.launch {
            try {
                val ctx = applicationContext
                val api = WatchApiClient(BuildConfig.GAUGE_API_BASE, { AccountPrefs.deviceToken(ctx) })
                while (true) {
                    val rows = log.unsent()
                    if (rows.isEmpty()) break
                    when (api.postEmotionWindows(WindowLog.syncBody(rows))) {
                        is ApiResult.Ok -> {
                            log.markSent(rows.last().t)
                            LocalEmotionBus.update { it.copy(lastSyncOk = true, unsynced = log.unsent(Int.MAX_VALUE).size) }
                        }
                        is ApiResult.Failure -> {
                            LocalEmotionBus.update { it.copy(lastSyncOk = false) }
                            break
                        }
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "sync failed: ${t.message}")
            } finally {
                syncing.set(false)
            }
        }
    }

    private fun notification(text: String): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL_ID, "On-watch emotion", NotificationManager.IMPORTANCE_LOW))
        }
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val TAG = "EmotionLocalService"
        private const val CHANNEL_ID = "emotion_local"
        private const val NOTIFICATION_ID = 2
        private const val SYNC_EVERY_MS = 5 * 60_000L

        fun hasMicPermission(ctx: Context): Boolean =
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

        /** Where the sideloaded model is looked for: the app's external files folder (adb push
         *  works with no root), then its private files folder (`adb shell run-as`, debug builds). */
        fun modelDirs(ctx: Context): List<File> = listOfNotNull(
            ctx.getExternalFilesDir(null)?.let { File(it, "emotion") },
            File(ctx.filesDir, "emotion"),
        )

        /** Make the service match the toggle. Safe to call from a foreground activity at any time. */
        fun apply(ctx: Context) {
            val i = Intent(ctx, EmotionLocalService::class.java)
            if (GaugePrefs.emotionLocalEnabled(ctx) && hasMicPermission(ctx)) ContextCompat.startForegroundService(ctx, i)
            else {
                ctx.stopService(i)
                if (!GaugePrefs.emotionLocalEnabled(ctx)) LocalEmotionBus.update { LocalEmotionStatus() }
            }
        }
    }
}
