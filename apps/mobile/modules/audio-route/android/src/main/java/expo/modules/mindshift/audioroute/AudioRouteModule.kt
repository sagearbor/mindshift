package expo.modules.mindshift.audioroute

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import java.util.concurrent.Executor

private const val ROUTE_CHANGE_EVENT = "onRouteChange"

/**
 * Reports the current audio OUTPUT route for Live Coach earpiece mode
 * (apps/mobile/src/live/audioRoute.ts classifies it). Read-only: it never
 * changes routing. Snapshot shape (see NativeRouteSnapshot in audioRoute.ts):
 *   platform      "android"
 *   outputs       every connected output (getDevices(GET_DEVICES_OUTPUTS))
 *   active        where media (TTS) plays now — getAudioDevicesForAttributes,
 *                 API 33+; null below
 *   communication the communication device, API 31+; null below
 *   mode          AudioManager.getMode() as a name
 */
class AudioRouteModule : Module() {
  private var deviceCallback: AudioDeviceCallback? = null
  private var commListener: Any? = null // AudioManager.OnCommunicationDeviceChangedListener (API 31+)
  private val mainHandler by lazy { Handler(Looper.getMainLooper()) }

  private val audioManager: AudioManager?
    get() = appContext.reactContext?.getSystemService(Context.AUDIO_SERVICE) as? AudioManager

  override fun definition() = ModuleDefinition {
    Name("MindShiftAudioRoute")

    Events(ROUTE_CHANGE_EVENT)

    Function("getRoute") {
      snapshot()
    }

    OnStartObserving(ROUTE_CHANGE_EVENT) {
      startWatching()
    }

    OnStopObserving(ROUTE_CHANGE_EVENT) {
      stopWatching()
    }

    OnDestroy {
      stopWatching()
    }
  }

  private fun emitChange() {
    try {
      sendEvent(ROUTE_CHANGE_EVENT, snapshot())
    } catch (_: Throwable) {
      // JS side keeps polling; a failed emit must never crash the app.
    }
  }

  private fun startWatching() {
    val am = audioManager ?: return
    if (deviceCallback == null) {
      val cb = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>?) = emitChange()
        override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>?) = emitChange()
      }
      am.registerAudioDeviceCallback(cb, mainHandler)
      deviceCallback = cb
    }
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && commListener == null) {
      val listener = AudioManager.OnCommunicationDeviceChangedListener { emitChange() }
      val executor = Executor { r -> mainHandler.post(r) }
      try {
        am.addOnCommunicationDeviceChangedListener(executor, listener)
        commListener = listener
      } catch (_: Throwable) {
        // Optional signal; the device callback + JS poll still cover it.
      }
    }
  }

  private fun stopWatching() {
    val am = audioManager
    deviceCallback?.let { cb ->
      try {
        am?.unregisterAudioDeviceCallback(cb)
      } catch (_: Throwable) {
      }
    }
    deviceCallback = null
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      (commListener as? AudioManager.OnCommunicationDeviceChangedListener)?.let { l ->
        try {
          am?.removeOnCommunicationDeviceChangedListener(l)
        } catch (_: Throwable) {
        }
      }
    }
    commListener = null
  }

  private fun snapshot(): Map<String, Any?> {
    val am = audioManager
      ?: return mapOf(
        "platform" to "android",
        "outputs" to null, // unreadable -> JS classifies "unknown"
        "active" to null,
        "communication" to null,
        "mode" to null
      )
    val outputs = am.getDevices(AudioManager.GET_DEVICES_OUTPUTS).map { typeName(it.type) }
    var active: List<String>? = null
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
      try {
        // expo-speech's TextToSpeech plays with the default TTS attributes:
        // media usage, speech content.
        val attrs = AudioAttributes.Builder()
          .setUsage(AudioAttributes.USAGE_MEDIA)
          .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
          .build()
        active = am.getAudioDevicesForAttributes(attrs).map { typeName(it.type) }
      } catch (_: Throwable) {
        active = null
      }
    }
    var communication: String? = null
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      communication = try {
        am.communicationDevice?.let { typeName(it.type) }
      } catch (_: Throwable) {
        null
      }
    }
    return mapOf(
      "platform" to "android",
      "outputs" to outputs,
      "active" to active,
      "communication" to communication,
      "mode" to modeName(am.mode)
    )
  }

  private fun modeName(mode: Int): String = when (mode) {
    AudioManager.MODE_NORMAL -> "normal"
    AudioManager.MODE_RINGTONE -> "ringtone"
    AudioManager.MODE_IN_CALL -> "call"
    AudioManager.MODE_IN_COMMUNICATION -> "communication"
    4 /* MODE_CALL_SCREENING, API 30 */ -> "call_screening"
    else -> "unknown"
  }

  /** AudioDeviceInfo.TYPE_* as its name without the prefix. Raw ints so the
   *  newer constants compile and resolve on any API level. */
  private fun typeName(type: Int): String = when (type) {
    1 -> "BUILTIN_EARPIECE"
    2 -> "BUILTIN_SPEAKER"
    3 -> "WIRED_HEADSET"
    4 -> "WIRED_HEADPHONES"
    5 -> "LINE_ANALOG"
    6 -> "LINE_DIGITAL"
    7 -> "BLUETOOTH_SCO"
    8 -> "BLUETOOTH_A2DP"
    9 -> "HDMI"
    10 -> "HDMI_ARC"
    11 -> "USB_DEVICE"
    12 -> "USB_ACCESSORY"
    13 -> "DOCK"
    14 -> "FM"
    15 -> "BUILTIN_MIC"
    16 -> "FM_TUNER"
    17 -> "TV_TUNER"
    18 -> "TELEPHONY"
    19 -> "AUX_LINE"
    20 -> "IP"
    21 -> "BUS"
    22 -> "USB_HEADSET"
    23 -> "HEARING_AID"
    24 -> "BUILTIN_SPEAKER_SAFE"
    25 -> "REMOTE_SUBMIX"
    26 -> "BLE_HEADSET"
    27 -> "BLE_SPEAKER"
    29 -> "HDMI_EARC"
    30 -> "BLE_BROADCAST"
    31 -> "DOCK_ANALOG"
    32 -> "MULTICHANNEL_GROUP"
    else -> "UNKNOWN"
  }
}
