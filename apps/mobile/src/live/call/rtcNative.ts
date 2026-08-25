/**
 * RtcAdapter over react-native-webrtc (the Pixel).
 *
 * Loaded lazily and only off-web: react-native-webrtc's index wires a
 * NativeEventEmitter at import time, which throws without the native
 * module (a build that predates it, Jest, the web bundle). Everything
 * degrades to a reason in `CallView.error` — never a crash at app start.
 *
 * Microphone: `mediaDevices.getUserMedia` with echo cancellation ON, 16 kHz
 * mono. This is a SECOND AudioRecord next to expo-audio's (the coaching
 * loop keeps feeding the fast loop from that one — react-native-webrtc has
 * no API to tap its track's PCM). Android allows concurrent capture within
 * one app, so both hear the room; documented in the plan.
 *
 * Remote audio: react-native-webrtc plays remote audio tracks itself; the
 * output route (loudspeaker vs earpiece) is set through expo-audio's audio
 * mode (utils/audioMode.ts), speaker by default.
 */
import { Platform } from "react-native";
import { setCallAudioRoute } from "../../utils/audioMode";
import type { AudioRoute, RtcAdapter, RtcPeerLike, RtcStreamLike } from "./rtc";
import type { IceServer } from "./types";

interface WebRtcModuleLike {
  RTCPeerConnection: new (config: { iceServers: IceServer[] }) => unknown;
  mediaDevices: { getUserMedia(constraints: unknown): Promise<unknown> };
}

let cached: WebRtcModuleLike | null | undefined;

/** The native module, or null with the reason it can't be used here. */
export function loadNativeWebRtc(): { module: WebRtcModuleLike | null; reason: string | null } {
  if (Platform.OS === "web") return { module: null, reason: "not the native build" };
  if (cached !== undefined) return { module: cached, reason: cached ? null : "react-native-webrtc unavailable" };
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("react-native-webrtc") as WebRtcModuleLike;
    if (!mod || typeof mod.RTCPeerConnection !== "function") throw new Error("no RTCPeerConnection export");
    cached = mod;
    return { module: mod, reason: null };
  } catch (err) {
    cached = null;
    return {
      module: null,
      reason: `react-native-webrtc isn't in this build (${err instanceof Error ? err.message : String(err)}) — install the build that has it`,
    };
  }
}

export function createNativeRtcAdapter(): RtcAdapter {
  const loaded = loadNativeWebRtc();
  return {
    createPeer(config) {
      if (!loaded.module) throw new Error(loaded.reason ?? "WebRTC unavailable");
      return new loaded.module.RTCPeerConnection({ iceServers: config.iceServers }) as RtcPeerLike;
    },
    async getLocalStream() {
      if (!loaded.module) throw new Error(loaded.reason ?? "WebRTC unavailable");
      const stream = await loaded.module.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,
          channelCount: 1,
        },
        video: false,
      });
      return stream as RtcStreamLike;
    },
    playRemote() {
      // react-native-webrtc renders every remote audio track without a view,
      // so a mesh already mixes all participants; nothing to wire per peer.
    },
    stopRemote() {
      // Nothing to release: playback stops when the peer connection closes.
    },
    async setRoute(route: AudioRoute) {
      await setCallAudioRoute(route);
    },
  };
}
