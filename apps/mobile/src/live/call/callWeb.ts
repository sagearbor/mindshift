/**
 * RtcAdapter for the browser build (Mom's iPhone Safari) — the same
 * CallSession over the browser's own RTCPeerConnection.
 *
 * Microphone: the fast loop already holds a getUserMedia stream with echo
 * cancellation on (utils/webAudioCapture.ts); the call sends THAT track
 * (`getCaptureStream`) rather than opening a second one — Safari is happier
 * with one capture per page, and it keeps what the coach hears and what
 * the other person hears identical. Falls back to its own getUserMedia
 * when no capture is running (e.g. a browser whose AudioWorklet failed).
 *
 * Remote audio: an <audio autoplay playsinline> element. Safari only lets
 * it play if the page had a user gesture: `prime()` creates the element
 * and calls play() inside the Answer tap (#152's gesture findings), so
 * the later srcObject assignment isn't blocked.
 */
import type { RtcAdapter, RtcPeerLike, RtcStreamLike, RtcTrackLike } from "./rtc";
import type { IceServer } from "./types";

export interface WebRtcAdapterOptions {
  /** The fast loop's live MediaStream, when capturing. */
  getCaptureStream?: () => MediaStream | null;
  /** Test seams. */
  peerCtor?: new (config: { iceServers: IceServer[] }) => unknown;
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  createAudioElement?: () => HTMLAudioElement | null;
}

export function webRtcSupported(g: Record<string, unknown> = globalThis as Record<string, unknown>): boolean {
  return typeof g.RTCPeerConnection === "function";
}

export function createWebRtcAdapter(options: WebRtcAdapterOptions = {}): RtcAdapter {
  // One <audio> element per peer uid: a mesh mixes every remote participant,
  // and each element can be primed independently inside the Answer tap.
  const elements = new Map<string, HTMLAudioElement>();
  let primed = false;
  const g = globalThis as Record<string, unknown>;
  const PeerCtor =
    options.peerCtor ??
    (g.RTCPeerConnection as new (config: { iceServers: IceServer[] }) => unknown | undefined);
  const getUserMedia =
    options.getUserMedia ??
    ((c: MediaStreamConstraints) =>
      typeof navigator !== "undefined" && navigator.mediaDevices
        ? navigator.mediaDevices.getUserMedia(c)
        : Promise.reject(new Error("getUserMedia unavailable")));
  const createAudioElement =
    options.createAudioElement ??
    (() => {
      if (typeof document === "undefined") return null;
      const el = document.createElement("audio");
      el.autoplay = true;
      el.setAttribute("playsinline", "true");
      el.setAttribute("aria-hidden", "true");
      el.style.display = "none";
      document.body?.appendChild(el);
      return el;
    });

  const ensureAudio = (peerUid: string): HTMLAudioElement | null => {
    const existing = elements.get(peerUid);
    if (existing) return existing;
    const el = createAudioElement();
    if (el) {
      elements.set(peerUid, el);
      // A peer that appears after the priming tap still needs its element
      // kicked once, so its first srcObject assignment isn't blocked.
      if (primed) void el.play?.().catch(() => {});
    }
    return el;
  };

  return {
    createPeer(config) {
      if (typeof PeerCtor !== "function") throw new Error("this browser has no WebRTC (RTCPeerConnection)");
      return new PeerCtor({ iceServers: config.iceServers }) as RtcPeerLike;
    },
    async getLocalStream() {
      const shared = options.getCaptureStream?.() ?? null;
      if (shared && shared.getAudioTracks().length > 0) return shared as unknown as RtcStreamLike;
      const own = await getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
        video: false,
      });
      return own as unknown as RtcStreamLike;
    },
    playRemote(peerUid: string, stream: RtcStreamLike | null, track: RtcTrackLike) {
      const el = ensureAudio(peerUid);
      if (!el) return;
      let src: MediaStream | null = stream as unknown as MediaStream | null;
      if (!src && typeof MediaStream === "function") {
        src = new MediaStream([track as unknown as MediaStreamTrack]);
      }
      if (!src) return;
      el.srcObject = src;
      void el.play?.().catch(() => {
        // Autoplay refused (no gesture yet): the element stays armed and
        // plays on the next tap — Safari retries play() on interaction.
      });
    },
    stopRemote(peerUid?: string) {
      const targets = peerUid ? [peerUid] : [...elements.keys()];
      for (const uid of targets) {
        const el = elements.get(uid);
        if (!el) continue;
        elements.delete(uid);
        try {
          el.pause?.();
          el.srcObject = null;
          el.remove?.();
        } catch {
          // Element already gone.
        }
      }
    },
    prime() {
      // Unlock a pooled element inside the gesture; peers that arrive later
      // reuse the `primed` flag so their fresh element is kicked too.
      primed = true;
      const el = ensureAudio("__prime__");
      if (el) void el.play?.().catch(() => {});
      elements.delete("__prime__");
      try {
        el?.remove?.();
      } catch {
        // ignore
      }
    },
  };
}
