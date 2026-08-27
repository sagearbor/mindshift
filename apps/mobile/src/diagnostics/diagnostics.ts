/**
 * "Send diagnostics" — the phone's own account of what happened, POSTed to
 * the server's existing `/telemetry` channel (server/watch/routers/
 * telemetry.py) as one `client_diagnostics` event, so the owner can read
 * the diagnostics id off the screen and an agent can pull the record with
 * `scripts/diagnostics_tail.py` — no adb, no screenshots.
 *
 * What goes in (all facts the app already has, nothing new is measured):
 *   - the last pre-flight capability probe (VAD / speaker-ID / LLM chain)
 *   - the last session: mode, turn count, latency summary (median / p90 of
 *     segment-end→speak, per-provider counts), STT restarts + failure,
 *     WebSocket reconnects, mic error, live status line, POST outcome,
 *     the in-app call's outcome
 *   - app version / build / runtime / OTA update id / channel
 *   - platform, OS version, device model (or the browser's user agent)
 *
 * Never throws: a diagnostics send that fails must not become one more
 * failure to diagnose — the outcome says what happened.
 */
import { Platform } from "react-native";
import * as Application from "expo-application";
import * as Updates from "expo-updates";
import Constants from "expo-constants";
import { create } from "zustand";
import type { FastLoopCapabilities } from "../live/defaultDeps";
import type { TurnLatency } from "../live/fastLoop";
import { authHeaders } from "../api/liveSessions";

const API_URL = process.env.EXPO_PUBLIC_API_URL || "http://localhost:8000";

export const DIAGNOSTICS_TAG = "client_diagnostics";

/** Crockford base32 without the look-alikes: readable over the phone. */
const ID_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789";

/** `dx-XXXX-XXXX` — 8 symbols from a 30-letter alphabet (~39 bits). */
export function newDiagnosticsId(random: () => number = Math.random): string {
  let out = "";
  for (let i = 0; i < 8; i++) {
    out += ID_ALPHABET[Math.floor(random() * ID_ALPHABET.length) % ID_ALPHABET.length];
    if (i === 3) out += "-";
  }
  return `dx-${out}`;
}

export const DIAGNOSTICS_ID_RE = /^dx-[A-Z2-9]{4}-[A-Z2-9]{4}$/;

export interface LatencySummary {
  turns: number;
  spoken: number;
  medianToSpeakMs: number | null;
  p90ToSpeakMs: number | null;
  medianLlmMs: number | null;
  medianSttWaitMs: number | null;
  /** Which provider answered how many turns ("os", "bundled", "cloud", …). */
  byProvider: Record<string, number>;
  /** Per-provider attempt outcomes across all turns, e.g. {"os:refused": 12,
   *  "bundled:unavailable": 12}. Answers WHY a local provider isn't carrying
   *  turns (Gemini Nano refusing coaching text vs timing out vs unavailable). */
  byOutcome: Record<string, number>;
  /** One representative error/refusal detail per "provider:outcome", so the
   *  actual on-device error text is visible (e.g. the Gemini Nano exception),
   *  not just the count. Truncated. */
  outcomeSamples: Record<string, string>;
  held: number;
}

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  const idx = Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)));
  return Math.round(sorted[idx]);
}

export function summarizeLatency(log: readonly TurnLatency[]): LatencySummary {
  const toSpeak = log.filter((l) => l.toSpeakMs !== null).map((l) => l.toSpeakMs as number).sort((a, b) => a - b);
  const llm = log.map((l) => l.llmMs).sort((a, b) => a - b);
  const stt = log.map((l) => l.sttWaitMs).sort((a, b) => a - b);
  const byProvider: Record<string, number> = {};
  for (const l of log) byProvider[l.provider] = (byProvider[l.provider] ?? 0) + 1;
  const byOutcome: Record<string, number> = {};
  const outcomeSamples: Record<string, string> = {};
  for (const l of log) {
    for (const a of l.attempts ?? []) {
      const key = `${a.provider}:${a.outcome}`;
      byOutcome[key] = (byOutcome[key] ?? 0) + 1;
      if (a.detail && !outcomeSamples[key]) outcomeSamples[key] = a.detail.slice(0, 200);
    }
  }
  return {
    turns: log.length,
    spoken: toSpeak.length,
    medianToSpeakMs: percentile(toSpeak, 0.5),
    p90ToSpeakMs: percentile(toSpeak, 0.9),
    medianLlmMs: percentile(llm, 0.5),
    medianSttWaitMs: percentile(stt, 0.5),
    byProvider,
    byOutcome,
    outcomeSamples,
    held: log.filter((l) => l.held).length,
  };
}

export interface SessionDiagnostics {
  sessionId: string;
  mode: string;
  startedAt: string | null;
  endedAt: string;
  turns: number;
  latency: LatencySummary;
  /** What the loop said it loaded (the status line), or why it didn't run. */
  liveStatus: string;
  onDevice: boolean;
  sttRestarts: number | null;
  sttFailure: string | null;
  wsReconnects: number;
  micError: string | null;
  transcriptionMessage: string | null;
  /** POST /sessions/live outcome. */
  postStatus: "created" | "unsupported" | "failed" | "none";
  call: { status: string; iceRestarts: number; error: string | null; connectedSeconds: number | null } | null;
  /** Every problem worth a diagnostics send, in plain words. */
  errors: string[];
}

export interface DeviceInfo {
  platform: string;
  osVersion: string | null;
  model: string | null;
  userAgent: string | null;
}

export function collectDeviceInfo(
  platform: { OS: string; Version?: unknown; constants?: unknown } = Platform,
  nav: { userAgent?: string } | null = typeof navigator !== "undefined" ? navigator : null,
): DeviceInfo {
  const c = (platform.constants ?? {}) as Record<string, unknown>;
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  const model =
    str(c.Model) ??
    (str(c.Brand) || str(c.Manufacturer) ? `${str(c.Brand) ?? str(c.Manufacturer)}` : null) ??
    str(c.systemName) ??
    null;
  return {
    platform: platform.OS,
    osVersion:
      platform.Version !== undefined && platform.Version !== null
        ? String(platform.Version)
        : str(c.osVersion) ?? str(c.Release) ?? null,
    model,
    userAgent: platform.OS === "web" ? nav?.userAgent ?? null : null,
  };
}

export interface AppInfo {
  version: string | null;
  build: string | null;
  runtimeVersion: string | null;
  updateId: string | null;
  channel: string | null;
  otaEnabled: boolean;
}

export function collectAppInfo(): AppInfo {
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  let updates: Partial<typeof Updates> = {};
  try {
    updates = Updates;
  } catch {
    updates = {};
  }
  return {
    version: str(Application.nativeApplicationVersion) ?? str(Constants.expoConfig?.version) ?? null,
    build:
      str(Application.nativeBuildVersion) ??
      (Constants.expoConfig?.android?.versionCode != null ? String(Constants.expoConfig.android.versionCode) : null),
    runtimeVersion: str(updates.runtimeVersion) ?? null,
    updateId: str(updates.updateId) ?? null,
    channel: str(updates.channel) ?? null,
    otaEnabled: updates.isEnabled === true && Platform.OS !== "web",
  };
}

export interface DiagnosticsPayload {
  diagnostics_id: string;
  created_at: string;
  trigger: "manual" | "auto";
  uid: string | null;
  email: string | null;
  app: AppInfo;
  device: DeviceInfo;
  capability: FastLoopCapabilities | null;
  capability_reason: string | null;
  last_session: SessionDiagnostics | null;
}

export interface BuildPayloadInput {
  trigger: "manual" | "auto";
  uid: string | null;
  email: string | null;
  capability: FastLoopCapabilities | null;
  capabilityReason: string | null;
  lastSession: SessionDiagnostics | null;
  id?: string;
  now?: () => Date;
}

export function buildDiagnosticsPayload(input: BuildPayloadInput): DiagnosticsPayload {
  return {
    diagnostics_id: input.id ?? newDiagnosticsId(),
    created_at: (input.now ?? (() => new Date()))().toISOString(),
    trigger: input.trigger,
    uid: input.uid,
    email: input.email,
    app: collectAppInfo(),
    device: collectDeviceInfo(),
    capability: input.capability,
    capability_reason: input.capabilityReason,
    last_session: input.lastSession,
  };
}

/** The `/telemetry` body one payload becomes (server/watch/routers/telemetry.py). */
export function telemetryBody(payload: DiagnosticsPayload): {
  device: string;
  app_version: string;
  events: { level: string; tag: string; message: string; stack: null; ts: string; data: DiagnosticsPayload }[];
} {
  const errors = payload.last_session?.errors ?? [];
  return {
    device: `phone:${payload.device.platform}:${payload.uid ?? "anon"}`,
    app_version: payload.app.version ?? "unknown",
    events: [
      {
        level: errors.length > 0 ? "warn" : "info",
        tag: DIAGNOSTICS_TAG,
        message:
          `${DIAGNOSTICS_TAG} ${payload.diagnostics_id} uid=${payload.uid ?? "-"} email=${payload.email ?? "-"}` +
          (errors.length > 0 ? ` errors=${errors.join(" | ")}` : ""),
        stack: null,
        ts: payload.created_at,
        data: payload,
      },
    ],
  };
}

export type SendOutcome = { ok: true; id: string } | { ok: false; id: string; error: string };

export async function sendDiagnostics(
  payload: DiagnosticsPayload,
  fetchImpl: typeof fetch = (...args) => fetch(...args),
): Promise<SendOutcome> {
  try {
    const res = await fetchImpl(`${API_URL}/telemetry`, {
      method: "POST",
      headers: await authHeaders(),
      body: JSON.stringify(telemetryBody(payload)),
    });
    if (!res.ok) return { ok: false, id: payload.diagnostics_id, error: `server answered ${res.status}` };
    return { ok: true, id: payload.diagnostics_id };
  } catch (err) {
    return { ok: false, id: payload.diagnostics_id, error: err instanceof Error ? err.message : String(err) };
  }
}

// --- the store the hook writes and Settings reads ----------------------------

export interface DiagnosticsState {
  capability: FastLoopCapabilities | null;
  capabilityReason: string | null;
  lastSession: SessionDiagnostics | null;
  sending: boolean;
  lastSent: { id: string; at: string; trigger: "manual" | "auto"; ok: boolean; error: string | null } | null;
  setCapability: (capability: FastLoopCapabilities | null, reason: string | null) => void;
  recordSession: (session: SessionDiagnostics) => void;
  /** Build + POST. Resolves with the outcome; also kept in `lastSent`. */
  send: (trigger: "manual" | "auto", who: { uid: string | null; email: string | null }) => Promise<SendOutcome>;
}

export const useDiagnosticsStore = create<DiagnosticsState>((set, get) => ({
  capability: null,
  capabilityReason: null,
  lastSession: null,
  sending: false,
  lastSent: null,
  setCapability: (capability, reason) => set({ capability, capabilityReason: reason }),
  recordSession: (session) => set({ lastSession: session }),
  send: async (trigger, who) => {
    const state = get();
    const payload = buildDiagnosticsPayload({
      trigger,
      uid: who.uid,
      email: who.email,
      capability: state.capability,
      capabilityReason: state.capabilityReason,
      lastSession: state.lastSession,
    });
    set({ sending: true });
    const outcome = await sendDiagnostics(payload);
    set({
      sending: false,
      lastSent: {
        id: outcome.id,
        at: payload.created_at,
        trigger,
        ok: outcome.ok,
        error: outcome.ok ? null : outcome.error,
      },
    });
    return outcome;
  },
}));
