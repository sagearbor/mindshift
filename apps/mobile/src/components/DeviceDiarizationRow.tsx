/**
 * "Separate voices on this phone (engine B)" — the experimental row under a
 * stored recording's heat chart (ReplayScreen). Shown only when the owner
 * flipped Advanced → "Experimental voice engine" on; runs
 * live/deviceDiarization.ts without blocking the screen (progress + Cancel),
 * then draws the engine's segments as a strip in the app's speaker colours
 * with k, the timings, and the diagnostics id the run was posted under (so
 * the result can be scored against a per-second rubric with
 * scripts/diagnostics_tail.py --score-rubric). A failure is one honest line.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { useAuthStore } from "../store/authStore";
import { useDiagnosticsStore, type DeviceDiarizationEvent } from "../diagnostics/diagnostics";
import { loadExperimentalVoiceEngine } from "../live/experimentalPrefs";
import { runDeviceDiarization, DeviceDiarizationError, embeddingsLookDegenerate, type DeviceDiarizationProgress, type DeviceDiarizationRun } from "../live/deviceDiarization";
import { unknownLabel } from "../live/speakerId";
import { resolveSpeakerColors } from "../utils/speakerColors";

export interface DeviceDiarizationRowProps {
  recordingId: string;
  /** Force the visibility (tests / callers that already read the pref);
   *  undefined = read the per-account Advanced switch. */
  enabled?: boolean;
  /** The engine entry point (injectable for tests). */
  run?: typeof runDeviceDiarization;
}

type Phase =
  | { status: "idle" }
  | { status: "running"; progress: DeviceDiarizationProgress | null }
  | { status: "done"; event: DeviceDiarizationEvent; sent: { ok: boolean; id: string; error: string | null } | null }
  | { status: "error"; message: string };

const INK = "#1F2937";
const MUTED = "#6B7280";
const PRIMARY = "#4A90D9";
const DANGER = "#DC2626";

function phraseError(err: unknown): string {
  if (err instanceof DeviceDiarizationError) {
    if (err.code === "model-unavailable") return `Can’t run yet — ${err.message}. Start a live session once so the model downloads, then try again.`;
    if (err.code === "cancelled") return "Cancelled.";
    return `Didn’t finish — ${err.message}.`;
  }
  const msg = err instanceof Error ? err.message : String(err);
  return `Didn’t finish — ${msg}.`;
}

function fmtS(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

export function timingsLine(ev: DeviceDiarizationEvent): string {
  const embed = ev.embed_ms_mean === null ? "no windows" : `embed ${ev.embed_ms_mean} ms/window (p90 ${ev.embed_ms_p90})`;
  return (
    `download ${fmtS(ev.download_ms)} (${(ev.download_bytes / 1e6).toFixed(1)} MB) · ${embed} · ` +
    `cluster ${fmtS(ev.cluster_ms)} · total ${fmtS(ev.total_ms)} · ${ev.windows}/${ev.windows_total} windows @ ${ev.hop_s} s hop` +
    (ev.mean_pairwise_cosine === null ? "" : ` · mean cos ${ev.mean_pairwise_cosine}`)
  );
}

/** The one line a degenerate run gets instead of a voice count. */
export function degenerateWarning(ev: DeviceDiarizationEvent): string {
  return (
    `No result — all ${ev.windows} windows came back as the same voiceprint (mean cosine ${ev.mean_pairwise_cosine}). ` +
    "That is a model or audio problem on this phone, not one voice; the strip above means nothing."
  );
}

export default function DeviceDiarizationRow({ recordingId, enabled, run = runDeviceDiarization }: DeviceDiarizationRowProps) {
  const user = useAuthStore((s) => s.user);
  const uid = user?.uid ?? null;
  const email = user?.email ?? null;
  const sendDeviceDiarization = useDiagnosticsStore((s) => s.sendDeviceDiarization);
  const [prefOn, setPrefOn] = useState<boolean>(enabled ?? false);
  const [phase, setPhase] = useState<Phase>({ status: "idle" });
  const runRef = useRef<DeviceDiarizationRun | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      runRef.current?.cancel();
    };
  }, []);

  useEffect(() => {
    if (enabled !== undefined) {
      setPrefOn(enabled);
      return;
    }
    let cancelled = false;
    void loadExperimentalVoiceEngine(uid).then((on) => {
      if (!cancelled) setPrefOn(on);
    });
    return () => {
      cancelled = true;
    };
  }, [enabled, uid]);

  const start = useCallback(() => {
    if (runRef.current) return;
    setPhase({ status: "running", progress: null });
    const handle = run(recordingId, {
      onProgress: (progress) => {
        if (mountedRef.current) setPhase({ status: "running", progress });
      },
    });
    runRef.current = handle;
    void handle.promise
      .then(async (event) => {
        if (!mountedRef.current) return;
        setPhase({ status: "done", event, sent: null });
        const outcome = await sendDeviceDiarization(event, { uid, email });
        if (mountedRef.current) setPhase({ status: "done", event, sent: { ok: outcome.ok, id: outcome.id, error: outcome.ok ? null : outcome.error } });
      })
      .catch((err: unknown) => {
        if (mountedRef.current) setPhase({ status: "error", message: phraseError(err) });
      })
      .finally(() => {
        runRef.current = null;
      });
  }, [recordingId, run, sendDeviceDiarization, uid, email]);

  const cancel = useCallback(() => {
    runRef.current?.cancel();
  }, []);

  if (!prefOn) return null;

  const running = phase.status === "running";
  return (
    <View style={styles.card} testID="device-diarization-row">
      <Text style={styles.title}>Separate voices on this phone (engine B)</Text>
      <Text style={styles.sub}>
        Experimental — runs the transcript-free window engine over this recording’s audio here on the phone and shows what it hears. Nothing on the server changes.
      </Text>
      {phase.status === "idle" || phase.status === "error" || phase.status === "done" ? (
        <TouchableOpacity testID="device-diarization-run" accessibilityRole="button" style={styles.button} onPress={start}>
          <Text style={styles.buttonText}>{phase.status === "done" ? "Run again" : "Separate voices"}</Text>
        </TouchableOpacity>
      ) : null}
      {running ? (
        <View style={styles.progressRow} testID="device-diarization-progress">
          <ActivityIndicator size="small" color={PRIMARY} />
          <Text style={styles.progressText} numberOfLines={2}>
            {phase.progress
              ? `${phase.progress.detail}${phase.progress.fraction !== null ? ` · ${Math.round(phase.progress.fraction * 100)}%` : ""}`
              : "starting…"}
          </Text>
          <TouchableOpacity testID="device-diarization-cancel" accessibilityRole="button" onPress={cancel} style={styles.cancel}>
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      ) : null}
      {phase.status === "error" ? (
        <Text style={styles.error} testID="device-diarization-error">
          {phase.message}
        </Text>
      ) : null}
      {phase.status === "done" ? <Result event={phase.event} sent={phase.sent} /> : null}
    </View>
  );
}

function Result({ event, sent }: { event: DeviceDiarizationEvent; sent: { ok: boolean; id: string; error: string | null } | null }) {
  const labels = Array.from({ length: Math.max(event.k, 1) }, (_, i) => unknownLabel(i));
  const colors = resolveSpeakerColors(labels);
  const span = Math.max(event.duration_s, 0.001);
  const degenerate = embeddingsLookDegenerate(event);
  return (
    <View testID="device-diarization-result">
      <View style={styles.strip} testID="device-diarization-strip" accessibilityLabel={degenerate ? "no result — embeddings identical" : `${event.k} voices found`}>
        {event.segments.map(([s, e, l], i) => (
          <View
            key={`${i}-${s}`}
            testID={`device-diarization-seg-${i}`}
            style={[styles.seg, { flex: Math.max(e - s, 0.01) / span, backgroundColor: colors.get(unknownLabel(l)) ?? PRIMARY }]}
            accessibilityLabel={`${unknownLabel(l)} ${s.toFixed(1)}–${e.toFixed(1)} s`}
          />
        ))}
      </View>
      <View style={styles.legend}>
        {labels.map((lab) => (
          <View key={lab} style={styles.legendItem}>
            <View style={[styles.swatch, { backgroundColor: colors.get(lab) ?? PRIMARY }]} />
            <Text style={styles.legendText}>{lab}</Text>
          </View>
        ))}
      </View>
      {degenerate ? (
        <Text style={[styles.k, styles.error]} testID="device-diarization-warning">
          {degenerateWarning(event)}
        </Text>
      ) : (
        <Text style={styles.k} testID="device-diarization-k">
          {event.k} {event.k === 1 ? "voice" : "voices"} found (eigengap {event.k_eigengap}) · {event.segments.length} runs
        </Text>
      )}
      <Text style={styles.timings} testID="device-diarization-timings">
        {timingsLine(event)}
      </Text>
      <Text style={styles.model}>
        model {event.model_rev ? event.model_rev.slice(0, 8) : "?"} ({event.model_source ?? "?"}) · {event.device.platform} {event.device.model ?? ""}
      </Text>
      <Text style={[styles.sent, sent && !sent.ok ? styles.error : null]} testID="device-diarization-sent">
        {sent === null ? "Sending diagnostics…" : sent.ok ? `Sent · ID ${sent.id} — read this ID to Claude` : `Couldn’t send diagnostics (${sent.error ?? "unknown"}) · ID ${sent.id}`}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginTop: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  title: { fontSize: 15, fontWeight: "700", color: INK },
  sub: { fontSize: 12.5, color: MUTED, marginTop: 4, lineHeight: 17 },
  button: {
    marginTop: 10,
    alignSelf: "flex-start",
    backgroundColor: PRIMARY,
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  buttonText: { color: "#FFFFFF", fontWeight: "600", fontSize: 14 },
  progressRow: { flexDirection: "row", alignItems: "center", marginTop: 10, gap: 8 },
  progressText: { flex: 1, fontSize: 13, color: INK },
  cancel: { paddingHorizontal: 10, paddingVertical: 6 },
  cancelText: { color: DANGER, fontWeight: "600" },
  error: { marginTop: 8, color: DANGER, fontSize: 13 },
  strip: { flexDirection: "row", height: 14, borderRadius: 4, overflow: "hidden", marginTop: 12, backgroundColor: "#E5E7EB" },
  seg: { height: "100%" },
  legend: { flexDirection: "row", flexWrap: "wrap", marginTop: 6, gap: 10 },
  legendItem: { flexDirection: "row", alignItems: "center", gap: 4 },
  swatch: { width: 10, height: 10, borderRadius: 2 },
  legendText: { fontSize: 12, color: MUTED },
  k: { marginTop: 6, fontSize: 13.5, fontWeight: "600", color: INK },
  timings: { marginTop: 4, fontSize: 12.5, color: MUTED, lineHeight: 17 },
  model: { marginTop: 2, fontSize: 12, color: MUTED },
  sent: { marginTop: 6, fontSize: 12.5, color: "#15803D", fontWeight: "600" },
});
