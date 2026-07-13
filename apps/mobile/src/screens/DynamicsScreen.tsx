import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
} from "react-native";
import { useSessionStore } from "../store/sessionStore";
import { postAnalyze, postCounterfactual } from "../api/client";
import type {
  AnalyzeResult,
  AnalyzePerSpeaker,
  AnalyzeTurnInput,
  CounterfactualResult,
  ReportCard,
  UploadAnalyzeResult,
} from "../api/client";
import HeatChart from "../components/HeatChart";
import { getSpeakerColor } from "../utils/speakerColors";
import { speakerLabel, type SpeakerLabels } from "../utils/speakerLabels";

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const AMBER = "#F59E0B";

interface DynamicsScreenProps {
  onBack: () => void;
  /**
   * A ready-made analysis from the recording-upload flow. When provided the
   * screen renders it directly and never calls /analyze on mount — the
   * transcript already lives in the session store (loaded from the upload's
   * turns), so the what-if / counterfactual path keeps working off it as usual.
   * Absent for the normal button-driven flow, which fetches on mount.
   */
  initialData?: AnalyzeResult;
  /**
   * The id of a *stored* recording that backs this analysis, passed from the
   * upload flow when consent+store both landed as true; null/undefined
   * otherwise. When absent, the screen falls back to the recording_id on its
   * own fetched UploadAnalyzeResult. Either source enables the "Replay
   * recording" entry point that opens the synced media replay.
   */
  recordingId?: string | null;
  /** True when this analysis came from a just-recorded in-app video (saved to
   *  the camera roll). Gates the "attach HD source later" popup — only shown for
   *  recorder-origin analyses that were actually stored. */
  cameFromRecorder?: boolean;
  /** Open the replay for the backing recording. Wired by App; the button only
   *  shows when an effective recording id and this handler are both present. */
  onReplay?: (recordingId: string) => void;
  /** Jump straight to the attach-HD-source flow (replay with its input open)
   *  for the backing recording. Wired by App; used by the HD-suggest popup. */
  onAttachSource?: (recordingId: string) => void;
}

/**
 * Post-session "Conversation Dynamics" analysis. Reads the current transcript
 * from the session store, POSTs it to /analyze on mount, and renders the
 * result: a per-speaker heat chart, speaker stat cards, relationship-dynamics
 * insights, and a narrative. On any failure it shows an honest error state with
 * a retry — never a fabricated result (house rule).
 *
 * Framing rule enforced throughout: there is no "winner". All speakers' stats
 * are always shown together with neutral labels; we never rank or single one out.
 */
export default function DynamicsScreen({
  onBack,
  initialData,
  recordingId,
  cameFromRecorder,
  onReplay,
  onAttachSource,
}: DynamicsScreenProps) {
  // The HD-later suggestion is dismissible; once the user taps "Later" (or acts
  // on it) it stays hidden for this analysis.
  const [hdPopupDismissed, setHdPopupDismissed] = useState(false);
  // Snapshot turns once from the store — the analysis is of a fixed transcript,
  // so we don't want it re-fetching if the store mutates underneath us.
  // When `initialData` is supplied (upload flow) we start already-loaded, so
  // there's no spinner and no on-mount fetch.
  const [loading, setLoading] = useState(!initialData);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalyzeResult | null>(initialData ?? null);
  // The transcript the analysis was run against, kept so the HeatChart inspector
  // can show each turn's actual words (the backend's per_turn carries no text).
  // Seed it from the store immediately when we start with data in hand.
  // Keep optional utterance timing alongside text: a live diarized / uploaded
  // conversation carries start/end times, which the HeatChart uses to draw its
  // time axis (dashes over real seconds). A pasted transcript has none, and the
  // chart falls back to even spacing.
  const [analyzedTurns, setAnalyzedTurns] = useState<
    { speaker: string; text: string; start_time?: number; end_time?: number }[]
  >(() =>
    initialData
      ? useSessionStore.getState().turns.map((t) => ({
          speaker: t.speaker,
          text: t.text,
          ...(t.start_time !== undefined ? { start_time: t.start_time } : {}),
          ...(t.end_time !== undefined ? { end_time: t.end_time } : {}),
        }))
      : [],
  );

  // --- What-if / counterfactual overlay state ---
  const [simData, setSimData] = useState<CounterfactualResult | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  // The pivot the current in-flight/errored/active request pertains to. Drives
  // which turn's inspector shows the spinner/error and anchors the overlay.
  const [simPivot, setSimPivot] = useState<number | null>(null);
  // Toggle the overlay without refetching. Defaults on when a sim arrives.
  const [showSim, setShowSim] = useState(true);
  // The exact turns payload sent to /analyze (incl. timing) — reused verbatim
  // for /analyze/counterfactual so the server sees the same conversation.
  const analyzedPayloadRef = useRef<AnalyzeTurnInput[]>([]);
  // In-flight guard for the counterfactual: a real, costed LLM call — one at a
  // time, and never a duplicate from a double-tap.
  const simInFlightRef = useRef(false);

  // In-flight guard: /analyze is a real LLM call with real cost, and React 18
  // StrictMode double-invokes the mount effect in dev — without this ref a
  // single visit would fire TWO requests. A ref (not state) so the second
  // invocation is rejected synchronously, before any await.
  const inFlightRef = useRef(false);
  // Unmount guard: the request can outlive the screen (user backs out while
  // analyzing); state sets after unmount are dropped.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const runAnalyze = useCallback(async () => {
    if (inFlightRef.current) return; // A request is already pending — one is enough.
    inFlightRef.current = true;
    setLoading(true);
    setError(null);
    try {
      const turns = useSessionStore.getState().turns;
      setAnalyzedTurns(
        turns.map((t) => ({
          speaker: t.speaker,
          text: t.text,
          ...(t.start_time !== undefined ? { start_time: t.start_time } : {}),
          ...(t.end_time !== undefined ? { end_time: t.end_time } : {}),
        })),
      );
      // Send utterance timestamps when the transcript has them (live diarized
      // sessions do) — that's what lets the server compute REAL interruption
      // counts. Pasted transcripts carry none; the server then returns
      // interruptions: null and the UI omits the row honestly.
      const payload: AnalyzeTurnInput[] = turns.map((t) => ({
        speaker: t.speaker,
        text: t.text,
        ...(t.start_time !== undefined ? { start_time: t.start_time } : {}),
        ...(t.end_time !== undefined ? { end_time: t.end_time } : {}),
      }));
      // Keep the exact payload so a later counterfactual sends the SAME turns.
      analyzedPayloadRef.current = payload;
      const result = await postAnalyze(payload);
      if (mountedRef.current) {
        setData(result);
        // A fresh analysis invalidates any prior simulation.
        setSimData(null);
        setSimError(null);
        setSimPivot(null);
      }
    } catch (e) {
      // Surface the status honestly; the message already reads "API error: 429"
      // etc. from the client. Never fall back to a fake analysis.
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : "Something went wrong.");
      }
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (initialData) {
      // Upload flow: analysis is already in hand. Skip the fetch entirely, but
      // seed the counterfactual payload from the store transcript so "what if"
      // still sends the same turns (with the upload's real utterance timing).
      analyzedPayloadRef.current = useSessionStore.getState().turns.map((t) => ({
        speaker: t.speaker,
        text: t.text,
        ...(t.start_time !== undefined ? { start_time: t.start_time } : {}),
        ...(t.end_time !== undefined ? { end_time: t.end_time } : {}),
      }));
      return;
    }
    void runAnalyze();
  }, [runAnalyze, initialData]);

  // Run a counterfactual for a given pivot turn. Replaces any prior overlay on
  // success (one simulation at a time). Honest error state on failure; never a
  // fabricated projection.
  const runCounterfactual = useCallback(async (pivotIndex: number) => {
    if (simInFlightRef.current) return; // One costed LLM call at a time.
    simInFlightRef.current = true;
    setSimPivot(pivotIndex);
    setSimLoading(true);
    setSimError(null);
    try {
      const result = await postCounterfactual(
        analyzedPayloadRef.current,
        pivotIndex,
      );
      if (mountedRef.current) {
        setSimData(result); // Replaces the previous overlay outright.
        setShowSim(true);
      }
    } catch (e) {
      if (mountedRef.current) {
        setSimError(e instanceof Error ? e.message : "Something went wrong.");
      }
    } finally {
      simInFlightRef.current = false;
      if (mountedRef.current) setSimLoading(false);
    }
  }, []);

  // Speaker whose turn was rewritten — its color accents the "What if" card.
  const pivotSpeaker =
    simData !== null ? analyzedTurns[simData.pivot_index]?.speaker ?? "" : "";
  // §3 — per-speaker display labels from the analysis (name → deeper/higher
  // voice → raw id). Absent on old/pre-labels analyses; the speakerLabel helper
  // then falls back to the raw speaker id, so nothing regresses.
  const speakerLabels: SpeakerLabels = data?.speaker_labels;
  // The overlay is visible only when we have a sim AND the toggle is on.
  const overlayVisible = simData !== null && showSim;
  // Prosody-unavailable note, present only on results from /analyze/upload.
  const voiceAnalysis = (data as UploadAnalyzeResult | null)?.voice_analysis;
  // The recording backing this analysis, from either source: the id handed in
  // by the upload flow, or — when we fetched the analysis ourselves — the
  // recording_id the server returned on an UploadAnalyzeResult. Either enables
  // the Replay entry point below.
  const effectiveRecordingId =
    recordingId ?? (data as UploadAnalyzeResult | null)?.recording_id ?? null;
  // The "attach HD source later" popup only makes sense when this analysis came
  // from an in-app recording (in the camera roll) that was actually STORED
  // server-side (a recording id means stored) — and only until dismissed.
  const showHdPopup =
    !!cameFromRecorder &&
    !!effectiveRecordingId &&
    !!onAttachSource &&
    !hdPopupDismissed;

  return (
    <View style={styles.flex}>
      {/* Header with back to the Session tab. Mirrors SessionDetail. */}
      <View style={styles.header}>
        <TouchableOpacity testID="dynamics-back" onPress={onBack}>
          <Text style={styles.backText}>‹ Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Conversation Dynamics</Text>
        <View style={styles.headerSpacer} />
      </View>

      {loading && (
        <View style={styles.centered} testID="dynamics-loading">
          <ActivityIndicator size="large" color={PRIMARY} />
          <Text style={styles.centeredText}>Reading the conversation…</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centered} testID="dynamics-error">
          <Text style={styles.errorTitle}>Couldn’t analyze this one</Text>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity
            testID="dynamics-retry"
            style={styles.retryButton}
            onPress={() => void runAnalyze()}
          >
            <Text style={styles.retryText}>Try again</Text>
          </TouchableOpacity>
        </View>
      )}

      {!loading && !error && data && (
        <ScrollView
          style={styles.flex}
          contentContainerStyle={styles.content}
          testID="dynamics-content"
        >
          {/* §1 — the conversation's title (user-provided, else the LLM's short
              auto-title). Shown only when the analysis carries one; old analyses
              omit it and this simply doesn't render. */}
          {data.title ? (
            <Text style={styles.analysisTitle} testID="analysis-title">
              {data.title}
            </Text>
          ) : null}

          {/* HD-later suggestion — after analyzing an in-app recording, nudge the
              user to attach a share link once their camera-roll clip backs up to
              the cloud, unlocking HD replay. Dismissible. */}
          {showHdPopup && effectiveRecordingId && (
            <View style={styles.hdPopup} testID="hd-suggest-popup">
              <Text style={styles.hdPopupTitle}>Want HD replay later?</Text>
              <Text style={styles.hdPopupBody}>
                Your video is in your camera roll. Once it backs up to your cloud
                (e.g. Google Photos), share a single-item link and attach it to
                this recording for full-quality replay.
              </Text>
              <View style={styles.hdPopupButtons}>
                <TouchableOpacity
                  testID="hd-suggest-later"
                  style={styles.hdPopupLater}
                  onPress={() => setHdPopupDismissed(true)}
                >
                  <Text style={styles.hdPopupLaterText}>Later</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  testID="hd-suggest-attach"
                  style={styles.hdPopupAttach}
                  onPress={() => {
                    setHdPopupDismissed(true);
                    onAttachSource?.(effectiveRecordingId);
                  }}
                >
                  <Text style={styles.hdPopupAttachText}>Attach link now</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* Replay entry point — only when this analysis is backed by a stored
              recording (from the upload flow or our own fetched result) and App
              wired up navigation. */}
          {effectiveRecordingId && onReplay && (
            <TouchableOpacity
              testID="replay-recording-button"
              style={styles.replayButton}
              onPress={() => onReplay(effectiveRecordingId)}
            >
              <Text style={styles.replayButtonText}>▶ Replay recording</Text>
            </TouchableOpacity>
          )}

          {/* Heat chart across the whole conversation. */}
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Heat over the conversation</Text>
            <HeatChart
              perTurn={data.per_turn}
              turns={analyzedTurns}
              speakerLabels={speakerLabels}
              // Draw the time axis only when EVERY turn has real timing (a
              // diarized/uploaded conversation). A pasted transcript has none, so
              // we omit turnsTiming and the chart falls back to even spacing —
              // never a fabricated timeline.
              turnsTiming={
                analyzedTurns.length > 0 &&
                analyzedTurns.every(
                  (t) => t.start_time !== undefined && t.end_time !== undefined,
                )
                  ? analyzedTurns.map((t) => ({
                      start_time: t.start_time as number,
                      end_time: t.end_time as number,
                    }))
                  : undefined
              }
              simulated={simData?.simulated_per_turn ?? null}
              showSimulation={showSim}
              onToggleSimulation={() => setShowSim((s) => !s)}
              onWhatIf={(pivotIndex) => void runCounterfactual(pivotIndex)}
              whatIfLoading={simLoading}
              whatIfPivotIndex={simPivot}
              whatIfError={simError}
              onRetryWhatIf={() =>
                simPivot !== null && void runCounterfactual(simPivot)
              }
            />

            {/* Honest prosody note from /analyze/upload: shown only when the
                server told us voice analysis was unavailable (degraded audio,
                etc.). Small and muted — never a claim we couldn't back up. */}
            {voiceAnalysis && (
              <Text style={styles.voiceNote} testID="voice-analysis-note">
                {voiceAnalysis}
              </Text>
            )}

            {/* "What if" card — the rewritten pivot, its rationale, and the
                server's disclaimer verbatim. Only while an overlay is shown. */}
            {overlayVisible && simData && (
              <View style={styles.whatIfCard} testID="what-if-card">
                <Text style={styles.whatIfCardTitle}>
                  What if{" "}
                  {pivotSpeaker
                    ? speakerLabel(pivotSpeaker, speakerLabels)
                    : "this"}{" "}
                  had said…
                </Text>
                <View
                  style={[
                    styles.whatIfQuote,
                    { borderLeftColor: getSpeakerColor(pivotSpeaker) },
                  ]}
                >
                  <Text style={styles.whatIfQuoteText}>
                    “{simData.rewritten_text}”
                  </Text>
                </View>
                <Text style={styles.whatIfRationale}>{simData.rationale}</Text>
                <Text style={styles.whatIfDisclaimer}>{simData.disclaimer}</Text>
              </View>
            )}
          </View>

          {/* Per-speaker stat cards, shown side by side. Never a winner. */}
          <View style={styles.speakerRow}>
            {Object.entries(data.per_speaker).map(([label, stats]) => (
              <SpeakerCard
                key={label}
                label={label}
                stats={stats}
                speakerLabels={speakerLabels}
              />
            ))}
          </View>

          {/* Report cards — one per speaker, directly under the stat cards.
              Scores are an absolute, intentionally-comparable conduct grade
              (owner's product decision); rendered plainly, no softening. */}
          {data.report_cards &&
            Object.keys(data.report_cards).length > 0 && (
              <View style={styles.card}>
                <Text style={styles.sectionTitle}>Report cards</Text>
                <View style={styles.speakerRow}>
                  {Object.entries(data.report_cards).map(([label, card]) => (
                    <ReportCardView
                      key={label}
                      label={label}
                      card={card}
                      speakerLabels={speakerLabels}
                    />
                  ))}
                </View>
              </View>
            )}

          {/* Relationship dynamics insights. */}
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Dynamics</Text>

            <InsightRow
              title="Coupling"
              body={data.dynamics.coupling.description}
            />
            <InsightRow
              title="De-escalation"
              body={data.dynamics.deescalation.description}
            />

            {data.dynamics.triggers.length > 0 && (
              <View style={styles.subBlock}>
                <Text style={styles.subTitle}>Triggers</Text>
                {data.dynamics.triggers.map((trg, i) => (
                  <Text key={i} style={styles.triggerLine}>
                    “{trg.phrase}” — {speakerLabel(trg.speaker, speakerLabels)}
                    {"  "}
                    {/* The server also emits triggers for COOLING phrases, so
                        heat_delta can be negative — a naive "+{delta}" would
                        render "+-10". Differentiate the copy by direction and
                        use a real minus sign for the cooling case. */}
                    <Text
                      style={
                        trg.heat_delta >= 0
                          ? styles.triggerDelta
                          : styles.triggerDeltaCool
                      }
                    >
                      {trg.heat_delta >= 0
                        ? `sparked +${trg.heat_delta} heat`
                        : `cooled −${Math.abs(trg.heat_delta)} heat`}
                    </Text>
                  </Text>
                ))}
              </View>
            )}

            {data.dynamics.requests.length > 0 && (
              <View style={styles.subBlock}>
                <Text style={styles.subTitle}>Requests &amp; outcomes</Text>
                {data.dynamics.requests.map((req, i) => (
                  <View key={i} style={styles.requestRow}>
                    <Text style={styles.requestWho}>
                      {speakerLabel(req.speaker, speakerLabels)}
                    </Text>
                    <Text style={styles.requestText}>
                      {req.request}
                      {"  "}
                      <Text style={styles.requestOutcome}>→ {req.outcome}</Text>
                    </Text>
                  </View>
                ))}
              </View>
            )}
          </View>

          {/* Narrative — the quiet "third chair" perspective. */}
          <View style={[styles.card, styles.narrativeCard]}>
            <Text style={styles.narrativeTitle}>The third chair</Text>
            <Text style={styles.narrativeText} testID="dynamics-narrative">
              {data.narrative}
            </Text>
          </View>

          {/* Ethics footer. */}
          <Text style={styles.footer}>
            Analyze only conversations everyone knows about.
          </Text>
        </ScrollView>
      )}
    </View>
  );
}

/** One speaker's stats. Neutral, never ranked against the other. */
function SpeakerCard({
  label,
  stats,
  speakerLabels,
}: {
  label: string;
  stats: AnalyzePerSpeaker;
  speakerLabels?: SpeakerLabels;
}) {
  const color = getSpeakerColor(label);
  return (
    <View style={styles.speakerCard} testID={`speaker-card-${label}`}>
      <View style={styles.speakerCardHeader}>
        <View style={[styles.swatch, { backgroundColor: color }]} />
        <Text style={styles.speakerName}>
          {speakerLabel(label, speakerLabels)}
        </Text>
      </View>

      <StatLine label="Avg heat" value={String(stats.avg_heat)} />
      <StatLine label="Peak" value={String(stats.peak_heat)} />
      <StatLine
        label="Talk share"
        value={`${Math.round(stats.talk_share * 100)}%`}
      />
      {/* Omit interruptions entirely when null — never show a fabricated 0. */}
      {stats.interruptions !== null && (
        <StatLine label="Interruptions" value={String(stats.interruptions)} />
      )}
      <StatLine
        label="Repairs"
        value={`${stats.repairs_accepted}/${stats.repair_attempts}`}
      />

      {/* Four horsemen as small labeled chips. */}
      <View style={styles.horsemenRow}>
        <HorsemanChip label="crit" value={stats.horsemen.criticism} />
        <HorsemanChip label="cont" value={stats.horsemen.contempt} />
        <HorsemanChip label="def" value={stats.horsemen.defensiveness} />
        <HorsemanChip label="stone" value={stats.horsemen.stonewalling} />
      </View>
    </View>
  );
}

/**
 * One speaker's "report card". The score is an absolute 0–100 conduct grade
 * (higher = better) and is intentionally comparable across speakers — so it's
 * shown plainly, big, with no hedging. The speaker's line color accents the
 * card so it ties back to the chart and stat card.
 */
function ReportCardView({
  label,
  card,
  speakerLabels,
}: {
  label: string;
  card: ReportCard;
  speakerLabels?: SpeakerLabels;
}) {
  const color = getSpeakerColor(label);
  return (
    <View
      style={[styles.reportCard, { borderTopColor: color }]}
      testID={`report-card-${label}`}
    >
      <View style={styles.reportCardHeader}>
        <View style={[styles.swatch, { backgroundColor: color }]} />
        <Text style={styles.speakerName}>
          {speakerLabel(label, speakerLabels)}
        </Text>
      </View>
      <View style={styles.scoreRow}>
        <Text style={[styles.scoreNumber, { color }]}>{card.score}</Text>
        <Text style={styles.scoreOutOf}>/100</Text>
      </View>
      <Text style={styles.reportHeadline}>{card.headline}</Text>
      <View style={styles.reportLine}>
        <Text style={styles.reportLabelGood}>Did well: </Text>
        <Text style={styles.reportBody}>{card.did_well}</Text>
      </View>
      <View style={styles.reportLine}>
        <Text style={styles.reportLabelWork}>Work on: </Text>
        <Text style={styles.reportBody}>{card.work_on}</Text>
      </View>
    </View>
  );
}

function StatLine({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.statLine}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

function HorsemanChip({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.horsemanChip}>
      <Text style={styles.horsemanValue}>{value}</Text>
      <Text style={styles.horsemanLabel}>{label}</Text>
    </View>
  );
}

function InsightRow({ title, body }: { title: string; body: string }) {
  return (
    <View style={styles.subBlock}>
      <Text style={styles.subTitle}>{title}</Text>
      <Text style={styles.insightBody}>{body}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#F9FAFB" },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingTop: 56,
    paddingBottom: 12,
    paddingHorizontal: 16,
    backgroundColor: "#FFFFFF",
    borderBottomWidth: 1,
    borderBottomColor: "#E5E7EB",
  },
  backText: { fontSize: 16, color: PRIMARY, fontWeight: "600", width: 64 },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: INK,
  },
  headerSpacer: { width: 64 },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  centeredText: { marginTop: 12, color: MUTED, fontSize: 14 },
  errorTitle: { fontSize: 18, fontWeight: "700", color: INK, marginBottom: 6 },
  errorText: { fontSize: 14, color: MUTED, textAlign: "center", marginBottom: 16 },
  retryButton: {
    backgroundColor: PRIMARY,
    paddingVertical: 12,
    paddingHorizontal: 28,
    borderRadius: 10,
  },
  retryText: { color: "#FFFFFF", fontSize: 15, fontWeight: "600" },
  content: { padding: 16, paddingBottom: 40 },
  replayButton: {
    marginBottom: 16,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    backgroundColor: PRIMARY,
  },
  replayButtonText: { color: "#FFFFFF", fontSize: 16, fontWeight: "700" },
  hdPopup: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
    backgroundColor: "#EAF3FC",
    borderWidth: 1,
    borderColor: PRIMARY,
  },
  hdPopupTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: INK,
    marginBottom: 6,
  },
  hdPopupBody: {
    fontSize: 13,
    lineHeight: 19,
    color: "#374151",
    marginBottom: 12,
  },
  hdPopupButtons: { flexDirection: "row", gap: 10, justifyContent: "flex-end" },
  hdPopupLater: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  hdPopupLaterText: { color: MUTED, fontSize: 14, fontWeight: "600" },
  hdPopupAttach: {
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 8,
    backgroundColor: PRIMARY,
  },
  hdPopupAttachText: { color: "#FFFFFF", fontSize: 14, fontWeight: "600" },
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: INK,
    marginBottom: 12,
  },
  analysisTitle: {
    fontSize: 20,
    fontWeight: "800",
    color: INK,
    marginBottom: 4,
  },
  speakerRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 16,
  },
  speakerCard: {
    flex: 1,
    minWidth: 150,
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  speakerCardHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  swatch: { width: 12, height: 12, borderRadius: 3 },
  speakerName: { fontSize: 15, fontWeight: "700", color: INK },
  statLine: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 3,
  },
  statLabel: { fontSize: 13, color: MUTED },
  statValue: { fontSize: 13, fontWeight: "600", color: INK },
  horsemenRow: {
    flexDirection: "row",
    gap: 6,
    marginTop: 10,
  },
  horsemanChip: {
    flex: 1,
    alignItems: "center",
    backgroundColor: "#F3F4F6",
    borderRadius: 8,
    paddingVertical: 5,
  },
  horsemanValue: { fontSize: 14, fontWeight: "700", color: INK },
  horsemanLabel: { fontSize: 10, color: MUTED, marginTop: 1 },
  subBlock: { marginBottom: 14 },
  subTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    marginBottom: 4,
  },
  insightBody: { fontSize: 14, lineHeight: 20, color: INK },
  triggerLine: { fontSize: 14, lineHeight: 21, color: INK, marginBottom: 3 },
  triggerDelta: { color: AMBER, fontWeight: "700" },
  // Cooling triggers get the calm green from the speaker palette, not alarm amber.
  triggerDeltaCool: { color: "#10B981", fontWeight: "700" },
  requestRow: { marginBottom: 8 },
  requestWho: { fontSize: 12, fontWeight: "700", color: MUTED },
  requestText: { fontSize: 14, lineHeight: 20, color: INK },
  requestOutcome: { color: PRIMARY, fontWeight: "600" },
  // --- Report cards ---
  reportCard: {
    flex: 1,
    minWidth: 150,
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    // Accent bar in the speaker's color, tying the card to the chart line.
    borderTopWidth: 4,
  },
  reportCardHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 6,
  },
  scoreRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    marginBottom: 6,
  },
  scoreNumber: { fontSize: 40, fontWeight: "800", lineHeight: 44 },
  scoreOutOf: { fontSize: 16, fontWeight: "600", color: MUTED, marginBottom: 6 },
  reportHeadline: {
    fontSize: 14,
    fontWeight: "700",
    color: INK,
    marginBottom: 8,
  },
  reportLine: { marginBottom: 6 },
  reportLabelGood: { fontSize: 13, fontWeight: "700", color: "#10B981" },
  reportLabelWork: { fontSize: 13, fontWeight: "700", color: AMBER },
  reportBody: { fontSize: 13, lineHeight: 19, color: "#374151" },
  // --- What-if card (below the chart) ---
  whatIfCard: {
    marginTop: 14,
    padding: 14,
    borderRadius: 10,
    backgroundColor: "#F9FAFB",
    borderWidth: 1,
    borderColor: "#E5E7EB",
  },
  whatIfCardTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: MUTED,
    textTransform: "uppercase",
    marginBottom: 8,
  },
  whatIfQuote: {
    borderLeftWidth: 3,
    paddingLeft: 10,
    marginBottom: 10,
  },
  whatIfQuoteText: {
    fontSize: 15,
    lineHeight: 22,
    color: INK,
    fontStyle: "italic",
  },
  whatIfRationale: {
    fontSize: 14,
    lineHeight: 20,
    color: "#374151",
    marginBottom: 8,
  },
  whatIfDisclaimer: {
    fontSize: 11,
    lineHeight: 16,
    color: "#9CA3AF",
    fontStyle: "italic",
  },
  voiceNote: {
    marginTop: 10,
    fontSize: 12,
    lineHeight: 17,
    color: "#9CA3AF",
    fontStyle: "italic",
  },
  narrativeCard: { backgroundColor: "#F3F4F6", borderColor: "#E5E7EB" },
  narrativeTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: INK,
    marginBottom: 8,
  },
  narrativeText: { fontSize: 15, lineHeight: 23, color: "#374151" },
  footer: {
    fontSize: 12,
    color: "#9CA3AF",
    textAlign: "center",
    fontStyle: "italic",
    marginTop: 4,
  },
});
