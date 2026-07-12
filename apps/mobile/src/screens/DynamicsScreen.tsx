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
import { postAnalyze } from "../api/client";
import type { AnalyzeResult, AnalyzePerSpeaker } from "../api/client";
import HeatChart from "../components/HeatChart";
import { getSpeakerColor } from "../utils/speakerColors";

// House colors.
const PRIMARY = "#4A90D9";
const INK = "#1F2937";
const MUTED = "#6B7280";
const AMBER = "#F59E0B";

interface DynamicsScreenProps {
  onBack: () => void;
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
export default function DynamicsScreen({ onBack }: DynamicsScreenProps) {
  // Snapshot turns once from the store — the analysis is of a fixed transcript,
  // so we don't want it re-fetching if the store mutates underneath us.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalyzeResult | null>(null);
  // The transcript the analysis was run against, kept so the HeatChart inspector
  // can show each turn's actual words (the backend's per_turn carries no text).
  const [analyzedTurns, setAnalyzedTurns] = useState<
    { speaker: string; text: string }[]
  >([]);

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
      setAnalyzedTurns(turns.map((t) => ({ speaker: t.speaker, text: t.text })));
      // Send utterance timestamps when the transcript has them (live diarized
      // sessions do) — that's what lets the server compute REAL interruption
      // counts. Pasted transcripts carry none; the server then returns
      // interruptions: null and the UI omits the row honestly.
      const result = await postAnalyze(
        turns.map((t) => ({
          speaker: t.speaker,
          text: t.text,
          ...(t.start_time !== undefined ? { start_time: t.start_time } : {}),
          ...(t.end_time !== undefined ? { end_time: t.end_time } : {}),
        })),
      );
      if (mountedRef.current) setData(result);
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
    void runAnalyze();
  }, [runAnalyze]);

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
          {/* Heat chart across the whole conversation. */}
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Heat over the conversation</Text>
            <HeatChart perTurn={data.per_turn} turns={analyzedTurns} />
          </View>

          {/* Per-speaker stat cards, shown side by side. Never a winner. */}
          <View style={styles.speakerRow}>
            {Object.entries(data.per_speaker).map(([label, stats]) => (
              <SpeakerCard key={label} label={label} stats={stats} />
            ))}
          </View>

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
                    “{trg.phrase}” — {trg.speaker}
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
                    <Text style={styles.requestWho}>{req.speaker}</Text>
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
}: {
  label: string;
  stats: AnalyzePerSpeaker;
}) {
  const color = getSpeakerColor(label);
  return (
    <View style={styles.speakerCard} testID={`speaker-card-${label}`}>
      <View style={styles.speakerCardHeader}>
        <View style={[styles.swatch, { backgroundColor: color }]} />
        <Text style={styles.speakerName}>{label}</Text>
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
