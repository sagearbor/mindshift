import React from "react";
import { View, Text, ActivityIndicator, StyleSheet } from "react-native";
import type { PreflightState } from "../hooks/useAudioStream";
import type { VoicePerson } from "../api/liveSessions";
import { iceProbeOk, type IceProbeResult } from "../live/call/iceProbe";
import { useDevModeStore } from "../store/devModeStore";

interface Props {
  /** Can this device run the on-device loop at all (on-device STT)? */
  liveCapable: boolean;
  liveCapabilityReason: string;
  /** Whether the next session will use the on-device loop (vs the server). */
  liveMode: boolean;
  preflight: PreflightState | null;
  /** Enrolled people expected in this session; null while loading. */
  people: VoicePerson[] | null;
  peopleError: string | null;
  /** Call mode only: what the ICE connectivity probe found (null = not run
   *  yet), and whether it is running right now. Off in the other modes —
   *  an in-person session has no peer connection to check. */
  isCall?: boolean;
  iceProbe?: IceProbeResult | null;
  iceProbing?: boolean;
}

/** "cloud" when every local provider is absent; else the first local name. */
export function describeLlm(chain: string[] | undefined): string {
  if (!chain || chain.length === 0) return "cloud";
  const local = chain.filter((p) => p !== "cloud");
  if (local.length === 0) return "cloud";
  return local.length === 1 ? local[0] : `${local[0]} → ${local.slice(1).join(" → ")}`;
}

function Row({
  testID,
  ok,
  label,
  detail,
}: {
  testID: string;
  ok: boolean | null;
  label: string;
  detail: string;
}) {
  const mark = ok === null ? "…" : ok ? "✓" : "✗";
  const color = ok === null ? "#9CA3AF" : ok ? "#15803D" : "#B45309";
  return (
    <View style={styles.row} testID={testID}>
      <Text style={[styles.mark, { color }]}>{mark}</Text>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={styles.rowDetail} numberOfLines={2}>
        {detail}
      </Text>
    </View>
  );
}

/**
 * The pre-session capability check, shown honestly: what will actually
 * run on this phone (on-device speech, speaker-ID with the reason it's off,
 * the local LLM provider or "cloud", the VAD) and who the loop expects to
 * hear (the account's enrolled people). Nothing here is a promise — it is
 * what the probe loaded a moment ago.
 */
export default function LivePreflightPanel({
  liveCapable,
  liveCapabilityReason,
  liveMode,
  preflight,
  people,
  peopleError,
  isCall = false,
  iceProbe = null,
  iceProbing = false,
}: Props) {
  const caps = preflight?.status === "ready" ? preflight.capabilities : null;
  const probing = preflight?.status === "probing";
  const onDevice = liveCapable && liveMode;
  // Developer mode off = just "who's here" + a plain readiness line; the
  // capability rows (models, providers, VAD) are the owner's instrument.
  const devMode = useDevModeStore((s) => s.devMode);
  return (
    <View style={styles.card} testID="live-preflight">
      <Text style={styles.title}>Before you start</Text>
      {!devMode ? (
        <Text style={styles.rowDetail} testID="preflight-plain">
          {probing ? "Getting ready…" : "Ready when you are — tap Start and speak first."}
        </Text>
      ) : null}
      {devMode ? (
      <>
      <Row
        testID="preflight-stt"
        ok={liveCapable}
        label="On-device speech"
        detail={liveCapable ? (liveMode ? "ready" : "off — using the server") : liveCapabilityReason}
      />
      <Row
        testID="preflight-speaker-id"
        ok={!onDevice ? false : caps ? caps.speakerId.active : probing ? null : false}
        label="Speaker-ID"
        detail={
          !onDevice
            ? "server labels voices by speaking order"
            : caps
              ? caps.speakerId.active
                ? `${caps.speakerId.enrolled} enrolled · model ${caps.speakerId.model ?? "?"}`
                : caps.speakerId.reason
              : probing
                ? "checking…"
                : preflight?.status === "failed"
                  ? preflight.reason
                  : "not checked yet"
        }
      />
      <Row
        testID="preflight-llm"
        ok={!onDevice ? true : caps ? true : probing ? null : false}
        label="Suggestions"
        detail={
          !onDevice
            ? "cloud"
            : caps
              ? describeLlm(caps.llm)
              : probing
                ? "checking…"
                : "not checked yet"
        }
      />
      </>
      ) : null}
      {/* Call mode only: can these two phones actually reach each other?
          One honest line from a real ICE gathering run against the server's
          own ice_servers — "relay needed — no TURN configured" is the
          answer that saves a demo, so it is never hidden. */}
      {isCall ? (
        <Row
          testID="preflight-peer-connection"
          ok={iceProbing ? null : iceProbeOk(iceProbe)}
          label="Peer connection"
          detail={
            iceProbing
              ? "checking…"
              : iceProbe
                ? iceProbe.line
                : "not checked yet"
          }
        />
      ) : null}
      {devMode && onDevice && caps ? (
        <Row
          testID="preflight-vad"
          ok={caps.vad === "silero"}
          label="Turn detection"
          detail={caps.vad === "silero" ? "Silero VAD" : "energy VAD (Silero unavailable)"}
        />
      ) : null}
      {probing ? (
        <View style={styles.probing}>
          <ActivityIndicator size="small" color="#4A90D9" />
          <Text style={styles.probingText}>Loading models…</Text>
        </View>
      ) : null}

      <Text style={styles.subTitle}>Who&apos;s here</Text>
      <View style={styles.peopleRow} testID="whos-here">
        {people === null ? (
          <Text style={styles.peopleHint}>Loading enrolled people…</Text>
        ) : people.length === 0 ? (
          <Text style={styles.peopleHint} testID="whos-here-empty">
            {peopleError
              ? `Couldn’t load enrolled people (${peopleError}).`
              : "Nobody enrolled yet — voices will be labelled by speaking order. Enroll yourself and the people you talk with in Settings → Voice."}
          </Text>
        ) : (
          people.map((p) => (
            <View
              key={p.personId}
              style={[styles.personChip, p.isSelf && styles.personChipSelf]}
              testID={`whos-here-${p.personId}`}
            >
              <Text style={[styles.personText, p.isSelf && styles.personTextSelf]}>
                {p.displayName}
              </Text>
            </View>
          ))
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginHorizontal: 16,
    marginVertical: 6,
    padding: 12,
    gap: 6,
  },
  title: {
    fontSize: 13,
    fontWeight: "700",
    color: "#374151",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  subTitle: {
    marginTop: 6,
    fontSize: 13,
    fontWeight: "700",
    color: "#374151",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
  },
  mark: {
    width: 16,
    fontSize: 14,
    fontWeight: "700",
  },
  rowLabel: {
    width: 118,
    fontSize: 13.5,
    fontWeight: "600",
    color: "#1F2937",
  },
  rowDetail: {
    flex: 1,
    fontSize: 12.5,
    lineHeight: 17,
    color: "#6B7280",
  },
  probing: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  probingText: {
    fontSize: 12,
    color: "#6B7280",
  },
  peopleRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
  },
  peopleHint: {
    fontSize: 12.5,
    lineHeight: 17,
    color: "#6B7280",
    flex: 1,
  },
  personChip: {
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  personChipSelf: {
    borderColor: "#4A90D9",
    backgroundColor: "#EFF6FF",
  },
  personText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#374151",
  },
  personTextSelf: {
    color: "#4A90D9",
  },
});
