import React, { useCallback, useEffect, useState } from "react";
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Platform } from "react-native";
import type { CallPeer, CallRole, CallView } from "../live/call/types";
import type { AudioRoute } from "../live/call/rtc";
import { shareInvite, type InviteOutcome } from "../live/call/invite";
import { callDeepLink, callWebUrl, isJoinCode } from "../nav/callLink";

/** The one line that explains why Call mode exists (also in the runbook). */
export const CALL_MODE_EXPLAINER =
  "Your phone can't listen during a normal phone call — MindShift places the call itself.";

interface CallPanelProps {
  call: CallView;
  sessionActive: boolean;
  /** A code that arrived through an invite link: show "Answer" instead of
   *  the start/join choices (the tap must do everything — Safari). */
  invitedCode?: string | null;
  /** The role that invite link encodes (participant / therapist). */
  invitedRole?: CallRole;
  onStart: () => void;
  onJoin: (code: string) => void;
  onAnswer?: (code: string) => void;
  onHangUp: () => void;
  onToggleMute: () => void;
  route?: AudioRoute;
  onToggleRoute?: () => void;
  /** Wall clock for the timer (tests). */
  now?: () => number;
}

export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function callStatusLabel(status: CallView["status"]): string {
  switch (status) {
    case "creating":
      return "setting up";
    case "waiting":
      return "waiting for them";
    case "connecting":
      return "connecting";
    case "connected":
      return "connected";
    case "reconnecting":
      return "reconnecting";
    case "ended":
      return "ended";
    case "failed":
      return "failed";
    default:
      return "";
  }
}

function peerStatus(peer: CallPeer): string {
  const base = peer.connected ? "connected" : "connecting";
  return peer.role === "therapist" ? `${base} · therapist` : base;
}

/**
 * Call mode's own controls, above the transcript: start / join / answer
 * while idle; the in-call header (self · call status · timer), one row per
 * other participant (up to two — a dad + a therapist), the invite while
 * waiting, and mute / route / hang up once the call exists.
 */
export default function CallPanel({
  call,
  sessionActive,
  invitedCode = null,
  invitedRole = "participant",
  onStart,
  onJoin,
  onAnswer,
  onHangUp,
  onToggleMute,
  route = "speaker",
  onToggleRoute,
  now = Date.now,
}: CallPanelProps) {
  const [code, setCode] = useState("");
  const [tick, setTick] = useState(0);
  const [invite, setInvite] = useState<{ outcome: InviteOutcome; role: CallRole; url: string } | null>(null);

  const active =
    call.status === "waiting" ||
    call.status === "connecting" ||
    call.status === "connected" ||
    call.status === "reconnecting";

  useEffect(() => {
    if (!call.connectedAt || call.status !== "connected") return;
    const timer = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(timer);
  }, [call.connectedAt, call.status]);

  const handleShare = useCallback(
    async (role: CallRole) => {
      if (!call.joinCode) return;
      const outcome = await shareInvite(call.joinCode, call.joinUrl, role);
      setInvite({ ...outcome, role });
    },
    [call.joinCode, call.joinUrl],
  );

  const elapsed = call.connectedAt ? formatElapsed(now() - call.connectedAt) : null;
  void tick;

  if (!active && !sessionActive && call.status !== "creating") {
    const trimmed = code.trim();
    return (
      <View style={styles.card} testID="call-panel-idle">
        <Text style={styles.explainer} testID="call-explainer">
          {CALL_MODE_EXPLAINER}
        </Text>
        <Text style={styles.explainer}>
          Each participant runs MindShift on their own phone and hears their own coach; a therapist
          can join in Safari to watch, read-only. Use earbuds or hold the phone to your ear so the
          coach&apos;s voice stays private.
        </Text>
        {call.status === "failed" && call.error ? (
          <Text style={styles.error} testID="call-error">
            {call.error}
          </Text>
        ) : null}
        {call.status === "ended" ? (
          <Text style={styles.muted} testID="call-ended-note">
            Call ended{call.connectedAt ? ` after ${formatElapsed(now() - call.connectedAt)}` : ""}.
          </Text>
        ) : null}
        {invitedCode ? (
          <>
            <Text style={styles.invited} testID="call-invited">
              You&apos;ve been invited to {invitedRole === "therapist" ? "watch a call, as the therapist" : "a call"}{" "}
              (code {invitedCode}).
            </Text>
            <TouchableOpacity
              testID="call-answer"
              accessibilityRole="button"
              style={[styles.button, styles.buttonPrimary]}
              onPress={() => (onAnswer ?? onJoin)(invitedCode)}
            >
              <Text style={styles.buttonPrimaryText}>Answer</Text>
            </TouchableOpacity>
          </>
        ) : (
          <>
            <TouchableOpacity
              testID="call-start"
              accessibilityRole="button"
              style={[styles.button, styles.buttonPrimary]}
              onPress={onStart}
            >
              <Text style={styles.buttonPrimaryText}>Start a call</Text>
            </TouchableOpacity>
            <View style={styles.joinRow}>
              <TextInput
                testID="call-code-input"
                style={styles.input}
                placeholder="Join with code"
                placeholderTextColor="#9CA3AF"
                autoCapitalize="none"
                autoCorrect={false}
                value={code}
                onChangeText={setCode}
              />
              <TouchableOpacity
                testID="call-join"
                accessibilityRole="button"
                accessibilityState={{ disabled: !isJoinCode(trimmed) }}
                disabled={!isJoinCode(trimmed)}
                style={[styles.button, styles.buttonSecondary, !isJoinCode(trimmed) && styles.buttonDisabled]}
                onPress={() => onJoin(trimmed)}
              >
                <Text style={styles.buttonSecondaryText}>Join</Text>
              </TouchableOpacity>
            </View>
          </>
        )}
      </View>
    );
  }

  const selfLabel = call.selfRole === "therapist" ? "You (therapist)" : "You";
  const statusLabel = callStatusLabel(call.status);
  return (
    <View style={styles.card} testID="call-panel-active">
      <Text style={styles.header} testID="call-header" numberOfLines={1}>
        {[selfLabel, statusLabel, elapsed].filter(Boolean).join(" · ")}
      </Text>
      {/* One row per other participant: name · connected/connecting · role. */}
      {call.peers.length > 0 ? (
        <View testID="call-peers">
          {call.peers.map((p) => (
            <Text key={p.uid} style={styles.peerRow} testID={`call-peer-${p.uid}`} numberOfLines={1}>
              {p.displayName} · {peerStatus(p)}
            </Text>
          ))}
        </View>
      ) : null}
      {call.status === "waiting" && call.joinCode ? (
        <View style={styles.inviteBox} testID="call-invite">
          <Text style={styles.inviteCode} testID="call-invite-code">
            Code: {call.joinCode}
          </Text>
          <Text style={styles.inviteUrl} numberOfLines={2}>
            {call.joinUrl || callWebUrl(call.joinCode)}
          </Text>
          {/* Host picks who each invite is for — the link encodes the role. */}
          {call.selfRole === "participant" ? (
            <View style={styles.inviteButtons}>
              <TouchableOpacity
                testID="call-invite-participant"
                accessibilityRole="button"
                style={[styles.button, styles.buttonSecondary]}
                onPress={() => handleShare("participant")}
              >
                <Text style={styles.buttonSecondaryText}>Invite a participant</Text>
              </TouchableOpacity>
              <TouchableOpacity
                testID="call-invite-therapist"
                accessibilityRole="button"
                style={[styles.button, styles.buttonSecondary]}
                onPress={() => handleShare("therapist")}
              >
                <Text style={styles.buttonSecondaryText}>Invite my therapist</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <TouchableOpacity
              testID="call-share"
              accessibilityRole="button"
              style={[styles.button, styles.buttonSecondary]}
              onPress={() => handleShare("participant")}
            >
              <Text style={styles.buttonSecondaryText}>Share invite</Text>
            </TouchableOpacity>
          )}
          {invite ? (
            <Text style={styles.muted} testID="call-invite-outcome">
              {invite.outcome === "shared"
                ? `Invite for the ${invite.role} shared.`
                : invite.outcome === "copied"
                  ? "Link copied — paste it to them."
                  : `Send them this link: ${invite.url}`}
            </Text>
          ) : null}
          {Platform.OS === "web" ? (
            <Text style={styles.muted}>In the app: {callDeepLink(call.joinCode)}</Text>
          ) : null}
        </View>
      ) : null}
      {call.status === "reconnecting" ? (
        <Text style={styles.muted} testID="call-reconnecting">
          A connection dropped — trying to recover it.
        </Text>
      ) : null}
      {!call.hasTurn && call.status !== "connected" ? (
        <Text style={styles.muted} testID="call-turn-note">
          No relay (TURN) server is configured — two phones on mobile data may not connect; Wi-Fi
          usually works.
        </Text>
      ) : null}
      <View style={styles.controls}>
        <TouchableOpacity
          testID="call-mute"
          accessibilityRole="button"
          style={[styles.button, styles.buttonSecondary, call.muted && styles.buttonMuted]}
          onPress={onToggleMute}
        >
          <Text style={styles.buttonSecondaryText}>{call.muted ? "Unmute" : "Mute"}</Text>
        </TouchableOpacity>
        {onToggleRoute && Platform.OS !== "web" ? (
          <TouchableOpacity
            testID="call-route"
            accessibilityRole="button"
            style={[styles.button, styles.buttonSecondary]}
            onPress={onToggleRoute}
          >
            <Text style={styles.buttonSecondaryText}>{route === "speaker" ? "Speaker" : "Earpiece"}</Text>
          </TouchableOpacity>
        ) : null}
        <TouchableOpacity
          testID="call-hangup"
          accessibilityRole="button"
          style={[styles.button, styles.buttonDanger]}
          onPress={onHangUp}
        >
          <Text style={styles.buttonPrimaryText}>Hang up</Text>
        </TouchableOpacity>
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
    padding: 14,
    gap: 8,
  },
  explainer: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#374151",
  },
  invited: {
    fontSize: 14,
    fontWeight: "600",
    color: "#1F2937",
  },
  error: {
    fontSize: 13,
    color: "#991B1B",
  },
  muted: {
    fontSize: 12.5,
    color: "#6B7280",
  },
  header: {
    fontSize: 16,
    fontWeight: "700",
    color: "#111827",
  },
  peerRow: {
    fontSize: 13.5,
    fontWeight: "600",
    color: "#374151",
    paddingVertical: 1,
  },
  inviteBox: {
    gap: 6,
    paddingVertical: 4,
  },
  inviteCode: {
    fontSize: 20,
    fontWeight: "700",
    letterSpacing: 1.5,
    color: "#1F2937",
  },
  inviteUrl: {
    fontSize: 12,
    color: "#4A90D9",
  },
  inviteButtons: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  joinRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    fontSize: 15,
    color: "#111827",
    backgroundColor: "#F9FAFB",
  },
  controls: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  button: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonPrimary: {
    backgroundColor: "#4A90D9",
  },
  buttonPrimaryText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "700",
  },
  buttonSecondary: {
    backgroundColor: "#EFF6FF",
    borderWidth: 1,
    borderColor: "#4A90D9",
  },
  buttonSecondaryText: {
    color: "#1D4ED8",
    fontSize: 14,
    fontWeight: "600",
  },
  buttonMuted: {
    backgroundColor: "#FEF3C7",
    borderColor: "#F59E0B",
  },
  buttonDanger: {
    backgroundColor: "#EF4444",
  },
  buttonDisabled: {
    opacity: 0.5,
  },
});
