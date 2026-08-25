import React, { useState, useCallback, useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Switch,
  StyleSheet,
  Platform,
} from "react-native";
import EmpathySlider from "../components/EmpathySlider";
import InterjectSlider from "../components/InterjectSlider";
import SuggestionCard from "../components/SuggestionCard";
import LiveTranscript from "../components/LiveTranscript";
import TherapistTranscript from "../components/TherapistTranscript";
import LiveModePicker, { LIVE_MODE_OPTIONS } from "../components/LiveModePicker";
import LivePreflightPanel from "../components/LivePreflightPanel";
import SessionSummaryCard from "../components/SessionSummaryCard";
import ScoreboardPanel from "../components/ScoreboardPanel";
import WhoIsThisSheet, { type LiveLabelChoice } from "../components/WhoIsThisSheet";
import CallPanel from "../components/CallPanel";
import { IDLE_CALL_VIEW } from "../live/call/types";
import { callApi } from "../live/call/callApi";
import { probeIce, iceProbeUnavailable, type IceProbeResult } from "../live/call/iceProbe";
import { useAudioStream, type TranscriptEntry } from "../hooks/useAudioStream";
import { useAuthStore } from "../store/authStore";
import { loadLiveMode, saveLiveMode } from "../live/modePrefs";
import { loadScoreboardVisible, saveScoreboardVisible } from "../live/scoreboardPrefs";
import type { LiveMode } from "../live/localLlm";
import type { CallRole } from "../live/call/types";
import { listVoicePeople, type VoicePerson } from "../api/liveSessions";
import { getTherapistLink, type TherapistLink } from "../api/therapist";
import * as apiClient from "../api/client";
import type { VoicePerson as ApiVoicePerson } from "../api/client";

const STATUS_COLORS: Record<string, string> = {
  idle: "#9CA3AF",
  connecting: "#F59E0B",
  live: "#10B981",
  disconnected: "#EF4444",
};

interface LiveCoachScreenProps {
  /** Return to Home (wired by App). Optional so the screen still renders
   *  standalone in tests; no back affordance is shown without it. */
  onBack?: () => void;
  /** Hand the finished live transcript off to the async-review Session screen.
   *  Turns carry utterance timing (seconds) when the live pipeline provided it,
   *  so post-session /analyze can compute real interruption stats. Optional so
   *  the screen still renders standalone (e.g. in isolation tests); the review
   *  button only calls it when present. */
  onReviewTranscript?: (
    turns: {
      speaker: string;
      text: string;
      start_time?: number;
      end_time?: number;
    }[],
  ) => void;
  /** A call code that arrived through an invite link (mindshift://call/<code>
   *  or https://…/call/<code>): opens in Call mode with an Answer button. */
  joinCode?: string | null;
  /** The role that invite link encodes (a therapist link -> observer view). */
  joinRole?: CallRole;
  /** The Answer tap consumed the code (so a re-render doesn't re-offer it). */
  onJoinCodeConsumed?: () => void;
}

export default function LiveCoachScreen({
  onBack,
  onReviewTranscript,
  joinCode = null,
  joinRole = "participant",
  onJoinCodeConsumed,
}: LiveCoachScreenProps = {}) {
  const {
    isRecording,
    sessionActive,
    transcript,
    suggestions,
    selfSpeaker,
    setSelfSpeaker,
    connectionStatus,
    transcriptionMessage,
    micError,
    speechAvailable,
    setSpeechEnabled,
    startSession,
    stopSession,
    sendEmpathyUpdate,
    sendInterjectUpdate,
    liveCapable,
    liveCapabilityReason,
    liveMode,
    setLiveMode,
    sessionMode,
    setSessionMode,
    liveStatus,
    nudgeFlash,
    clearNudgeFlash,
    latencySummary,
    toneFlags,
    preflight,
    runPreflight,
    escalationCount,
    sessionSummary,
    lastEpisode,
    speakerNames,
    displayNameOf,
    labelSpeaker,
    scoreboard,
    call,
    startCall,
    joinCall,
    hangUp,
    setCallMuted,
    callRoute,
    setCallRoute,
  } = useAudioStream();
  const callView = call ?? IDLE_CALL_VIEW;

  const userId = useAuthStore((s) => s.user?.uid ?? null);
  const [empathyLevel, setEmpathyLevel] = useState(50);
  const [interjectLevel, setInterjectLevel] = useState(0);
  // Scoreboard (opt-in, remembered per account) + mid-call naming state.
  const [scoreboardOn, setScoreboardOn] = useState(false);
  const scoreboardLoadedRef = useRef(false);
  const [who, setWho] = useState<{ speaker: string; label: string } | null>(null);
  const [sheetPeople, setSheetPeople] = useState<ApiVoicePerson[]>([]);
  const names = speakerNames ?? {};
  const nameOf = displayNameOf ?? ((s: string) => s);
  // "Who's here" (enrolled people, read-only) and the therapist link (for the
  // end-of-session share affordance). null = still loading / unavailable.
  const [people, setPeople] = useState<VoicePerson[] | null>(null);
  const [peopleError, setPeopleError] = useState<string | null>(null);
  const [therapist, setTherapist] = useState<TherapistLink | null>(null);
  const modeLoadedRef = useRef(false);
  // Call mode connectivity pre-flight (src/live/call/iceProbe.ts).
  const [iceProbe, setIceProbe] = useState<IceProbeResult | null>(null);
  const [iceProbing, setIceProbing] = useState(false);

  // A haptic nudge on the user's own delivery also flashes on screen for a
  // moment (the phone may be face-down on the table; the buzz is primary).
  useEffect(() => {
    if (!nudgeFlash) return;
    const timer = setTimeout(() => clearNudgeFlash?.(), 1500);
    return () => clearTimeout(timer);
  }, [nudgeFlash, clearNudgeFlash]);

  // The mode decides whether the coach speaks: earpiece, in person and call
  // do (free on-device TTS; the fast loop additionally holds speech until
  // the room is quiet), therapist mode never does. The hook stops any
  // in-flight utterance when this flips to false.
  useEffect(() => {
    setSpeechEnabled(sessionMode !== "therapist");
  }, [sessionMode, setSpeechEnabled]);

  // Remember the mode per account (Sage's phone opens on the mode he used
  // last, Mom's on therapist) — loaded once, saved on every explicit change.
  // An invite link overrides it for this visit (Call mode, not persisted).
  useEffect(() => {
    let cancelled = false;
    void loadLiveMode(userId).then((mode) => {
      if (cancelled || modeLoadedRef.current) return;
      modeLoadedRef.current = true;
      setSessionMode?.(joinCode ? "call" : mode);
    });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);
  useEffect(() => {
    if (joinCode) setSessionMode?.("call");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [joinCode]);

  const handleModeChange = useCallback(
    (mode: LiveMode) => {
      setSessionMode?.(mode);
      void saveLiveMode(userId, mode);
    },
    [setSessionMode, userId],
  );

  // The scoreboard is a game two people opt into — off until this account
  // turned it on, then remembered like the mode.
  useEffect(() => {
    let cancelled = false;
    void loadScoreboardVisible(userId).then((on) => {
      if (cancelled || scoreboardLoadedRef.current) return;
      scoreboardLoadedRef.current = true;
      setScoreboardOn(on);
    });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  const handleScoreboardToggle = useCallback(
    (on: boolean) => {
      setScoreboardOn(on);
      void saveScoreboardVisible(userId, on);
    },
    [userId],
  );

  // Mid-call naming: tap a speaker (chip, transcript label or therapist
  // column) → "Who is this?" over the enrolled people. The people list is
  // fetched when the sheet opens so a person enrolled a moment ago shows.
  const openWho = useCallback(
    (entry: Pick<TranscriptEntry, "speaker" | "speakerId">) => {
      const raw = entry.speakerId ?? entry.speaker;
      setWho({ speaker: raw, label: entry.speaker });
      void Promise.resolve()
        .then(() => apiClient.listVoicePeople())
        .then((res) => setSheetPeople(Array.isArray(res?.people) ? res.people : []))
        .catch(() => {
          // No people list (older server / offline): the sheet still offers
          // "New person…" — naming this call never depends on it.
        });
    },
    [],
  );

  const handleLiveLabel = useCallback(
    async (choice: LiveLabelChoice) => {
      if (!who || !labelSpeaker) return { text: "Naming isn’t available right now." };
      const outcome = await labelSpeaker(who.speaker, choice);
      return { text: outcome.text };
    },
    [who, labelSpeaker],
  );

  // Every distinct voice heard so far (raw label → shown name), for the chips.
  const speakerChips = React.useMemo(() => {
    const seen: { speaker: string; label: string; named: boolean }[] = [];
    for (const t of transcript) {
      const raw = t.speakerId ?? t.speaker;
      if (seen.some((s) => s.speaker === raw)) continue;
      seen.push({ speaker: raw, label: t.speaker, named: Boolean(names[raw]) });
    }
    return seen;
  }, [transcript, names]);
  const isNamed = useCallback(
    (entry: TranscriptEntry) => Boolean(names[entry.speakerId ?? entry.speaker]),
    [names],
  );

  // Pre-flight: probe what the on-device loop would load, once per mount
  // (and again if the on-device switch is flipped back on).
  useEffect(() => {
    if (liveCapable && liveMode) void runPreflight?.();
  }, [liveCapable, liveMode, runPreflight]);

  useEffect(() => {
    let cancelled = false;
    void listVoicePeople().then((res) => {
      if (cancelled) return;
      setPeople(res.people);
      setPeopleError(res.error);
    });
    getTherapistLink()
      .then((l) => {
        if (!cancelled) setTherapist(l);
      })
      .catch(() => {
        if (!cancelled) setTherapist(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleToggle = useCallback(async () => {
    // Toggle on sessionActive, not isRecording: a session can be live with
    // mic capture unavailable (e.g. web) and must still be stoppable.
    if (sessionActive) {
      await stopSession();
    } else {
      const sessionId = `live-${Date.now()}`;
      await startSession(sessionId, empathyLevel, interjectLevel);
    }
  }, [sessionActive, stopSession, startSession, empathyLevel, interjectLevel]);

  // Flip the coached user's identity between the two diarized speakers. The
  // server labels the first voice it hears "Speaker A", so that's the default.
  const handleToggleSelfSpeaker = useCallback(() => {
    setSelfSpeaker(selfSpeaker === "Speaker B" ? "Speaker A" : "Speaker B");
  }, [selfSpeaker, setSelfSpeaker]);

  // Call mode (src/live/call): every action here is one tap that does
  // everything — on Safari the tap is what unlocks the mic and audio.
  const handleStartCall = useCallback(() => {
    void startCall?.(empathyLevel, interjectLevel);
  }, [startCall, empathyLevel, interjectLevel]);
  const handleJoinCall = useCallback(
    (code: string) => {
      void joinCall?.(code, empathyLevel, interjectLevel);
    },
    [joinCall, empathyLevel, interjectLevel],
  );
  const handleAnswerCall = useCallback(
    (code: string) => {
      onJoinCodeConsumed?.();
      // The invite link's role decides the seat: a therapist link joins as
      // the read-only observer (no TTS, no coaching for them).
      void joinCall?.(code, empathyLevel, interjectLevel, joinRole);
    },
    [joinCall, onJoinCodeConsumed, empathyLevel, interjectLevel, joinRole],
  );
  const handleHangUp = useCallback(() => {
    void hangUp?.();
  }, [hangUp]);
  const handleToggleMute = useCallback(() => {
    setCallMuted?.(!callView.muted);
  }, [setCallMuted, callView.muted]);
  const handleToggleRoute = useCallback(() => {
    setCallRoute?.(callRoute === "speaker" ? "earpiece" : "speaker");
  }, [setCallRoute, callRoute]);

  const handleReview = useCallback(() => {
    onReviewTranscript?.(
      // Carry utterance timing through to review (camelCase hook fields →
      // snake_case Turn fields, matching the /analyze wire contract). Only
      // when actually present — legacy-server entries have none.
      transcript.map((t) => ({
        speaker: t.speaker,
        text: t.text,
        ...(t.startTime !== undefined ? { start_time: t.startTime } : {}),
        ...(t.endTime !== undefined ? { end_time: t.endTime } : {}),
      })),
    );
  }, [onReviewTranscript, transcript]);

  const handleEmpathyChange = useCallback(
    (value: number) => {
      setEmpathyLevel(value);
      sendEmpathyUpdate(value);
    },
    [sendEmpathyUpdate],
  );

  const handleInterjectChange = useCallback(
    (value: number) => {
      // Round at the source: the server takes an int, and this state also
      // feeds startSession's initial config — keep both paths in sync.
      const rounded = Math.round(value);
      setInterjectLevel(rounded);
      sendInterjectUpdate(rounded);
    },
    [sendInterjectUpdate],
  );

  const statusColor = STATUS_COLORS[connectionStatus] || STATUS_COLORS.idle;
  const mode: LiveMode = sessionMode ?? "earpiece";
  const isCall = mode === "call";
  // A therapist observer (Therapist mode, or a therapist-role call) gets the
  // two-column observer layout and never sees "speak"/nudge affordances
  // aimed at themselves.
  const isTherapistCall = isCall && callView.selfRole === "therapist";
  const isTherapist = mode === "therapist" || isTherapistCall;
  const modeLabel = LIVE_MODE_OPTIONS.find((o) => o.mode === mode)?.label ?? mode;
  const idle = connectionStatus === "idle" && transcript.length === 0 && !sessionActive;

  // Call mode: before anyone dials, find out whether these two phones can
  // actually reach each other. Fetch the server's own ICE servers
  // (GET /calls/ice — same list a call hands out, with this account's
  // short-lived TURN credentials) and gather candidates against them. The
  // answer is one line in the pre-flight panel; a failure to check says so
  // instead of quietly looking fine.
  useEffect(() => {
    if (!isCall || !idle) return;
    let cancelled = false;
    setIceProbing(true);
    void (async () => {
      let result: IceProbeResult;
      try {
        const config = await callApi.ice();
        result = await probeIce(config.iceServers);
      } catch (err) {
        result = iceProbeUnavailable(err instanceof Error ? err.message : String(err));
      }
      if (cancelled) return;
      setIceProbe(result);
      setIceProbing(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [isCall, idle]);

  return (
    <View style={styles.container}>
      {/* Header with connection status. The heading takes the flexible space
          and the status pins to the right at a fixed width, so a long status
          word ("disconnected") can never overlap the title (a real Pixel bug). */}
      <View style={styles.header}>
        {onBack && (
          <TouchableOpacity
            testID="live-coach-back"
            accessibilityRole="button"
            style={styles.backButton}
            onPress={onBack}
            hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
          >
            <Text style={styles.backButtonText}>←</Text>
          </TouchableOpacity>
        )}
        <Text style={styles.heading} numberOfLines={1}>
          Live Coach
        </Text>
        <View style={styles.statusRow}>
          <View
            style={[styles.statusDot, { backgroundColor: statusColor }]}
            testID="connection-status"
          />
          <Text
            style={[styles.statusText, { color: statusColor }]}
            numberOfLines={1}
          >
            {connectionStatus}
          </Text>
        </View>
      </View>

      {/* Identity chip: which diarized voice is the user's. Shown once there's
          a session or a first transcript line — before that the toggle would
          be meaningless. Tapping flips A↔B; the hint reminds the "you speak
          first" convention while idle. Therapist mode has no "you" on the
          mic, so the chip is hidden there. */}
      {!isTherapist && !isCall && (sessionActive || transcript.length > 0) && (
        <View style={styles.identityRow}>
          <TouchableOpacity
            testID="self-speaker-chip"
            style={styles.identityChip}
            onPress={handleToggleSelfSpeaker}
          >
            <Text style={styles.identityChipText}>
              You: {selfSpeaker ?? "Speaker A"} ⇄
            </Text>
          </TouchableOpacity>
          {connectionStatus === "idle" && (
            <Text style={styles.identityHint}>you speak first</Text>
          )}
        </View>
      )}

      {/* Microphone error banner — honest failure state, never a fake session */}
      {micError ? (
        <View style={styles.errorBanner} testID="mic-error-banner">
          <Text style={styles.errorBannerText}>{micError}</Text>
        </View>
      ) : null}

      {/* Transcription availability banner */}
      {transcriptionMessage ? (
        <View style={styles.banner} testID="transcription-banner">
          <Text style={styles.bannerText}>
            Transcription unavailable: {transcriptionMessage}
          </Text>
        </View>
      ) : null}

      {/* The one choice that shapes the session: who's on the mic and whether
          the coach speaks. Locked while a session runs (the loop reads it at
          start). Persisted per account. */}
      <LiveModePicker value={mode} onChange={handleModeChange} disabled={sessionActive} />

      {/* Call mode: start / join / answer, then the in-call header and
          controls. The transcript and suggestions below are shared with
          every other mode — the other person's turns arrive as transcript
          events with their name. */}
      {isCall ? (
        <CallPanel
          call={callView}
          sessionActive={sessionActive}
          invitedCode={joinCode}
          invitedRole={joinRole}
          onStart={handleStartCall}
          onJoin={handleJoinCall}
          onAnswer={handleAnswerCall}
          onHangUp={handleHangUp}
          onToggleMute={handleToggleMute}
          route={callRoute ?? "speaker"}
          onToggleRoute={handleToggleRoute}
        />
      ) : null}

      {/* On-device fast loop (Track 3): only offered when the device can run
          it (on-device STT present). Off = the legacy server path. */}
      {liveCapable ? (
        <View style={styles.modeRow} testID="live-mode-row">
          <Text style={styles.modeLabel}>On-device coaching</Text>
          <Switch
            testID="live-mode-switch"
            value={Boolean(liveMode)}
            onValueChange={setLiveMode}
            disabled={sessionActive}
          />
          <Text style={styles.modeHint} numberOfLines={1}>
            {liveMode ? "phone does the work" : "server does the work"}
          </Text>
        </View>
      ) : null}

      {/* What the loop actually loaded (or why it isn't running). */}
      {liveStatus ? (
        <Text style={styles.speechUnavailableText} testID="live-status">
          {liveStatus}
        </Text>
      ) : null}

      {/* Pleasantness scoreboard (PRD §6): opt-in, off by default, remembered
          per account. A race to be nicer — both lines climbing is the win. */}
      <View style={styles.modeRow} testID="scoreboard-row">
        <Text style={styles.modeLabel}>Scoreboard</Text>
        <Switch
          testID="scoreboard-switch"
          value={scoreboardOn}
          onValueChange={handleScoreboardToggle}
        />
        <Text style={styles.modeHint} numberOfLines={1}>
          {scoreboardOn ? "who's being nicer — a race to be kind" : "off"}
        </Text>
      </View>
      {scoreboardOn || isTherapistCall ? (
        <ScoreboardPanel
          board={scoreboard ?? null}
          nameOf={nameOf}
          emptyText={
            !liveCapable
              ? "Scores need on-device coaching, which this device can't run yet."
              : !liveMode
                ? "Scores need on-device coaching — switch it on above, then start."
                : undefined
          }
        />
      ) : null}

      {/* Who's talking: one chip per voice heard. Tap to name them ("Who is
          this?") — the name applies for the rest of the call at once. */}
      {speakerChips.length > 0 ? (
        <View style={styles.chipRow} testID="speaker-chips">
          {speakerChips.map((c) => (
            <TouchableOpacity
              key={c.speaker}
              testID={`speaker-chip-${c.speaker}`}
              accessibilityRole="button"
              accessibilityLabel={`Who is ${c.label}?`}
              style={[styles.speakerChip, c.named && styles.speakerChipNamed]}
              onPress={() => openWho({ speaker: c.label, speakerId: c.speaker })}
            >
              <Text style={[styles.speakerChipText, c.named && styles.speakerChipTextNamed]}>
                {c.label}
                {c.named ? "" : " · who?"}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      ) : null}

      {/* Session strip: mode + escalation count while live. */}
      {sessionActive ? (
        <View style={styles.sessionStrip} testID="session-strip">
          <Text style={styles.sessionStripText}>{modeLabel}</Text>
          <Text
            style={[styles.sessionStripText, escalationCount > 0 && styles.sessionStripWarn]}
            testID="escalation-count"
          >
            {isTherapist ? "observing" : `escalations: ${escalationCount ?? 0}`}
          </Text>
        </View>
      ) : null}

      {/* Haptic nudge mirror: "you're getting loud/heated" on the user's own
          turn. Level 1–3 from the shared nudge policy. */}
      {nudgeFlash ? (
        <View style={styles.nudgeFlash} testID="nudge-flash">
          <Text style={styles.nudgeFlashText}>
            {`Easy — level ${nudgeFlash.level}${
              nudgeFlash.vectors.length > 0
                ? ` (${nudgeFlash.vectors.join(", ").replace(/_/g, " ")})`
                : ""
            }`}
          </Text>
        </View>
      ) : null}

      {/* Server tone flag (additive to on-device coaching). */}
      {toneFlags && toneFlags.length > 0 ? (
        <Text style={styles.toneFlagText} testID="tone-flag">
          {`⚠ ${toneFlags[0].speaker}: ${toneFlags[0].label} (${toneFlags[0].source} tone)`}
        </Text>
      ) : null}

      {/* Honest state: a spoken mode selected but this platform has no TTS —
          suggestions stay visual-only instead of silently pretending. */}
      {!isTherapist && !speechAvailable ? (
        <Text style={styles.speechUnavailableText} testID="speech-unavailable-note">
          Spoken suggestions aren&apos;t available on this platform — showing
          them on screen only.
        </Text>
      ) : null}

      {/* Empathy slider */}
      <EmpathySlider
        value={empathyLevel}
        onValueChange={handleEmpathyChange}
      />

      {/* How often the coach should interject */}
      <InterjectSlider
        value={interjectLevel}
        onValueChange={handleInterjectChange}
      />

      {/* Idle: the honest pre-flight (what will run on this phone, who the
          loop expects to hear) and the short how-to. Disappears the moment a
          session starts or any transcript arrives. */}
      {idle ? (
        <>
          <LivePreflightPanel
            liveCapable={liveCapable}
            liveCapabilityReason={liveCapabilityReason ?? ""}
            liveMode={Boolean(liveMode)}
            preflight={preflight ?? null}
            people={people}
            peopleError={peopleError}
            isCall={isCall}
            iceProbe={iceProbe}
            iceProbing={iceProbing}
          />
          <View style={styles.explainerCard} testID="idle-explainer">
            {isTherapist ? (
              <Text style={styles.explainerLine}>
                Place the phone between the two of them and tap Start — you&apos;ll
                see both sides labelled; nothing is spoken aloud.
              </Text>
            ) : isCall ? (
              <Text style={styles.explainerLine} testID="call-mode-explainer">
                Start a call and share the code, or join theirs — only your voice is on
                this mic, so the coach always knows which one is you.
              </Text>
            ) : (
              <>
                <Text style={styles.explainerLine}>
                  {mode === "speaker"
                    ? "Both of you in the room: place the phone between you."
                    : "Hold the phone to your ear as you normally would."}
                </Text>
                <Text style={styles.explainerLine}>
                  Tap Start, then speak first — the coach learns which voice is yours.
                </Text>
              </>
            )}
            {Platform.OS === "web" ? (
              // The browser build (Safari on an iPhone): the mic only runs while
              // this page is on screen — a locked screen or another app ends it.
              <Text style={styles.explainerLine} testID="web-foreground-note">
                In the browser, keep this page open and the screen on — locking
                the phone or switching apps stops the microphone.
              </Text>
            ) : null}
          </View>
        </>
      ) : null}

      {/* Live transcript: two labelled columns in therapist mode. */}
      {isTherapist ? (
        <TherapistTranscript entries={transcript} onSpeakerPress={openWho} isNamed={isNamed} />
      ) : (
        <LiveTranscript entries={transcript} onSpeakerPress={openWho} isNamed={isNamed} />
      )}

      {who ? (
        <WhoIsThisSheet
          visible
          speaker={who.speaker}
          currentLabel={nameOf(who.speaker)}
          currentPersonId={names[who.speaker]?.personId ?? null}
          people={sheetPeople}
          hasAudio={false}
          onClose={() => setWho(null)}
          onLiveLabel={handleLiveLabel}
        />
      ) : null}

      {/* Suggestion feed: newest first, older entries faded so the eye lands
          on the latest. Nudges (about the user's OWN turn) render as a compact
          banner; responses render the usual SuggestionCard stack. Each entry
          says where it came from (the phone's fast loop or the cloud) — the
          local one lands first, the cloud one augments it. */}
      {suggestions.length > 0 && (
        <ScrollView
          style={styles.suggestionsContainer}
          horizontal={false}
          testID="suggestions-list"
        >
          <Text style={styles.suggestionsTitle}>Suggestions</Text>
          {suggestions.map((entry, i) => {
            // Newest at full strength; older entries fade uniformly. Muted
            // entries keep their own extra dimming (in SuggestionCard / the
            // banner style) on top of this.
            const ageStyle =
              i === 0 ? styles.feedEntryNewest : styles.feedEntryOlder;
            const sourceTag = entry.source ? (
              <Text
                style={[
                  styles.sourceTag,
                  entry.source === "on-device" ? styles.sourceTagLocal : styles.sourceTagCloud,
                ]}
                testID={`suggestion-source-${entry.id}`}
              >
                {entry.forName ? `for ${entry.forName} · ` : ""}
                {entry.source === "on-device" ? "on-device" : "cloud"}
                {entry.partial ? " · writing…" : ""}
              </Text>
            ) : entry.forName ? (
              <Text style={[styles.sourceTag, styles.sourceTagCloud]} testID={`suggestion-for-${entry.id}`}>
                for {entry.forName}
              </Text>
            ) : null;
            if (entry.kind === "nudge") {
              return (
                <View
                  key={entry.id}
                  testID="nudge-banner"
                  style={[
                    styles.nudgeBanner,
                    ageStyle,
                    entry.muted && styles.nudgeBannerMuted,
                  ]}
                >
                  <View style={styles.nudgeBadge}>
                    <Text style={styles.nudgeBadgeText}>NUDGE</Text>
                  </View>
                  <Text style={styles.nudgeText} numberOfLines={1}>
                    {entry.texts[0]}
                  </Text>
                  {sourceTag}
                </View>
              );
            }
            return (
              <View key={entry.id} style={ageStyle}>
                {sourceTag ? <View style={styles.sourceRow}>{sourceTag}</View> : null}
                {entry.texts.map((text, j) => (
                  <SuggestionCard
                    key={j}
                    text={text}
                    tone={entry.tone}
                    muted={entry.muted}
                  />
                ))}
              </View>
            );
          })}
        </ScrollView>
      )}

      {/* Session end: the summary card (duration, turns, escalations,
          first-words latency) with "Share with my therapist" when linked. */}
      {!sessionActive && sessionSummary ? (
        <SessionSummaryCard
          summary={sessionSummary}
          episode={lastEpisode ?? null}
          therapist={therapist}
        />
      ) : null}

      {/* Latency report from the on-device loop (printed in full to the
          console at session end; the headline lands here). */}
      {!sessionActive && latencySummary ? (
        <Text style={styles.speechUnavailableText} testID="latency-summary">
          {latencySummary}
        </Text>
      ) : null}

      {/* Post-session review handoff: after a session ends with something to
          review, offer a prominent jump to the async-review Session screen. */}
      {!sessionActive && transcript.length > 0 && (
        <TouchableOpacity
          testID="review-transcript-button"
          style={styles.reviewButton}
          onPress={handleReview}
        >
          <Text style={styles.reviewButtonText}>
            Review this conversation →
          </Text>
        </TouchableOpacity>
      )}

      {/* Start/Stop button. Call mode starts from the call panel (Start a
          call / Join / Answer) and stops by hanging up. */}
      {isCall && !sessionActive ? null : (
        <TouchableOpacity
          testID="mic-toggle"
          style={[
            styles.micButton,
            isRecording && styles.micButtonRecording,
          ]}
          onPress={isCall ? handleHangUp : handleToggle}
        >
          <Text style={styles.micButtonText}>
            {isCall ? "Hang up" : sessionActive ? "Stop Listening" : "Start Listening"}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F9FAFB",
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingTop: 16,
    paddingBottom: 8,
  },
  backButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 8,
  },
  backButtonText: {
    fontSize: 22,
    fontWeight: "600",
    color: "#4A90D9",
  },
  heading: {
    fontSize: 24,
    fontWeight: "700",
    color: "#111827",
    // Take the flexible space and shrink/ellipsize rather than shove the
    // status text off the right edge.
    flex: 1,
    flexShrink: 1,
    marginRight: 8,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    // Never shrink: the status keeps its full width, the heading yields.
    flexShrink: 0,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  statusText: {
    fontSize: 13,
    fontWeight: "600",
    textTransform: "capitalize",
    // A fixed-ish width right-aligned so the dot doesn't jump as the word
    // length changes ("live" vs "disconnected").
    minWidth: 92,
    textAlign: "right",
  },
  identityRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingBottom: 4,
    gap: 8,
  },
  identityChip: {
    paddingVertical: 6,
    paddingHorizontal: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#4A90D9",
    backgroundColor: "#EFF6FF",
  },
  identityChipText: {
    fontSize: 13,
    fontWeight: "600",
    color: "#4A90D9",
  },
  identityHint: {
    fontSize: 12,
    color: "#6B7280",
    fontStyle: "italic",
  },
  chipRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    paddingHorizontal: 16,
    paddingBottom: 6,
  },
  speakerChip: {
    paddingVertical: 4,
    paddingHorizontal: 10,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#FFFFFF",
  },
  speakerChipNamed: {
    borderColor: "#10B981",
    backgroundColor: "#ECFDF5",
  },
  speakerChipText: {
    fontSize: 12,
    fontWeight: "600",
    color: "#374151",
  },
  speakerChipTextNamed: {
    color: "#047857",
  },
  explainerCard: {
    backgroundColor: "#FFFFFF",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#E5E7EB",
    marginHorizontal: 16,
    marginVertical: 6,
    padding: 14,
    gap: 6,
  },
  explainerLine: {
    fontSize: 13.5,
    lineHeight: 19,
    color: "#374151",
  },
  banner: {
    backgroundColor: "#FEF3C7",
    borderLeftWidth: 4,
    borderLeftColor: "#F59E0B",
    marginHorizontal: 16,
    marginBottom: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  bannerText: {
    fontSize: 13,
    color: "#92400E",
  },
  errorBanner: {
    backgroundColor: "#FEE2E2",
    borderLeftWidth: 4,
    borderLeftColor: "#EF4444",
    marginHorizontal: 16,
    marginBottom: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  errorBannerText: {
    fontSize: 13,
    color: "#991B1B",
  },
  modeRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 6,
    gap: 8,
  },
  modeLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#374151",
    marginRight: 4,
  },
  modeHint: {
    flex: 1,
    fontSize: 12,
    color: "#6B7280",
  },
  sessionStrip: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingBottom: 4,
  },
  sessionStripText: {
    fontSize: 12.5,
    fontWeight: "600",
    color: "#6B7280",
  },
  sessionStripWarn: {
    color: "#B45309",
  },
  speechUnavailableText: {
    fontSize: 12,
    color: "#6B7280",
    paddingHorizontal: 16,
    paddingBottom: 4,
  },
  nudgeFlash: {
    backgroundColor: "#FEE2E2",
    borderLeftWidth: 4,
    borderLeftColor: "#EF4444",
    marginHorizontal: 16,
    marginBottom: 4,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  nudgeFlashText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#991B1B",
  },
  toneFlagText: {
    fontSize: 12,
    color: "#92400E",
    paddingHorizontal: 16,
    paddingBottom: 4,
  },
  suggestionsContainer: {
    maxHeight: 200,
    paddingBottom: 8,
  },
  suggestionsTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1F2937",
    paddingHorizontal: 16,
    paddingTop: 8,
    marginBottom: 4,
  },
  feedEntryNewest: {
    opacity: 1,
  },
  feedEntryOlder: {
    // Faded so the eye lands on the newest advice, but still legible.
    opacity: 0.75,
  },
  sourceRow: {
    flexDirection: "row",
    paddingHorizontal: 16,
    marginTop: 4,
  },
  sourceTag: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    paddingVertical: 2,
    paddingHorizontal: 7,
    borderRadius: 6,
    overflow: "hidden",
  },
  sourceTagLocal: {
    color: "#15803D",
    backgroundColor: "#DCFCE7",
  },
  sourceTagCloud: {
    color: "#1D4ED8",
    backgroundColor: "#DBEAFE",
  },
  nudgeBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    backgroundColor: "#FFFBEB",
    borderLeftWidth: 4,
    borderLeftColor: "#F59E0B",
    marginHorizontal: 16,
    marginVertical: 6,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  nudgeBannerMuted: {
    opacity: 0.5,
  },
  nudgeBadge: {
    backgroundColor: "#F59E0B",
    paddingVertical: 2,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  nudgeBadgeText: {
    fontSize: 10,
    fontWeight: "700",
    color: "#FFFFFF",
    letterSpacing: 0.5,
  },
  nudgeText: {
    flex: 1,
    fontSize: 14,
    fontWeight: "700",
    color: "#92400E",
  },
  reviewButton: {
    backgroundColor: "#EFF6FF",
    borderWidth: 1,
    borderColor: "#4A90D9",
    marginHorizontal: 16,
    marginTop: 8,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
  },
  reviewButtonText: {
    color: "#4A90D9",
    fontSize: 16,
    fontWeight: "700",
  },
  micButton: {
    backgroundColor: "#4A90D9",
    marginHorizontal: 16,
    marginVertical: 12,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  micButtonRecording: {
    backgroundColor: "#EF4444",
  },
  micButtonText: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "700",
  },
});
