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
import MoodCheck from "../components/MoodCheck";
import SessionSummaryCard from "../components/SessionSummaryCard";
import ScoreboardPanel from "../components/ScoreboardPanel";
import WhoIsThisSheet, { type LiveLabelChoice } from "../components/WhoIsThisSheet";
import CallPanel from "../components/CallPanel";
import JournalPanel, { type JournalGate } from "../components/JournalPanel";
import { IDLE_JOURNAL_STATE } from "../live/journalRecorder";
import { IDLE_CALL_VIEW } from "../live/call/types";
import { callApi } from "../live/call/callApi";
import { probeIce, iceProbeUnavailable, type IceProbeResult } from "../live/call/iceProbe";
import { useAudioStream, type TranscriptEntry } from "../hooks/useAudioStream";
import { useAuthStore } from "../store/authStore";
import { useDevModeStore } from "../store/devModeStore";
import { useMoodStore } from "../store/moodStore";
import { loadLiveMode, saveLiveMode } from "../live/modePrefs";
import { loadScoreboardVisible, saveScoreboardVisible } from "../live/scoreboardPrefs";
import { DEFAULT_KEEP_AUDIO, loadKeepAudio, saveKeepAudio } from "../live/keepAudioPrefs";
import type { LiveMode } from "../live/localLlm";
import type { CallRole } from "../live/call/types";
import { listVoicePeople, patchSessionMood, type VoicePerson } from "../api/liveSessions";
import { getTherapistLink, type TherapistLink } from "../api/therapist";
import * as apiClient from "../api/client";
import type { VoicePerson as ApiVoicePerson } from "../api/client";

const STATUS_COLORS: Record<string, string> = {
  idle: "#9CA3AF",
  connecting: "#F59E0B",
  live: "#10B981",
  disconnected: "#EF4444",
};

/** Plain words for the header status when developer mode is off — the raw
 *  socket state ("disconnected") reads as jargon to an invited tester. */
const FRIENDLY_STATUS: Record<string, string> = {
  idle: "ready",
  connecting: "connecting…",
  live: "listening",
  disconnected: "offline",
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
  /** "Hey Google, start my journal" (mindshift://journal/start|stop, wired in
   *  App.tsx): "start" selects Journal mode and starts it — honoring the
   *  existing gates (a missing owner voiceprint lands here with the gate
   *  message visible instead; a mic failure shows the usual micError banner);
   *  "stop" stops a running journal session. */
  journalAction?: "start" | "stop" | null;
  /** The action was executed (or ruled out) — so a re-render doesn't redo it. */
  onJournalActionConsumed?: () => void;
}

export default function LiveCoachScreen({
  onBack,
  onReviewTranscript,
  joinCode = null,
  joinRole = "participant",
  onJoinCodeConsumed,
  journalAction = null,
  onJournalActionConsumed,
}: LiveCoachScreenProps = {}) {
  // Keep this session's audio (default ON — see keepAudioPrefs.ts): read
  // once per account before the hook needs it; the switch below changes it
  // for the NEXT session (a running session keeps whatever it started with).
  const keepAudioUserId = useAuthStore((s) => s.user?.uid ?? null);
  const [keepAudio, setKeepAudio] = useState(DEFAULT_KEEP_AUDIO);
  const keepAudioLoadedRef = useRef(false);
  useEffect(() => {
    let cancelled = false;
    void loadKeepAudio(keepAudioUserId).then((on) => {
      if (cancelled || keepAudioLoadedRef.current) return;
      keepAudioLoadedRef.current = true;
      setKeepAudio(on);
    });
    return () => {
      cancelled = true;
    };
  }, [keepAudioUserId]);
  const handleKeepAudioToggle = useCallback(
    (on: boolean) => {
      setKeepAudio(on);
      void saveKeepAudio(keepAudioUserId, on);
    },
    [keepAudioUserId],
  );

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
    watchConnected,
    preflight,
    runPreflight,
    escalationCount,
    sessionSummary,
    lastEpisode,
    journal,
    retryJournalUploads,
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
  } = useAudioStream({ keepAudio });
  const callView = call ?? IDLE_CALL_VIEW;

  const userId = useAuthStore((s) => s.user?.uid ?? null);
  const [empathyLevel, setEmpathyLevel] = useState(50);
  const [interjectLevel, setInterjectLevel] = useState(0);
  // Speak the coach's suggestions aloud (default on). Off = on-screen only,
  // for when you have no earbud — which also stops the phone's own TTS from
  // playing on the speaker, being re-heard by the mic, and showing up as a
  // phantom extra speaker (a real feedback loop on a Pixel 10). Therapist mode
  // is always silent regardless.
  const [speakAloud, setSpeakAloud] = useState(true);
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
  // Journal mode never speaks either (nothing is coached while it listens).
  useEffect(() => {
    setSpeechEnabled(sessionMode !== "therapist" && sessionMode !== "journal" && speakAloud);
  }, [sessionMode, speakAloud, setSpeechEnabled]);

  // Remember the mode per account (Sage's phone opens on the mode he used
  // last, Mom's on therapist) — loaded once, saved on every explicit change.
  // An invite link overrides it for this visit (Call mode, not persisted).
  useEffect(() => {
    let cancelled = false;
    void loadLiveMode(userId).then((mode) => {
      if (cancelled || modeLoadedRef.current) return;
      modeLoadedRef.current = true;
      // A call invite or a journal start link overrides the remembered mode
      // for this visit (neither is persisted — see handleModeChange).
      setSessionMode?.(joinCode ? "call" : journalAction === "start" ? "journal" : mode);
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
  // "Hey Google, start my journal": the start link selects Journal mode the
  // same way an invite link selects Call mode — for this visit only, never
  // persisted. (A stop link doesn't touch the mode: stopping must never
  // silently re-mode a running non-journal session.)
  useEffect(() => {
    if (journalAction === "start") setSessionMode?.("journal");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [journalAction]);

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

  // Voiceprint gate data. Refetches on a schedule while the gate is not
  // "ok" (a fetch that raced sign-in or hit a cold instance must heal
  // without the user leaving the screen — on-device 2026-08-30 the enroll
  // banner showed to an enrolled owner and never went away).
  const peopleRetryRef = useRef(0);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const fetchPeople = () => {
      void listVoicePeople().then((res) => {
        if (cancelled) return;
        setPeople(res.people);
        setPeopleError(res.error);
        const gateOk = res.people.some((p) => p.isSelf && Math.max(p.settings, p.enrollCount) >= 1);
        if (!gateOk && peopleRetryRef.current < 5) {
          peopleRetryRef.current += 1;
          timer = setTimeout(fetchPeople, 4000 * peopleRetryRef.current);
        }
      });
    };
    fetchPeople();
    getTherapistLink()
      .then((l) => {
        if (!cancelled) setTherapist(l);
      })
      .catch(() => {
        if (!cancelled) setTherapist(null);
      });
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  const handleToggle = useCallback(async () => {
    // Toggle on sessionActive, not isRecording: a session can be live with
    // mic capture unavailable (e.g. web) and must still be stoppable.
    if (sessionActive) {
      await stopSession();
    } else {
      // A fresh session must never carry a PREVIOUS session's mood-check
      // answers. `sessionSummary` non-null means a session already ended —
      // this Start is for session N+1, so whatever the store holds is
      // stale (the idle BEFORE check, gated on an empty transcript, only
      // ever shows once per app open — it does not reappear between back-
      // to-back sessions). When it's null, this is the very first Start and
      // the store holds exactly the BEFORE answer just given for THIS
      // session — resetting here would erase it before stop can read it.
      if (sessionSummary) useMoodStore.getState().reset();
      const sessionId = `live-${Date.now()}`;
      await startSession(sessionId, empathyLevel, interjectLevel);
    }
  }, [sessionActive, stopSession, startSession, empathyLevel, interjectLevel, sessionSummary]);

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

  // Outcome engine (Workstream 4): CANDOR's single mood item, one tap
  // before and one after. BEFORE rides to the server on the stop POST
  // (useAudioStream reads the store directly at stop); AFTER PATCHes the
  // stored episode the moment it's answered, since the POST above already
  // happened before this check is even shown.
  const beforeMood = useMoodStore((s) => s.before);
  const afterMood = useMoodStore((s) => s.after);
  const setBeforeMood = useMoodStore((s) => s.setBefore);
  const setAfterMood = useMoodStore((s) => s.setAfter);
  const handleBeforeMoodChange = useCallback(
    (value: number | null) => {
      // No episode exists yet to key persistence by — the pair is
      // persisted once the AFTER half lands (see handleAfterMoodChange).
      setBeforeMood(null, value);
    },
    [setBeforeMood],
  );
  const handleAfterMoodChange = useCallback(
    (value: number | null) => {
      const episodeId = lastEpisode?.episodeId ?? null;
      setAfterMood(episodeId, value);
      if (value !== null && episodeId) void patchSessionMood(episodeId, value);
    },
    [setAfterMood, lastEpisode],
  );

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
  // Developer mode (Settings → Diagnostics): raw states, capability and
  // latency lines. Off = the clean tester surface; nothing is lost, hidden.
  const devMode = useDevModeStore((s) => s.devMode);
  const mode: LiveMode = sessionMode ?? "earpiece";
  const isCall = mode === "call";
  // Journal mode ("listen for my voice"): no coaching, no transcript, no
  // server while it runs — the screen collapses to the journal panel + Stop.
  const isJournal = mode === "journal";
  const journalState = journal ?? IDLE_JOURNAL_STATE;
  // The honest gate: an enrolled OWNER voiceprint with at least one
  // recording pooled into it. The hook re-checks at Start (the labeler's
  // own `hasSelfPrint`); this decides what the idle screen says.
  const journalGate: JournalGate = React.useMemo(() => {
    if (people === null) return peopleError ? "unknown" : "checking";
    if (people.length === 0 && peopleError) return "unknown";
    const self = people.find((p) => p.isSelf);
    if (!self) return "missing";
    return Math.max(self.settings, self.enrollCount) >= 1 ? "ok" : "missing";
  }, [people, peopleError]);
  const journalBlocked = isJournal && !sessionActive && journalGate === "missing";
  const handleRetryJournalUploads = useCallback(() => {
    void retryJournalUploads?.();
  }, [retryJournalUploads]);

  // Deep-link executor ("Hey Google, start my journal"). Runs the delivered
  // action at most once (the ref guards dependency churn until the parent's
  // consume callback clears the prop; it re-arms when the prop clears), and
  // honors the same gates as the buttons: "start" waits for the Journal mode
  // override and the voiceprint gate check, then either starts the session or
  // — on a missing owner print — lands with JournalPanel's gate message
  // visible without starting. A mic-permission failure surfaces through
  // startSession's own micError banner exactly as a manual tap would. "stop"
  // only ever stops a running *journal* session.
  const journalActionDoneRef = useRef(false);
  useEffect(() => {
    if (!journalAction) {
      journalActionDoneRef.current = false;
      return;
    }
    if (journalActionDoneRef.current) return;
    if (journalAction === "stop") {
      journalActionDoneRef.current = true;
      onJournalActionConsumed?.();
      if (sessionActive && mode === "journal") void stopSession();
      return;
    }
    // "start": wait until the mode override has landed in the hook.
    if (mode !== "journal") return;
    if (sessionActive) {
      // Already listening — nothing to start twice.
      journalActionDoneRef.current = true;
      onJournalActionConsumed?.();
      return;
    }
    // Wait for the voiceprint gate to resolve; "missing" stays on screen with
    // the gate message (never a silent failed start), anything else starts —
    // the hook re-checks the gate at Start anyway (its own hasSelfPrint).
    if (journalGate === "checking") return;
    journalActionDoneRef.current = true;
    onJournalActionConsumed?.();
    if (journalGate === "missing") return;
    void startSession(`live-${Date.now()}`, empathyLevel, interjectLevel);
  }, [
    journalAction,
    mode,
    sessionActive,
    journalGate,
    onJournalActionConsumed,
    startSession,
    stopSession,
    empathyLevel,
    interjectLevel,
  ]);
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
      {/* Everything above the Start/Stop button scrolls: the setup content
          (mode, toggles, sliders, pre-flight card, tips) is taller than the
          screen on a phone, and before this it was clipped under the tab bar
          with no way to reach it (regression found on a Pixel 10, 2026-08-26).
          The Start/Stop button stays OUTSIDE the ScrollView as a fixed footer
          so the primary action is always reachable. */}
      <ScrollView
        style={styles.scrollArea}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        testID="live-coach-scroll"
      >
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
            {devMode ? connectionStatus : (FRIENDLY_STATUS[connectionStatus] ?? connectionStatus)}
          </Text>
        </View>
      </View>

      {/* Identity chip: which diarized voice is the user's. Shown once there's
          a session or a first transcript line — before that the toggle would
          be meaningless. Tapping flips A↔B; the hint reminds the "you speak
          first" convention while idle. Therapist mode has no "you" on the
          mic, so the chip is hidden there. */}
      {!isTherapist && !isCall && !isJournal && (sessionActive || transcript.length > 0) && (
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
      {isJournal ? (
        <JournalPanel
          state={journalState}
          sessionActive={sessionActive}
          gate={journalGate}
          onRetryUploads={handleRetryJournalUploads}
        />
      ) : null}

      {devMode && liveCapable && !isJournal ? (
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
      {devMode && liveStatus && !isJournal ? (
        <Text style={styles.speechUnavailableText} testID="live-status">
          {liveStatus}
        </Text>
      ) : null}

      {/* Pleasantness scoreboard (PRD §6): opt-in, off by default, remembered
          per account. A race to be nicer — both lines climbing is the win. */}
      {/* Speak aloud — only meaningful in a mode that would speak (therapist
          is always silent). Off keeps nudges on screen without any TTS. */}
      {sessionMode !== "therapist" && !isJournal ? (
        <View style={styles.modeRow} testID="speak-aloud-row">
          <Text style={styles.modeLabel}>Speak aloud</Text>
          <Switch
            testID="speak-aloud-switch"
            value={speakAloud}
            onValueChange={setSpeakAloud}
          />
          <Text style={styles.modeHint} numberOfLines={1}>
            {speakAloud ? "coach speaks — use an earbud" : "silent — nudges on screen"}
          </Text>
        </View>
      ) : null}
      {isJournal ? null : (
      <>
      <View style={styles.modeRow} testID="keep-audio-row">
        <Text style={styles.modeLabel}>Keep audio</Text>
        <Switch
          testID="keep-audio-switch"
          value={keepAudio}
          onValueChange={handleKeepAudioToggle}
          disabled={liveStatus === "live"}
        />
        <Text style={styles.modeHint} numberOfLines={1}>
          {keepAudio ? "saved with the session — replay + re-analyze" : "off — transcribed and thrown away"}
        </Text>
      </View>
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
      </>
      )}

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
      {sessionActive && !isJournal ? (
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

      {/* Tier B: a paired watch has its (companion) nudge socket open on the
          server — escalations on your own turns buzz the wrist even with this
          phone pocketed. */}
      {sessionActive && watchConnected ? (
        <Text style={styles.watchConnectedText} testID="watch-connected">
          ⌚ watch connected — nudges on your wrist
        </Text>
      ) : null}

      {/* Haptic nudge mirror: "you're getting loud/heated" on the user's own
          turn. Level 1–3 from the shared nudge policy. */}
      {nudgeFlash ? (
        <View style={styles.nudgeFlash} testID="nudge-flash">
          <Text style={styles.nudgeFlashText}>
            {devMode
              ? `Easy — level ${nudgeFlash.level}${
                  nudgeFlash.vectors.length > 0
                    ? ` (${nudgeFlash.vectors.join(", ").replace(/_/g, " ")})`
                    : ""
                }`
              : nudgeFlash.vectors.includes("interrupting")
                ? "Let them finish"
                : "Easy — take a breath"}
          </Text>
        </View>
      ) : null}

      {/* Server tone flag (additive to on-device coaching). */}
      {toneFlags && toneFlags.length > 0 ? (
        <Text style={styles.toneFlagText} testID="tone-flag">
          {devMode
            ? `⚠ ${toneFlags[0].speaker}: ${toneFlags[0].label} (${toneFlags[0].source} tone)`
            : `⚠ ${toneFlags[0].speaker}: ${toneFlags[0].label}`}
        </Text>
      ) : null}

      {/* Honest state: a spoken mode selected but this platform has no TTS —
          suggestions stay visual-only instead of silently pretending. */}
      {!isTherapist && !isJournal && !speechAvailable ? (
        <Text style={styles.speechUnavailableText} testID="speech-unavailable-note">
          Spoken suggestions aren&apos;t available on this platform — showing
          them on screen only.
        </Text>
      ) : null}

      {/* Empathy slider + interject: the coach's knobs — none in Journal mode. */}
      {isJournal ? null : (
        <>
          <EmpathySlider
            value={empathyLevel}
            onValueChange={handleEmpathyChange}
          />
          <InterjectSlider
            value={interjectLevel}
            onValueChange={handleInterjectChange}
          />
        </>
      )}

      {/* Idle: the honest pre-flight (what will run on this phone, who the
          loop expects to hear) and the short how-to. Disappears the moment a
          session starts or any transcript arrives. */}
      {idle && !isJournal ? (
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
          {/* One-tap BEFORE mood check (CANDOR's single outcome item) —
              right above the Start button, the app's therapy-evidence
              primitive. Gone the moment a session starts (idle flips false)
              or in Journal mode (excluded above). */}
          <MoodCheck phase="before" value={beforeMood} onChange={handleBeforeMoodChange} />
        </>
      ) : null}

      {/* Live transcript: two labelled columns in therapist mode; none in
          Journal mode (nothing is transcribed while it listens). */}
      {isJournal ? null : isTherapist ? (
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
      {!isJournal && suggestions.length > 0 && (
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
            const sourceTag = devMode && entry.source ? (
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

      {/* Session end: the one-tap AFTER mood check (CANDOR's single
          outcome item — PATCHes the stored episode the moment it's
          answered) above the summary card (duration, turns, escalations,
          first-words latency) with "Share with my therapist" when linked. */}
      {!sessionActive && !isJournal && sessionSummary ? (
        <>
          <MoodCheck phase="after" value={afterMood} onChange={handleAfterMoodChange} />
          <SessionSummaryCard
            summary={sessionSummary}
            episode={lastEpisode ?? null}
            therapist={therapist}
          />
        </>
      ) : null}

      {/* Latency report from the on-device loop (printed in full to the
          console at session end; the headline lands here). */}
      {devMode && !sessionActive && latencySummary ? (
        <Text style={styles.speechUnavailableText} testID="latency-summary">
          {latencySummary}
        </Text>
      ) : null}

      {/* Post-session review handoff: after a session ends with something to
          review, offer a prominent jump to the async-review Session screen. */}
      {!sessionActive && !isJournal && transcript.length > 0 && (
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

      </ScrollView>

      {/* Start/Stop button — fixed footer, outside the ScrollView so it is
          always reachable. Call mode starts from the call panel (Start a
          call / Join / Answer) and stops by hanging up. */}
      {isCall && !sessionActive ? null : (
        <TouchableOpacity
          testID="mic-toggle"
          style={[
            styles.micButton,
            isRecording && styles.micButtonRecording,
            journalBlocked && styles.micButtonDisabled,
          ]}
          disabled={journalBlocked}
          accessibilityState={{ disabled: journalBlocked }}
          onPress={isCall ? handleHangUp : handleToggle}
        >
          <Text style={styles.micButtonText}>
            {isCall
              ? "Hang up"
              : isJournal
                ? sessionActive
                  ? "Stop Journal"
                  : "Start Journal"
                : sessionActive
                  ? "Stop Listening"
                  : "Start Listening"}
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
  scrollArea: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 8,
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
  watchConnectedText: {
    fontSize: 12,
    color: "#047857",
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
  micButtonDisabled: {
    backgroundColor: "#9CA3AF",
  },
  micButtonText: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "700",
  },
});
