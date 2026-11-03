import React, { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import * as Speech from "expo-speech";
import {
  LANGUAGES,
  getLanguagePrefs,
  loadLanguagePrefs,
  resolveVoice,
  saveLanguagePrefs,
  setLanguagePrefs,
  subscribeLanguagePrefs,
  voiceOptions,
  type LanguagePrefs,
  type VoiceInfo,
} from "../live/languagePrefs";

const SAMPLE_LINE = "Hi, I'm your coach.";
/** Voices listed before "Show more". */
export const VOICES_SHOWN = 6;

function phoneLanguage(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().locale || "en-US";
  } catch {
    return "en-US";
  }
}

const labelOf = (code: string) => LANGUAGES.find((l) => l.bcp47 === code)?.label ?? "Auto";

/** A collapsed-by-default row whose header shows the current value. */
function Accordion({
  testID,
  title,
  value,
  children,
}: {
  testID: string;
  title: string;
  value?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <View testID={testID} style={styles.acc}>
      <TouchableOpacity
        testID={`${testID}-header`}
        accessibilityRole="button"
        accessibilityState={{ expanded: open }}
        onPress={() => setOpen((o) => !o)}
        style={styles.accHeader}
      >
        <Text style={styles.label}>{value !== undefined ? `${title}: ${value}` : title}</Text>
        <Text style={styles.chevron}>{open ? "▲" : "▼"}</Text>
      </TouchableOpacity>
      {open ? <View style={styles.accBody}>{children}</View> : null}
    </View>
  );
}

function Chips({
  testID,
  value,
  onChange,
}: {
  testID: string;
  value: string;
  onChange: (code: string) => void;
}) {
  return (
    <View style={styles.chips} testID={testID}>
      {LANGUAGES.map((l) => (
        <TouchableOpacity
          key={l.bcp47}
          testID={`${testID}-${l.bcp47}`}
          accessibilityRole="button"
          accessibilityState={{ selected: value === l.bcp47 }}
          style={[styles.chip, value === l.bcp47 && styles.chipOn]}
          onPress={() => onChange(l.bcp47)}
        >
          <Text style={[styles.chipText, value === l.bcp47 && styles.chipTextOn]}>{l.label}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

/** One collapsed "Language" group: coach language, conversation language and
 *  coach voice, each its own collapsed row showing the current value. */
export default function LanguageSettings({ userId }: { userId: string | null }) {
  const prefs = useSyncExternalStore(subscribeLanguagePrefs, getLanguagePrefs);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void loadLanguagePrefs(userId).then((p) => {
      if (!cancelled) setLanguagePrefs(p);
    });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  useEffect(() => {
    let cancelled = false;
    void Promise.resolve(Speech.getAvailableVoicesAsync?.())
      .then((vs) => {
        if (!cancelled) setVoices((vs ?? []) as VoiceInfo[]);
      })
      .catch(() => {
        if (!cancelled) setVoices([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const update = useCallback(
    (patch: Partial<LanguagePrefs>) => {
      const next = { ...getLanguagePrefs(), ...patch };
      setLanguagePrefs(next);
      void saveLanguagePrefs(userId, next);
    },
    [userId],
  );

  const phone = phoneLanguage();
  const offered = voiceOptions(voices, prefs.coachLanguage, phone);
  const visible = showAll ? offered : offered.slice(0, VOICES_SHOWN);
  const voiceLabel = prefs.coachVoice
    ? offered.find((o) => o.identifier === prefs.coachVoice)?.label ?? "Default"
    : "Default";

  // A stored voice that this phone/language no longer offers: clear it.
  useEffect(() => {
    if (voices.length === 0 || !prefs.coachVoice) return;
    if (resolveVoice(prefs.coachVoice, voices, prefs.coachLanguage, phone) === null) {
      update({ coachVoice: null });
    }
  }, [voices, prefs.coachVoice, prefs.coachLanguage, phone, update]);

  const playSample = () => {
    try {
      Speech.stop();
      Speech.speak(SAMPLE_LINE, {
        ...(prefs.coachLanguage !== "auto" ? { language: prefs.coachLanguage } : {}),
        ...(prefs.coachVoice ? { voice: prefs.coachVoice } : {}),
      });
    } catch {
      // No TTS here: nothing to play, and nothing pretended.
    }
  };

  return (
    <View style={styles.card} testID="language-settings">
      <Accordion testID="language-group" title="Language">
        <Accordion
          testID="coach-language-acc"
          title="Coach language"
          value={labelOf(prefs.coachLanguage)}
        >
          <Text style={styles.hint}>What the coach writes and speaks. Auto = English.</Text>
          <Chips
            testID="coach-language"
            value={prefs.coachLanguage}
            onChange={(c) => update({ coachLanguage: c, coachVoice: null })}
          />
        </Accordion>

        <Accordion
          testID="conversation-language-acc"
          title="Conversation language"
          value={labelOf(prefs.conversationLanguage)}
        >
          <Text style={styles.hint}>
            What the people you talk with speak. Auto = recognizer default.
          </Text>
          <Chips
            testID="conversation-language"
            value={prefs.conversationLanguage}
            onChange={(c) => update({ conversationLanguage: c })}
          />
        </Accordion>

        {offered.length > 0 ? (
          <Accordion testID="coach-voice-row" title="Coach voice" value={voiceLabel}>
            <View style={styles.chips}>
              <TouchableOpacity
                testID="coach-voice-default"
                style={[styles.chip, !prefs.coachVoice && styles.chipOn]}
                onPress={() => update({ coachVoice: null })}
              >
                <Text style={[styles.chipText, !prefs.coachVoice && styles.chipTextOn]}>
                  Default
                </Text>
              </TouchableOpacity>
              {visible.map((v) => (
                <TouchableOpacity
                  key={v.identifier}
                  testID={`coach-voice-${v.identifier}`}
                  style={[styles.chip, prefs.coachVoice === v.identifier && styles.chipOn]}
                  onPress={() => update({ coachVoice: v.identifier })}
                >
                  <Text
                    style={[styles.chipText, prefs.coachVoice === v.identifier && styles.chipTextOn]}
                  >
                    {v.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            {offered.length > VOICES_SHOWN ? (
              <TouchableOpacity
                testID="coach-voice-more"
                accessibilityRole="button"
                style={styles.sample}
                onPress={() => setShowAll((s) => !s)}
              >
                <Text style={styles.sampleText}>
                  {showAll ? "Show fewer" : `Show more (${offered.length - VOICES_SHOWN})`}
                </Text>
              </TouchableOpacity>
            ) : null}
            <TouchableOpacity
              testID="coach-voice-sample"
              accessibilityRole="button"
              style={styles.sample}
              onPress={playSample}
            >
              <Text style={styles.sampleText}>Play sample</Text>
            </TouchableOpacity>
          </Accordion>
        ) : null}
      </Accordion>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 18,
    paddingVertical: 6,
    marginBottom: 12,
  },
  acc: { borderBottomWidth: 0 },
  accHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    minHeight: 48,
  },
  accBody: { paddingBottom: 12 },
  chevron: { fontSize: 12, color: "#6B7280", marginLeft: 8 },
  label: { fontSize: 16, fontWeight: "600", color: "#111827", flexShrink: 1 },
  hint: { fontSize: 13, color: "#6B7280", marginBottom: 8 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#D1D5DB",
    backgroundColor: "#F9FAFB",
  },
  chipOn: { backgroundColor: "#2563EB", borderColor: "#2563EB" },
  chipText: { fontSize: 14, color: "#374151" },
  chipTextOn: { color: "#FFFFFF" },
  sample: { marginTop: 10, alignSelf: "flex-start", paddingVertical: 8 },
  sampleText: { fontSize: 14, color: "#2563EB", fontWeight: "600" },
});
