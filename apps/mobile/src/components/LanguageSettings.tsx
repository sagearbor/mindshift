import React, { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import * as Speech from "expo-speech";
import {
  LANGUAGES,
  filterVoices,
  friendlyVoiceName,
  getLanguagePrefs,
  loadLanguagePrefs,
  resolveVoice,
  saveLanguagePrefs,
  setLanguagePrefs,
  subscribeLanguagePrefs,
  type LanguagePrefs,
  type VoiceInfo,
} from "../live/languagePrefs";

const SAMPLE_LINE = "Hi, I'm your coach.";

function phoneLanguage(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().locale || "en-US";
  } catch {
    return "en-US";
  }
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

/** One "Language" group: coach language, conversation language, coach voice. */
export default function LanguageSettings({ userId }: { userId: string | null }) {
  const prefs = useSyncExternalStore(subscribeLanguagePrefs, getLanguagePrefs);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);

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
  const offered = filterVoices(voices, prefs.coachLanguage, phone);

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
      <Text style={styles.label}>Coach language</Text>
      <Text style={styles.hint}>What the coach writes and speaks. Auto = English.</Text>
      <Chips
        testID="coach-language"
        value={prefs.coachLanguage}
        onChange={(c) => update({ coachLanguage: c, coachVoice: null })}
      />

      <Text style={[styles.label, styles.gap]}>Conversation language</Text>
      <Text style={styles.hint}>What the people you talk with speak. Auto = recognizer default.</Text>
      <Chips
        testID="conversation-language"
        value={prefs.conversationLanguage}
        onChange={(c) => update({ conversationLanguage: c })}
      />

      {offered.length > 0 ? (
        <View testID="coach-voice-row">
          <Text style={[styles.label, styles.gap]}>Coach voice</Text>
          <View style={styles.chips}>
            <TouchableOpacity
              testID="coach-voice-default"
              style={[styles.chip, !prefs.coachVoice && styles.chipOn]}
              onPress={() => update({ coachVoice: null })}
            >
              <Text style={[styles.chipText, !prefs.coachVoice && styles.chipTextOn]}>Default</Text>
            </TouchableOpacity>
            {offered.map((v) => (
              <TouchableOpacity
                key={v.identifier}
                testID={`coach-voice-${v.identifier}`}
                style={[styles.chip, prefs.coachVoice === v.identifier && styles.chipOn]}
                onPress={() => update({ coachVoice: v.identifier })}
              >
                <Text
                  style={[styles.chipText, prefs.coachVoice === v.identifier && styles.chipTextOn]}
                >
                  {friendlyVoiceName(v)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          <TouchableOpacity
            testID="coach-voice-sample"
            accessibilityRole="button"
            style={styles.sample}
            onPress={playSample}
          >
            <Text style={styles.sampleText}>Play sample</Text>
          </TouchableOpacity>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    borderColor: "#D1D5DB",
    borderRadius: 14,
    backgroundColor: "#FFFFFF",
    padding: 18,
    marginBottom: 12,
  },
  label: { fontSize: 16, fontWeight: "600", color: "#111827" },
  hint: { fontSize: 13, color: "#6B7280", marginTop: 2, marginBottom: 8 },
  gap: { marginTop: 14 },
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
