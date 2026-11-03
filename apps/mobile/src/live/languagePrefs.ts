/**
 * Language prefs: COACH language (what the coach writes and speaks), CONVERSATION
 * language (what the people speak; passed to speech recognition) and the
 * optional COACH VOICE. Default "auto"/null = the behaviour before these
 * existed (English prompts, phone default voice, recognizer default).
 *
 * Persisted per account like modePrefs.ts. A module-level current value lets
 * the audio hook / fast loop read the choice synchronously at use time.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export interface LanguageOption {
  label: string;
  bcp47: string; // "auto" = unset
  englishName: string;
}

export const LANGUAGES: readonly LanguageOption[] = [
  { label: "Auto", bcp47: "auto", englishName: "" },
  { label: "English", bcp47: "en-US", englishName: "English" },
  { label: "Spanish", bcp47: "es-ES", englishName: "Spanish" },
  { label: "French", bcp47: "fr-FR", englishName: "French" },
  { label: "German", bcp47: "de-DE", englishName: "German" },
  { label: "Portuguese", bcp47: "pt-BR", englishName: "Portuguese" },
  { label: "Italian", bcp47: "it-IT", englishName: "Italian" },
  { label: "Hindi", bcp47: "hi-IN", englishName: "Hindi" },
  { label: "Japanese", bcp47: "ja-JP", englishName: "Japanese" },
  { label: "Mandarin", bcp47: "zh-CN", englishName: "Mandarin Chinese" },
];

export interface LanguagePrefs {
  coachLanguage: string;
  conversationLanguage: string;
  coachVoice: string | null;
}

export const DEFAULT_LANGUAGE_PREFS: LanguagePrefs = {
  coachLanguage: "auto",
  conversationLanguage: "auto",
  coachVoice: null,
};

export function isLanguageCode(v: unknown): v is string {
  return typeof v === "string" && LANGUAGES.some((l) => l.bcp47 === v);
}

const idOf = (userId: string | null | undefined) =>
  (userId || "anon").replace(/[^A-Za-z0-9._-]/g, "_");
export const languageKey = (userId: string | null | undefined, field: string) =>
  `mindshift.lang.${field}.v1.${idOf(userId)}`;

function webStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}
async function read(key: string): Promise<string | null> {
  try {
    return Platform.OS === "web"
      ? webStorage()?.getItem(key) ?? null
      : await SecureStore.getItemAsync(key);
  } catch {
    return null;
  }
}
async function write(key: string, value: string): Promise<void> {
  try {
    if (Platform.OS === "web") webStorage()?.setItem(key, value);
    else await SecureStore.setItemAsync(key, value);
  } catch {
    // Fail-open: the choice still applies to this session.
  }
}

export async function loadLanguagePrefs(userId: string | null | undefined): Promise<LanguagePrefs> {
  const [c, v, voice] = await Promise.all([
    read(languageKey(userId, "coach")),
    read(languageKey(userId, "conversation")),
    read(languageKey(userId, "voice")),
  ]);
  return {
    coachLanguage: isLanguageCode(c) ? c : "auto",
    conversationLanguage: isLanguageCode(v) ? v : "auto",
    coachVoice: voice ? voice : null,
  };
}

export async function saveLanguagePrefs(
  userId: string | null | undefined,
  prefs: LanguagePrefs,
): Promise<void> {
  await Promise.all([
    write(languageKey(userId, "coach"), prefs.coachLanguage),
    write(languageKey(userId, "conversation"), prefs.conversationLanguage),
    write(languageKey(userId, "voice"), prefs.coachVoice ?? ""),
  ]);
}

// ---- synchronous current value --------------------------------------------

let current: LanguagePrefs = { ...DEFAULT_LANGUAGE_PREFS };
const listeners = new Set<() => void>();
export const getLanguagePrefs = (): LanguagePrefs => current;
export function setLanguagePrefs(next: LanguagePrefs): void {
  current = { ...next };
  listeners.forEach((l) => l());
}
export function subscribeLanguagePrefs(cb: () => void): () => void {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

// One-line visible note when the OS can't do a chosen language.
let note: string | null = null;
export const getLanguageNote = () => note;
export function setLanguageNote(n: string | null): void {
  note = n;
  listeners.forEach((l) => l());
}

// ---- pure helpers ------------------------------------------------------------

/** "Write the suggestion in Spanish." or null when unset/unknown. */
export function languageInstruction(code: string | null | undefined): string | null {
  const l = LANGUAGES.find((x) => x.bcp47 === code);
  return l && l.bcp47 !== "auto" ? `Write the suggestion in ${l.englishName}.` : null;
}

export function recognizerLang(code: string | null | undefined): string | undefined {
  return isLanguageCode(code) && code !== "auto" ? code : undefined;
}

/** Options for Speech.speak beyond onError; {} on defaults. */
export function speakOptionsFor(p: LanguagePrefs): { language?: string; voice?: string } {
  const out: { language?: string; voice?: string } = {};
  if (p.coachLanguage !== "auto") out.language = p.coachLanguage;
  if (p.coachVoice) out.voice = p.coachVoice;
  return out;
}

export interface VoiceInfo {
  identifier: string;
  name: string;
  language: string;
  quality?: string;
}

const base = (lang: string) => lang.replace("_", "-").split("-")[0].toLowerCase();

/** Voices for the chosen coach language, or the phone language on Auto. */
export function filterVoices(
  voices: VoiceInfo[],
  coachLanguage: string,
  phoneLanguage: string,
): VoiceInfo[] {
  const want = base(coachLanguage === "auto" ? phoneLanguage : coachLanguage);
  return voices.filter((v) => base(v.language || "") === want);
}

export function friendlyVoiceName(v: VoiceInfo): string {
  const looksLikeId = !v.name || /[._]/.test(v.name) || /-x-|local|network/i.test(v.name);
  const lang = (v.language || "").replace("_", "-");
  const name = looksLikeId ? `Voice (${lang})` : v.name;
  return /enhanced|premium/i.test(v.quality ?? "") ? `${name} (enhanced)` : name;
}

/** The stored voice if it is still offered for this language, else null. */
export function resolveVoice(
  stored: string | null,
  voices: VoiceInfo[],
  coachLanguage: string,
  phoneLanguage: string,
): string | null {
  if (!stored) return null;
  return filterVoices(voices, coachLanguage, phoneLanguage).some((v) => v.identifier === stored)
    ? stored
    : null;
}
