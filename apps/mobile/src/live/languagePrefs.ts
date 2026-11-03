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

const norm = (lang: string) => lang.replace("_", "-").toLowerCase();
const base = (lang: string) => norm(lang).split("-")[0];

/** Voices for the chosen coach language (phone language on Auto). Strict: an
 *  exact locale match (es-ES) wins; only if the phone has none do we widen to
 *  the same base language (es-*), and never to another language. Duplicate
 *  identifiers are dropped. */
export function filterVoices(
  voices: VoiceInfo[],
  coachLanguage: string,
  phoneLanguage: string,
): VoiceInfo[] {
  const want = norm(coachLanguage === "auto" ? phoneLanguage : coachLanguage);
  const seen = new Set<string>();
  const unique = voices.filter((v) => {
    if (!v?.identifier || seen.has(v.identifier)) return false;
    seen.add(v.identifier);
    return true;
  });
  const exact = unique.filter((v) => norm(v.language || "") === want);
  if (exact.length > 0) return exact;
  return unique.filter((v) => base(v.language || "") === base(want));
}

type DisplayNamesCtor = new (l: string[], o: object) => { of(c: string): string | undefined };

function languageLabel(tag: string): string {
  const t = tag.replace("_", "-");
  try {
    const [lang, region] = t.split("-");
    const dn = (Intl as unknown as { DisplayNames?: DisplayNamesCtor }).DisplayNames;
    if (dn) {
      const l = new dn(["en"], { type: "language" }).of(lang);
      if (l) return region ? `${l} (${region.toUpperCase()})` : l;
    }
  } catch {
    // fall through to the raw tag
  }
  return t;
}

function meaningfulName(v: VoiceInfo): boolean {
  const n = (v.name ?? "").trim();
  if (!n || n === v.identifier) return false;
  if (/[._]/.test(n) || /-language$/i.test(n) || /-x-|local|network/i.test(n)) return false;
  if (/^([a-z]{2}|[a-z]{2,3}-[A-Za-z0-9]{2,4})$/.test(n)) return false; // a bare locale
  return true;
}

export interface VoiceOption {
  identifier: string;
  label: string;
}

/** Filtered, de-duplicated voices with readable labels: the device's own name
 *  when meaningful, else "<Language> voice N" (numbered per language). */
export function voiceOptions(
  voices: VoiceInfo[],
  coachLanguage: string,
  phoneLanguage: string,
): VoiceOption[] {
  const counters: Record<string, number> = {};
  return filterVoices(voices, coachLanguage, phoneLanguage).map((v) => {
    const enhanced = /enhanced|premium/i.test(v.quality ?? "") ? " (enhanced)" : "";
    let label: string;
    if (meaningfulName(v)) {
      label = v.name.trim();
    } else if (/-language$/i.test(v.identifier)) {
      // Generic per-language placeholder some devices list; kept but labelled.
      label = `${languageLabel(v.language || "")} system voice`;
    } else {
      const k = norm(v.language || "");
      counters[k] = (counters[k] ?? 0) + 1;
      label = `${languageLabel(v.language || "")} voice ${counters[k]}`;
    }
    return { identifier: v.identifier, label: label + enhanced };
  });
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
