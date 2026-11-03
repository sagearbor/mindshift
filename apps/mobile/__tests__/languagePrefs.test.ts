import { buildPrompt, SUGGESTION_SYSTEM_PROMPT } from "../src/live/localLlm";
import * as SecureStore from "expo-secure-store";
import {
  LANGUAGES,
  loadLanguagePrefs,
  saveLanguagePrefs,
  setLanguagePrefs,
  languageInstruction,
  speakOptionsFor,
  recognizerLang,
  filterVoices,
  friendlyVoiceName,
  resolveVoice,
  type VoiceInfo,
} from "../src/live/languagePrefs";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;
beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
  setLanguagePrefs({ coachLanguage: "auto", conversationLanguage: "auto", coachVoice: null });
});

describe("languagePrefs", () => {
  it("offers the ten languages with Auto first, one constant", () => {
    expect(LANGUAGES.map((l) => l.bcp47)).toEqual([
      "auto", "en-US", "es-ES", "fr-FR", "de-DE", "pt-BR", "it-IT", "hi-IN", "ja-JP", "zh-CN",
    ]);
    expect(LANGUAGES.find((l) => l.bcp47 === "es-ES")!.englishName).toBe("Spanish");
  });

  it("defaults to auto/auto/null and ignores unknown stored values", async () => {
    expect(await loadLanguagePrefs("u")).toEqual({
      coachLanguage: "auto", conversationLanguage: "auto", coachVoice: null,
    });
    getItem.mockResolvedValue("xx-XX");
    expect((await loadLanguagePrefs("u")).coachLanguage).toBe("auto");
  });

  it("round-trips through SecureStore per account", async () => {
    await saveLanguagePrefs("u1", { coachLanguage: "fr-FR", conversationLanguage: "de-DE", coachVoice: "v1" });
    const stored: Record<string, string> = {};
    for (const [k, v] of setItem.mock.calls) stored[k as string] = v as string;
    getItem.mockImplementation(async (k: string) => stored[k] ?? null);
    expect(await loadLanguagePrefs("u1")).toEqual({
      coachLanguage: "fr-FR", conversationLanguage: "de-DE", coachVoice: "v1",
    });
  });

  it("adds a reply-language instruction only when set", () => {
    expect(languageInstruction("auto")).toBeNull();
    expect(languageInstruction("zz")).toBeNull();
    expect(languageInstruction("es-ES")).toBe("Write the suggestion in Spanish.");
  });

  it("speak options carry language/voice only when set (default is unchanged)", () => {
    expect(speakOptionsFor({ coachLanguage: "auto", conversationLanguage: "auto", coachVoice: null })).toEqual({});
    expect(speakOptionsFor({ coachLanguage: "es-ES", conversationLanguage: "auto", coachVoice: null })).toEqual({ language: "es-ES" });
    expect(speakOptionsFor({ coachLanguage: "es-ES", conversationLanguage: "auto", coachVoice: "v9" })).toEqual({ language: "es-ES", voice: "v9" });
  });

  it("recognizer lang is undefined on auto", () => {
    expect(recognizerLang("auto")).toBeUndefined();
    expect(recognizerLang("ja-JP")).toBe("ja-JP");
  });
});

describe("on-device suggestion prompt", () => {
  const base = { text: "hi", speaker: "Mom", isSelf: false, empathy: 50, context: [], mode: "earpiece" as const };
  it("is byte-identical when no reply language is set", () => {
    expect(buildPrompt(base).system).toBe(SUGGESTION_SYSTEM_PROMPT);
    expect(buildPrompt({ ...base, replyLanguage: "auto" }).system).toBe(SUGGESTION_SYSTEM_PROMPT);
  });
  it("appends the language instruction when set", () => {
    expect(buildPrompt({ ...base, replyLanguage: "fr-FR" }).system).toBe(
      `${SUGGESTION_SYSTEM_PROMPT} Write the suggestion in French.`,
    );
  });
});

describe("voices", () => {
  const voices: VoiceInfo[] = [
    { identifier: "com.apple.voice.compact.es-ES.Monica", name: "Monica", language: "es-ES", quality: "Default" },
    { identifier: "es-es-x-eea-local", name: "es-es-x-eea-local", language: "es-ES" },
    { identifier: "com.apple.voice.enhanced.en-US.Ava", name: "Ava", language: "en-US", quality: "Enhanced" },
    { identifier: "fr", name: "Amelie", language: "fr_FR" },
  ];
  it("filters by language, Auto uses the phone language", () => {
    expect(filterVoices(voices, "es-ES", "en-US").map((v) => v.name)).toEqual(["Monica", "es-es-x-eea-local"]);
    expect(filterVoices(voices, "auto", "en-GB").map((v) => v.name)).toEqual(["Ava"]);
    expect(filterVoices(voices, "fr-FR", "en-US").map((v) => v.name)).toEqual(["Amelie"]);
  });
  it("friendly names strip identifiers and mark enhanced", () => {
    expect(friendlyVoiceName(voices[2])).toBe("Ava (enhanced)");
    expect(friendlyVoiceName(voices[0])).toBe("Monica");
    expect(friendlyVoiceName(voices[1])).toBe("Voice (es-ES)");
  });
  it("resolveVoice keeps an available voice and drops a vanished one", () => {
    expect(resolveVoice("fr", voices, "fr-FR", "en-US")).toBe("fr");
    expect(resolveVoice("gone", voices, "fr-FR", "en-US")).toBeNull();
    expect(resolveVoice("fr", voices, "es-ES", "en-US")).toBeNull();
    expect(resolveVoice(null, voices, "fr-FR", "en-US")).toBeNull();
  });
});
