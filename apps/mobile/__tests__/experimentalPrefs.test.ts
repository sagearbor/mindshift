import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_EXPERIMENTAL_VOICE_ENGINE,
  experimentalVoiceEngineKey,
  loadExperimentalVoiceEngine,
  saveExperimentalVoiceEngine,
} from "../src/live/experimentalPrefs";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
});

describe("experimentalPrefs (Experimental voice engine)", () => {
  it("is OFF by default — a research instrument, not a product feature", () => {
    expect(DEFAULT_EXPERIMENTAL_VOICE_ENGINE).toBe(false);
  });

  it("keys by account, sanitized for SecureStore", () => {
    expect(experimentalVoiceEngineKey("uid-123")).toBe("mindshift.experimentalVoiceEngine.v1.uid-123");
    expect(experimentalVoiceEngineKey("mom@example.com")).toBe("mindshift.experimentalVoiceEngine.v1.mom_example.com");
    expect(experimentalVoiceEngineKey(null)).toBe("mindshift.experimentalVoiceEngine.v1.anon");
  });

  it("loads the remembered choice and defaults to off when absent/unknown/failing", async () => {
    getItem.mockResolvedValueOnce("on");
    expect(await loadExperimentalVoiceEngine("sage")).toBe(true);
    expect(getItem).toHaveBeenLastCalledWith("mindshift.experimentalVoiceEngine.v1.sage");
    getItem.mockResolvedValueOnce("off");
    expect(await loadExperimentalVoiceEngine("sage")).toBe(false);
    getItem.mockResolvedValueOnce("maybe");
    expect(await loadExperimentalVoiceEngine("sage")).toBe(false);
    getItem.mockRejectedValueOnce(new Error("keychain locked"));
    expect(await loadExperimentalVoiceEngine("sage")).toBe(false);
  });

  it("saves under the account's key and swallows storage errors", async () => {
    await saveExperimentalVoiceEngine("mom", true);
    expect(setItem).toHaveBeenCalledWith("mindshift.experimentalVoiceEngine.v1.mom", "on");
    await saveExperimentalVoiceEngine("mom", false);
    expect(setItem).toHaveBeenCalledWith("mindshift.experimentalVoiceEngine.v1.mom", "off");
    setItem.mockRejectedValueOnce(new Error("full"));
    await expect(saveExperimentalVoiceEngine("mom", true)).resolves.toBeUndefined();
  });

  it("uses localStorage on web", async () => {
    const original = Platform.OS;
    Object.defineProperty(Platform, "OS", { value: "web", configurable: true });
    const store = new Map<string, string>();
    (globalThis as { localStorage?: unknown }).localStorage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    };
    try {
      await saveExperimentalVoiceEngine("w", true);
      expect(store.get("mindshift.experimentalVoiceEngine.v1.w")).toBe("on");
      expect(await loadExperimentalVoiceEngine("w")).toBe(true);
      expect(setItem).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(Platform, "OS", { value: original, configurable: true });
      delete (globalThis as { localStorage?: unknown }).localStorage;
    }
  });
});
