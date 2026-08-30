import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_LIVE_MODE,
  isLiveMode,
  loadLiveMode,
  modeKey,
  saveLiveMode,
} from "../src/live/modePrefs";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
});

describe("modePrefs", () => {
  it("keys by account, sanitized for SecureStore", () => {
    expect(modeKey("uid-123")).toBe("mindshift.liveMode.v1.uid-123");
    expect(modeKey("mom@example.com")).toBe("mindshift.liveMode.v1.mom_example.com");
    expect(modeKey(null)).toBe("mindshift.liveMode.v1.anon");
  });

  it("recognises only the four modes", () => {
    expect(isLiveMode("earpiece")).toBe(true);
    expect(isLiveMode("speaker")).toBe(true); // "In person" keeps its stored value
    expect(isLiveMode("therapist")).toBe(true);
    expect(isLiveMode("call")).toBe(true);
    expect(isLiveMode("visual")).toBe(false);
    expect(isLiveMode(null)).toBe(false);
  });

  it("loads the remembered mode per account and defaults when absent/unknown/failing", async () => {
    getItem.mockResolvedValueOnce("speaker");
    expect(await loadLiveMode("sage")).toBe("speaker");
    expect(getItem).toHaveBeenLastCalledWith("mindshift.liveMode.v1.sage");
    getItem.mockResolvedValueOnce("bogus");
    expect(await loadLiveMode("sage")).toBe(DEFAULT_LIVE_MODE);
    getItem.mockRejectedValueOnce(new Error("keychain locked"));
    expect(await loadLiveMode("sage")).toBe(DEFAULT_LIVE_MODE);
  });

  it("saves under the account's key and swallows storage errors", async () => {
    await saveLiveMode("mom", "therapist");
    expect(setItem).toHaveBeenCalledWith("mindshift.liveMode.v1.mom", "therapist");
    setItem.mockRejectedValueOnce(new Error("full"));
    await expect(saveLiveMode("mom", "earpiece")).resolves.toBeUndefined();
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
      await saveLiveMode("w", "speaker");
      expect(await loadLiveMode("w")).toBe("speaker");
      expect(getItem).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(Platform, "OS", { value: original, configurable: true });
      delete (globalThis as { localStorage?: unknown }).localStorage;
    }
  });
});

describe("modePrefs — journal", () => {
  it("accepts the journal mode (listen for my voice) as a remembered choice", async () => {
    expect(isLiveMode("journal")).toBe(true);
    getItem.mockResolvedValueOnce("journal");
    expect(await loadLiveMode("sage")).toBe("journal");
  });
});
