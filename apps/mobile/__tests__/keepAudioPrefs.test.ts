import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_KEEP_AUDIO,
  loadKeepAudio,
  saveKeepAudio,
  keepAudioKey,
} from "../src/live/keepAudioPrefs";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
});

describe("keepAudioPrefs", () => {
  it("is ON by default — you'll likely want to go back to a session you recorded", () => {
    expect(DEFAULT_KEEP_AUDIO).toBe(true);
  });

  it("keys by account, sanitized for SecureStore", () => {
    expect(keepAudioKey("uid-123")).toBe("mindshift.keepAudio.v1.uid-123");
    expect(keepAudioKey("mom@example.com")).toBe("mindshift.keepAudio.v1.mom_example.com");
    expect(keepAudioKey(null)).toBe("mindshift.keepAudio.v1.anon");
  });

  it("loads the remembered choice per account and defaults when absent/unknown/failing", async () => {
    getItem.mockResolvedValueOnce("on");
    expect(await loadKeepAudio("sage")).toBe(true);
    expect(getItem).toHaveBeenLastCalledWith("mindshift.keepAudio.v1.sage");
    getItem.mockResolvedValueOnce("off");
    expect(await loadKeepAudio("sage")).toBe(false);
    getItem.mockResolvedValueOnce("maybe");
    expect(await loadKeepAudio("sage")).toBe(true);
    getItem.mockRejectedValueOnce(new Error("keychain locked"));
    expect(await loadKeepAudio("sage")).toBe(true);
  });

  it("saves under the account's key and swallows storage errors", async () => {
    await saveKeepAudio("mom", true);
    expect(setItem).toHaveBeenCalledWith("mindshift.keepAudio.v1.mom", "on");
    await saveKeepAudio("mom", false);
    expect(setItem).toHaveBeenCalledWith("mindshift.keepAudio.v1.mom", "off");
    setItem.mockRejectedValueOnce(new Error("full"));
    await expect(saveKeepAudio("mom", true)).resolves.toBeUndefined();
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
      await saveKeepAudio("w", true);
      expect(store.get("mindshift.keepAudio.v1.w")).toBe("on");
      expect(await loadKeepAudio("w")).toBe(true);
      expect(setItem).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(Platform, "OS", { value: original, configurable: true });
      delete (globalThis as { localStorage?: unknown }).localStorage;
    }
  });
});
