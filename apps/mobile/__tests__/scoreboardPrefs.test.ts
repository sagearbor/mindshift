import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_SCOREBOARD_VISIBLE,
  loadScoreboardVisible,
  saveScoreboardVisible,
  scoreboardKey,
} from "../src/live/scoreboardPrefs";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
});

describe("scoreboardPrefs", () => {
  it("is off by default — a game two people opt into", () => {
    expect(DEFAULT_SCOREBOARD_VISIBLE).toBe(false);
  });

  it("keys by account, sanitized for SecureStore", () => {
    expect(scoreboardKey("uid-123")).toBe("mindshift.scoreboard.v1.uid-123");
    expect(scoreboardKey("mom@example.com")).toBe("mindshift.scoreboard.v1.mom_example.com");
    expect(scoreboardKey(null)).toBe("mindshift.scoreboard.v1.anon");
  });

  it("loads the remembered choice per account and defaults when absent/unknown/failing", async () => {
    getItem.mockResolvedValueOnce("on");
    expect(await loadScoreboardVisible("sage")).toBe(true);
    expect(getItem).toHaveBeenLastCalledWith("mindshift.scoreboard.v1.sage");
    getItem.mockResolvedValueOnce("off");
    expect(await loadScoreboardVisible("sage")).toBe(false);
    getItem.mockResolvedValueOnce("maybe");
    expect(await loadScoreboardVisible("sage")).toBe(false);
    getItem.mockRejectedValueOnce(new Error("keychain locked"));
    expect(await loadScoreboardVisible("sage")).toBe(false);
  });

  it("saves under the account's key and swallows storage errors", async () => {
    await saveScoreboardVisible("mom", true);
    expect(setItem).toHaveBeenCalledWith("mindshift.scoreboard.v1.mom", "on");
    await saveScoreboardVisible("mom", false);
    expect(setItem).toHaveBeenCalledWith("mindshift.scoreboard.v1.mom", "off");
    setItem.mockRejectedValueOnce(new Error("full"));
    await expect(saveScoreboardVisible("mom", true)).resolves.toBeUndefined();
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
      await saveScoreboardVisible("w", true);
      expect(store.get("mindshift.scoreboard.v1.w")).toBe("on");
      expect(await loadScoreboardVisible("w")).toBe(true);
      expect(setItem).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(Platform, "OS", { value: original, configurable: true });
      delete (globalThis as { localStorage?: unknown }).localStorage;
    }
  });
});
