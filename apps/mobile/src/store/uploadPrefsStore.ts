import { create } from "zustand";
import * as SecureStore from "expo-secure-store";

/**
 * Persisted upload preferences for the Analyze flow.
 *
 * `sendOriginalQuality` is the visible "Send original quality" override. It
 * defaults to OFF, meaning we compress a large local video on-device before
 * upload to save mobile data. Turning it ON uploads the file untouched. It's
 * persisted (via expo-secure-store, the same backend the app already uses) so
 * the choice survives app restarts, and hydrated once on the Analyze screen's
 * mount. Until `hydrate()` resolves, `sendOriginalQuality` reads its default
 * (OFF) — the safe, data-saving choice.
 */
const STORAGE_KEY = "mindshift.upload.sendOriginalQuality";

interface UploadPrefsState {
  sendOriginalQuality: boolean;
  /** True once the persisted value has been read (or found absent). */
  hydrated: boolean;
  /** Set + persist the preference. Persist is best-effort; a write failure
   *  leaves the in-memory value updated so the current session still honors it. */
  setSendOriginalQuality: (value: boolean) => void;
  /** Load the persisted value once (idempotent-ish; safe to call on each mount).
   *  A read failure just leaves the default and marks hydrated so the UI settles. */
  hydrate: () => Promise<void>;
}

export const useUploadPrefsStore = create<UploadPrefsState>((set) => ({
  sendOriginalQuality: false,
  hydrated: false,
  setSendOriginalQuality: (value) => {
    set({ sendOriginalQuality: value });
    // Fire-and-forget persist; the in-memory value is the source of truth for
    // the live session, so a keystore hiccup never blocks the toggle.
    void SecureStore.setItemAsync(STORAGE_KEY, value ? "1" : "0").catch(() => {
      // ignore — the preference still applies this session
    });
  },
  hydrate: async () => {
    try {
      const stored = await SecureStore.getItemAsync(STORAGE_KEY);
      if (stored !== null) {
        set({ sendOriginalQuality: stored === "1" });
      }
    } catch {
      // ignore — fall back to the OFF default
    } finally {
      set({ hydrated: true });
    }
  },
}));
