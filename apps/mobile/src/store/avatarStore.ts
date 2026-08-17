/**
 * Persisted selfie-avatar uri (Task N6 of P3-10 — the owner's "shows you're
 * logged in as YOU by your face", P3-9 RESOLVED).
 *
 * This store only ever holds a URI STRING (or null — no photo set): the
 * actual capture/permission/preview flow lives in AvatarCaptureScreen, and
 * the actual file-move-into-permanent-storage call lives there too (an
 * injectable dependency, same DI seam VoiceTrainingFlow.ts uses for its own
 * expo-file-system write — see that file's `saveWav` deps). Keeping this
 * store dumb (persist/clear a string) makes it trivial to unit-test fully,
 * the same split layoutStore.ts uses between pure state/persistence and the
 * native specifics that live in the screens that call it.
 *
 * Persistence mirrors layoutStore.ts / onboardingStorage.ts's established
 * cross-platform pattern exactly rather than adding a new dependency:
 *  - native (iOS/Android): expo-secure-store
 *  - web: window.localStorage (via globalThis — see onboardingStorage.ts's
 *    comment on why not `window`)
 *
 * Fail-open, always: a storage error never throws — `hydrate()` falls back
 * to null (the honest initial-letter avatar), and a write/delete failure is
 * swallowed (best-effort — the in-memory state already updated, so the UI is
 * correct for this session even if it doesn't survive a relaunch).
 */
import { create } from "zustand";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

export const KEY = "mindshift.avatar.uri.v1";

/** `globalThis.localStorage`, not `window.localStorage` — see
 *  onboardingStorage.ts for why (no `window` in this repo's default jest
 *  environment). */
function webLocalStorage(): Storage | null {
  const g = globalThis as unknown as { localStorage?: Storage };
  return g.localStorage ?? null;
}

async function readRaw(): Promise<string | null> {
  try {
    return Platform.OS === "web"
      ? (webLocalStorage()?.getItem(KEY) ?? null)
      : await SecureStore.getItemAsync(KEY);
  } catch {
    return null;
  }
}

async function writeRaw(value: string): Promise<void> {
  try {
    if (Platform.OS === "web") {
      webLocalStorage()?.setItem(KEY, value);
      return;
    }
    await SecureStore.setItemAsync(KEY, value);
  } catch {
    // Best-effort — a write failure just means the photo may revert to the
    // initial-letter avatar next launch, which is safe, never a crash.
  }
}

async function clearRaw(): Promise<void> {
  try {
    if (Platform.OS === "web") {
      webLocalStorage()?.removeItem(KEY);
      return;
    }
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    // Best-effort — see writeRaw's comment above.
  }
}

/** Best-effort delete of the on-disk file behind a previously-set avatar
 *  uri. Never throws: a leftover file (e.g. this call landing on a mock or a
 *  platform without the module) is harmless once its uri is no longer
 *  persisted anywhere — nothing else in the app can reach it. Lazy-required,
 *  same pattern VoiceTrainingFlow.ts uses, so this module never touches
 *  native code at import time. */
async function deleteAvatarFile(uri: string): Promise<void> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { File } = require("expo-file-system");
    const file = new File(uri);
    if (file.exists) file.delete();
  } catch {
    // Best-effort — see doc comment above.
  }
}

interface AvatarState {
  /** The captured selfie's persisted local uri, or null when none has been
   *  set — Avatar.tsx's honest fallback is the account's initial letter. */
  uri: string | null;
  /** True once `hydrate()` has completed at least once — same gate pattern
   *  layoutStore/authStore use, so a future caller can avoid flashing the
   *  no-photo state over a real saved one. */
  hydrated: boolean;

  /** Load the persisted uri (or null, fail-open) and apply it. Idempotent;
   *  call once from App-level init, same as layoutStore.hydrate(). */
  hydrate: () => Promise<void>;
  /** Record a freshly-persisted photo's final uri (already moved into
   *  permanent app storage by the caller) and save it. Retaking calls this
   *  again — the caller reuses the same fixed filename, so this is always an
   *  overwrite, never an accumulation of stale files. */
  setPhoto: (uri: string) => void;
  /** Clear the photo — Avatar.tsx reverts to the initial-letter fallback —
   *  and best-effort delete the underlying file. */
  removePhoto: () => void;
}

export const useAvatarStore = create<AvatarState>((set, get) => ({
  uri: null,
  hydrated: false,

  hydrate: async () => {
    const uri = await readRaw();
    set({ uri, hydrated: true });
  },

  setPhoto: (uri) => {
    set({ uri });
    void writeRaw(uri);
  },

  removePhoto: () => {
    const prev = get().uri;
    set({ uri: null });
    void clearRaw();
    if (prev) void deleteAvatarFile(prev);
  },
}));
