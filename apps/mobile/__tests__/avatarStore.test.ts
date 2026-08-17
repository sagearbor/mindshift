import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { useAvatarStore, KEY } from "../src/store/avatarStore";

const originalOS = Platform.OS;

function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

/** Matches layoutStore.test.ts's polyfill — this repo's default jest
 *  environment has no real `localStorage`. */
function installFakeLocalStorage() {
  const store = new Map<string, string>();
  const fake: Partial<Storage> = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => {
      store.set(k, v);
    },
    removeItem: (k: string) => {
      store.delete(k);
    },
    clear: () => store.clear(),
  };
  (globalThis as unknown as { localStorage: Storage }).localStorage =
    fake as Storage;
}

function uninstallFakeLocalStorage() {
  delete (globalThis as unknown as { localStorage?: Storage }).localStorage;
}

/** Reset the singleton store's in-memory state between tests (zustand
 *  stores persist across tests otherwise — they're module singletons). */
function resetStoreState() {
  useAvatarStore.setState({ uri: null, hydrated: false });
}

beforeEach(() => {
  installFakeLocalStorage();
  resetStoreState();
});

afterEach(() => {
  setPlatform(originalOS);
  jest.clearAllMocks();
  uninstallFakeLocalStorage();
});

describe("useAvatarStore defaults", () => {
  it("starts with no photo (the honest initial-letter fallback) and not hydrated", () => {
    const state = useAvatarStore.getState();
    expect(state.uri).toBeNull();
    expect(state.hydrated).toBe(false);
  });
});

describe("useAvatarStore.setPhoto / removePhoto (in-memory)", () => {
  it("setPhoto sets the uri", () => {
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    expect(useAvatarStore.getState().uri).toBe(
      "file:///doc/avatar/profile.jpg",
    );
  });

  it("removePhoto clears the uri back to null", () => {
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    useAvatarStore.getState().removePhoto();
    expect(useAvatarStore.getState().uri).toBeNull();
  });

  it("removePhoto is a no-op (never throws) when there was no photo", () => {
    expect(() => useAvatarStore.getState().removePhoto()).not.toThrow();
    expect(useAvatarStore.getState().uri).toBeNull();
  });

  it("retaking (setPhoto again) overwrites rather than accumulates", () => {
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    expect(useAvatarStore.getState().uri).toBe(
      "file:///doc/avatar/profile.jpg",
    );
  });
});

describe("useAvatarStore persistence round-trip — native (expo-secure-store)", () => {
  beforeEach(() => setPlatform("ios"));

  it("hydrate() leaves uri null when nothing was ever saved", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(null);
    await useAvatarStore.getState().hydrate();
    expect(useAvatarStore.getState().uri).toBeNull();
    expect(useAvatarStore.getState().hydrated).toBe(true);
  });

  it("round-trips a set photo through setPhoto -> persist -> hydrate", async () => {
    let stored: string | null = null;
    (SecureStore.setItemAsync as jest.Mock).mockImplementation(
      async (_key: string, value: string) => {
        stored = value;
      },
    );
    (SecureStore.getItemAsync as jest.Mock).mockImplementation(
      async () => stored,
    );

    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    // setPhoto persists fire-and-forget; give the microtask queue a turn.
    await Promise.resolve();
    await Promise.resolve();

    // Simulate a fresh app launch reading the persisted value back.
    resetStoreState();
    await useAvatarStore.getState().hydrate();

    expect(useAvatarStore.getState().uri).toBe(
      "file:///doc/avatar/profile.jpg",
    );
    expect(SecureStore.setItemAsync).toHaveBeenCalledWith(
      KEY,
      "file:///doc/avatar/profile.jpg",
    );
  });

  it("removePhoto clears the persisted key so a later hydrate reads null", async () => {
    let stored: string | null = "file:///doc/avatar/profile.jpg";
    (SecureStore.deleteItemAsync as jest.Mock).mockImplementation(async () => {
      stored = null;
    });
    (SecureStore.getItemAsync as jest.Mock).mockImplementation(
      async () => stored,
    );

    useAvatarStore.setState({ uri: "file:///doc/avatar/profile.jpg" });
    useAvatarStore.getState().removePhoto();
    await Promise.resolve();
    await Promise.resolve();

    resetStoreState();
    await useAvatarStore.getState().hydrate();
    expect(useAvatarStore.getState().uri).toBeNull();
    expect(SecureStore.deleteItemAsync).toHaveBeenCalledWith(KEY);
  });

  it("never crashes when the storage read throws", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    await expect(useAvatarStore.getState().hydrate()).resolves.toBeUndefined();
    expect(useAvatarStore.getState().uri).toBeNull();
    expect(useAvatarStore.getState().hydrated).toBe(true);
  });

  it("swallows a write failure rather than throwing out of setPhoto", async () => {
    (SecureStore.setItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    expect(() =>
      useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg"),
    ).not.toThrow();
    await Promise.resolve();
    await Promise.resolve();
    // The in-memory state still updated even though the persist failed.
    expect(useAvatarStore.getState().uri).toBe(
      "file:///doc/avatar/profile.jpg",
    );
  });

  it("swallows a delete failure rather than throwing out of removePhoto", async () => {
    (SecureStore.deleteItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    useAvatarStore.setState({ uri: "file:///doc/avatar/profile.jpg" });
    expect(() => useAvatarStore.getState().removePhoto()).not.toThrow();
    await Promise.resolve();
    expect(useAvatarStore.getState().uri).toBeNull();
  });
});

describe("useAvatarStore persistence round-trip — web (localStorage)", () => {
  beforeEach(() => setPlatform("web"));

  it("round-trips a set photo through localStorage", async () => {
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    await Promise.resolve();

    resetStoreState();
    await useAvatarStore.getState().hydrate();

    expect(useAvatarStore.getState().uri).toBe(
      "file:///doc/avatar/profile.jpg",
    );
  });

  it("never touches expo-secure-store on web", async () => {
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    await Promise.resolve();
    await useAvatarStore.getState().hydrate();
    expect(SecureStore.setItemAsync).not.toHaveBeenCalled();
    expect(SecureStore.getItemAsync).not.toHaveBeenCalled();
  });

  it("removePhoto clears localStorage so a later hydrate reads null", async () => {
    useAvatarStore.getState().setPhoto("file:///doc/avatar/profile.jpg");
    await Promise.resolve();
    useAvatarStore.getState().removePhoto();
    await Promise.resolve();

    resetStoreState();
    await useAvatarStore.getState().hydrate();
    expect(useAvatarStore.getState().uri).toBeNull();
  });
});
