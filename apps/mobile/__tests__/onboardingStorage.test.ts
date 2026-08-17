import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  getOnboardingSeen,
  setOnboardingSeen,
} from "../src/utils/onboardingStorage";

const originalOS = Platform.OS;

function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

/** This repo's jest preset has no DOM, so there's no real `localStorage` to
 *  exercise the web branch against — polyfill a minimal in-memory one on
 *  `globalThis`, matching what a real browser provides. */
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

beforeEach(() => installFakeLocalStorage());

afterEach(() => {
  setPlatform(originalOS);
  jest.clearAllMocks();
  uninstallFakeLocalStorage();
});

describe("onboardingStorage on native (expo-secure-store)", () => {
  beforeEach(() => setPlatform("ios"));

  it("reads false when nothing has been written yet", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(null);
    expect(await getOnboardingSeen()).toBe(false);
  });

  it("round-trips true through setOnboardingSeen/getOnboardingSeen", async () => {
    let stored: string | null = null;
    (SecureStore.setItemAsync as jest.Mock).mockImplementation(
      async (_key: string, value: string) => {
        stored = value;
      },
    );
    (SecureStore.getItemAsync as jest.Mock).mockImplementation(
      async () => stored,
    );

    expect(await getOnboardingSeen()).toBe(false);
    await setOnboardingSeen(true);
    expect(await getOnboardingSeen()).toBe(true);
  });

  it("fails open to false when the read throws", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    expect(await getOnboardingSeen()).toBe(false);
  });

  it("swallows a write failure rather than throwing", async () => {
    (SecureStore.setItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    await expect(setOnboardingSeen(true)).resolves.toBeUndefined();
  });
});

describe("onboardingStorage on web (window.localStorage)", () => {
  beforeEach(() => setPlatform("web"));

  it("reads false when nothing has been written yet", async () => {
    expect(await getOnboardingSeen()).toBe(false);
  });

  it("round-trips true through setOnboardingSeen/getOnboardingSeen", async () => {
    expect(await getOnboardingSeen()).toBe(false);
    await setOnboardingSeen(true);
    expect(await getOnboardingSeen()).toBe(true);
  });

  it("never touches expo-secure-store on web", async () => {
    await setOnboardingSeen(true);
    await getOnboardingSeen();
    expect(SecureStore.setItemAsync).not.toHaveBeenCalled();
    expect(SecureStore.getItemAsync).not.toHaveBeenCalled();
  });
});
