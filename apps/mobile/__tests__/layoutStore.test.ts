import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  useLayoutStore,
  sanitizeSlots,
  DEFAULT_TAB_SLOTS,
  DEFAULT_HOME_BOXES,
  TAB_SLOT_CAP,
  HOME_BOX_CAP,
} from "../src/store/layoutStore";
import type { DestId } from "../src/nav/destinations";

const originalOS = Platform.OS;

function setPlatform(os: string) {
  Object.defineProperty(Platform, "OS", { value: os, configurable: true });
}

/** Matches onboardingStorage.test.ts's polyfill — this repo's default jest
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
  useLayoutStore.setState({
    tabSlots: [...DEFAULT_TAB_SLOTS],
    homeBoxes: [...DEFAULT_HOME_BOXES],
    hydrated: false,
  });
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

describe("sanitizeSlots", () => {
  it("passes through a valid, unique, primary-eligible list", () => {
    expect(sanitizeSlots(["coach", "analyze"], TAB_SLOT_CAP)).toEqual([
      "coach",
      "analyze",
    ]);
  });

  it("drops duplicate ids, keeping the first occurrence", () => {
    expect(sanitizeSlots(["coach", "analyze", "coach"], TAB_SLOT_CAP)).toEqual(
      ["coach", "analyze"],
    );
  });

  it("drops ids that aren't primary-eligible (settings, a catalog-only id)", () => {
    expect(sanitizeSlots(["coach", "settings"], TAB_SLOT_CAP)).toEqual([
      "coach",
    ]);
  });

  it("drops unknown/garbage ids", () => {
    expect(
      sanitizeSlots(["coach", "not-a-real-id", 42, null], TAB_SLOT_CAP),
    ).toEqual(["coach"]);
  });

  it("truncates to the cap", () => {
    expect(
      sanitizeSlots(
        ["coach", "analyze", "recordings", "growth", "coach"],
        2,
      ),
    ).toEqual(["coach", "analyze"]);
  });

  it("returns an empty list for non-array input", () => {
    expect(sanitizeSlots(null, TAB_SLOT_CAP)).toEqual([]);
    expect(sanitizeSlots(undefined, TAB_SLOT_CAP)).toEqual([]);
    expect(sanitizeSlots("coach", TAB_SLOT_CAP)).toEqual([]);
    expect(sanitizeSlots({}, TAB_SLOT_CAP)).toEqual([]);
  });

  it("allows an empty list (0 tabs / 0 boxes is valid)", () => {
    expect(sanitizeSlots([], TAB_SLOT_CAP)).toEqual([]);
  });
});

describe("useLayoutStore defaults", () => {
  it("starts with the documented default tab slots and home boxes", () => {
    const state = useLayoutStore.getState();
    expect(state.tabSlots).toEqual(["coach", "analyze", "growth"]);
    expect(state.homeBoxes).toEqual(["recordings", "growth"]);
    expect(state.hydrated).toBe(false);
  });
});

describe("useLayoutStore.setTabSlots", () => {
  it("validates: dedups, drops ineligible ids, enforces the tab cap", () => {
    // The setter's runtime validation is defense-in-depth for callers that
    // aren't statically typed (e.g. a value round-tripped through JSON) —
    // cast past the DestId[] signature to exercise it here.
    useLayoutStore.getState().setTabSlots([
      "coach",
      "coach",
      "settings",
      "analyze",
      "recordings",
      "growth",
      "not-a-real-id",
    ] as unknown as DestId[]);
    // cap is 5; "settings" and the unknown id are dropped before the cap
    // truncation, "coach" deduped.
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "recordings",
      "growth",
    ]);
  });

  it("allows clearing the tab bar to zero slots", () => {
    useLayoutStore.getState().setTabSlots([]);
    expect(useLayoutStore.getState().tabSlots).toEqual([]);
  });
});

describe("useLayoutStore.setHomeBoxes", () => {
  it("validates against the box cap of 4 and eligibility rules", () => {
    useLayoutStore
      .getState()
      .setHomeBoxes(["coach", "analyze", "recordings", "growth", "tutorial"]);
    expect(useLayoutStore.getState().homeBoxes).toEqual([
      "coach",
      "analyze",
      "recordings",
      "growth",
    ]);
  });
});

describe("useLayoutStore.resetToDefaults", () => {
  it("restores both lists to the shipped defaults", () => {
    useLayoutStore.getState().setTabSlots(["growth"]);
    useLayoutStore.getState().setHomeBoxes([]);
    useLayoutStore.getState().resetToDefaults();
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
    expect(useLayoutStore.getState().homeBoxes).toEqual([
      "recordings",
      "growth",
    ]);
  });
});

describe("useLayoutStore persistence round-trip — native (expo-secure-store)", () => {
  beforeEach(() => setPlatform("ios"));

  it("hydrate() leaves defaults in place when nothing was ever saved", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(null);
    await useLayoutStore.getState().hydrate();
    const state = useLayoutStore.getState();
    expect(state.tabSlots).toEqual(["coach", "analyze", "growth"]);
    expect(state.homeBoxes).toEqual(["recordings", "growth"]);
    expect(state.hydrated).toBe(true);
  });

  it("round-trips a custom layout through setTabSlots -> persist -> hydrate", async () => {
    let stored: string | null = null;
    (SecureStore.setItemAsync as jest.Mock).mockImplementation(
      async (_key: string, value: string) => {
        stored = value;
      },
    );
    (SecureStore.getItemAsync as jest.Mock).mockImplementation(
      async () => stored,
    );

    useLayoutStore.getState().setTabSlots(["growth", "coach"]);
    useLayoutStore.getState().setHomeBoxes(["recordings"]);
    // setters persist fire-and-forget; give the microtask queue a turn.
    await Promise.resolve();
    await Promise.resolve();

    // Simulate a fresh app launch reading the persisted value back.
    resetStoreState();
    await useLayoutStore.getState().hydrate();

    expect(useLayoutStore.getState().tabSlots).toEqual(["growth", "coach"]);
    expect(useLayoutStore.getState().homeBoxes).toEqual(["recordings"]);
  });

  it("drops unknown ids found in otherwise-valid persisted data", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(
      JSON.stringify({
        tabSlots: ["coach", "a-removed-destination", "growth"],
        homeBoxes: ["recordings"],
      }),
    );
    await useLayoutStore.getState().hydrate();
    expect(useLayoutStore.getState().tabSlots).toEqual(["coach", "growth"]);
    expect(useLayoutStore.getState().homeBoxes).toEqual(["recordings"]);
  });

  it("fails open to defaults when the persisted payload is corrupt JSON", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(
      "{not valid json",
    );
    await useLayoutStore.getState().hydrate();
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
    expect(useLayoutStore.getState().homeBoxes).toEqual([
      "recordings",
      "growth",
    ]);
    expect(useLayoutStore.getState().hydrated).toBe(true);
  });

  it("fails open to defaults when the persisted payload isn't the expected shape", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockResolvedValueOnce(
      JSON.stringify("just a string, not an object"),
    );
    await useLayoutStore.getState().hydrate();
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
  });

  it("never crashes when the storage read throws", async () => {
    (SecureStore.getItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    await expect(useLayoutStore.getState().hydrate()).resolves.toBeUndefined();
    expect(useLayoutStore.getState().tabSlots).toEqual([
      "coach",
      "analyze",
      "growth",
    ]);
  });

  it("swallows a write failure rather than throwing out of a setter", async () => {
    (SecureStore.setItemAsync as jest.Mock).mockRejectedValueOnce(
      new Error("keychain unavailable"),
    );
    expect(() =>
      useLayoutStore.getState().setTabSlots(["growth"]),
    ).not.toThrow();
    await Promise.resolve();
    await Promise.resolve();
    // The in-memory state still updated even though the persist failed.
    expect(useLayoutStore.getState().tabSlots).toEqual(["growth"]);
  });
});

describe("useLayoutStore persistence round-trip — web (localStorage)", () => {
  beforeEach(() => setPlatform("web"));

  it("round-trips a custom layout through localStorage", async () => {
    useLayoutStore.getState().setTabSlots(["analyze"]);
    useLayoutStore.getState().setHomeBoxes(["growth", "recordings"]);
    await Promise.resolve();

    resetStoreState();
    await useLayoutStore.getState().hydrate();

    expect(useLayoutStore.getState().tabSlots).toEqual(["analyze"]);
    expect(useLayoutStore.getState().homeBoxes).toEqual([
      "growth",
      "recordings",
    ]);
  });

  it("never touches expo-secure-store on web", async () => {
    useLayoutStore.getState().setTabSlots(["analyze"]);
    await Promise.resolve();
    await useLayoutStore.getState().hydrate();
    expect(SecureStore.setItemAsync).not.toHaveBeenCalled();
    expect(SecureStore.getItemAsync).not.toHaveBeenCalled();
  });
});
