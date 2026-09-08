/**
 * src/store/devModeStore.ts — the persisted per-account "Developer mode"
 * flag behind the Settings switch. OFF by default (clean tester surface);
 * survives relaunch via SecureStore (native) / localStorage (web).
 */
import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_DEV_MODE,
  devModeKey,
  loadDevMode,
  saveDevMode,
  useDevModeStore,
} from "../src/store/devModeStore";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
  useDevModeStore.setState({ devMode: DEFAULT_DEV_MODE });
});

describe("devModeKey", () => {
  it("is per account and safe for storage backends", () => {
    expect(devModeKey("u1")).toBe("mindshift.developerMode.v1.u1");
    expect(devModeKey(null)).toBe("mindshift.developerMode.v1.anon");
    expect(devModeKey("a b/c@d")).toBe("mindshift.developerMode.v1.a_b_c_d");
  });
});

describe("loadDevMode / saveDevMode", () => {
  it("defaults OFF: nothing stored, garbage stored, or storage throwing", async () => {
    expect(await loadDevMode("u1")).toBe(false);
    getItem.mockResolvedValue("true"); // legacy/foreign value, not ours
    expect(await loadDevMode("u1")).toBe(false);
    getItem.mockRejectedValue(new Error("keystore locked"));
    expect(await loadDevMode("u1")).toBe(false);
  });

  it("round-trips on and off", async () => {
    await saveDevMode("u1", true);
    expect(setItem).toHaveBeenCalledWith(devModeKey("u1"), "on");
    getItem.mockResolvedValue("on");
    expect(await loadDevMode("u1")).toBe(true);
    getItem.mockResolvedValue("off");
    expect(await loadDevMode("u1")).toBe(false);
  });
});

describe("useDevModeStore", () => {
  it("hydrate loads the stored flag; setDevMode flips now and persists", async () => {
    getItem.mockResolvedValue("on");
    await useDevModeStore.getState().hydrate("u1");
    expect(useDevModeStore.getState().devMode).toBe(true);

    useDevModeStore.getState().setDevMode("u1", false);
    expect(useDevModeStore.getState().devMode).toBe(false);
    expect(setItem).toHaveBeenCalledWith(devModeKey("u1"), "off");
  });
});
