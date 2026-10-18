/**
 * src/store/devModeStore.ts — the persisted per-account "Developer mode"
 * flag behind the Settings switch. OFF by default (clean tester surface);
 * survives relaunch via SecureStore (native) / localStorage (web).
 */
import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_DEBUG_CAPTURE,
  DEFAULT_DEV_MODE,
  debugCaptureKey,
  devModeKey,
  isDebugCaptureActive,
  loadDebugCapture,
  loadDevMode,
  saveDebugCapture,
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

// "Debug session capture" (2026-10-18): the owner-only switch under
// Developer mode that keeps local session payload copies and asks the server
// to keep its transcript. OFF by default; only ACTIVE while Developer mode
// is on too (turning Developer mode off can't leave a hidden capture on).
describe("debug session capture", () => {
  beforeEach(() => {
    useDevModeStore.setState({ devMode: false, debugCapture: DEFAULT_DEBUG_CAPTURE });
  });

  it("is OFF by default, per account, and fails open to OFF", async () => {
    expect(DEFAULT_DEBUG_CAPTURE).toBe(false);
    expect(useDevModeStore.getState().debugCapture).toBe(false);
    expect(debugCaptureKey("u1")).toBe("mindshift.debugSessionCapture.v1.u1");
    expect(debugCaptureKey(null)).toBe("mindshift.debugSessionCapture.v1.anon");
    expect(await loadDebugCapture("u1")).toBe(false);
    getItem.mockRejectedValue(new Error("keystore locked"));
    expect(await loadDebugCapture("u1")).toBe(false);
    expect(isDebugCaptureActive()).toBe(false);
  });

  it("round-trips and persists under its own key", async () => {
    await saveDebugCapture("u1", true);
    expect(setItem).toHaveBeenCalledWith(debugCaptureKey("u1"), "on");
    getItem.mockResolvedValue("on");
    expect(await loadDebugCapture("u1")).toBe(true);
  });

  it("hydrate loads both flags; setDebugCapture flips now and persists", async () => {
    getItem.mockImplementation(async (key: string) => (key === debugCaptureKey("u1") ? "on" : "off"));
    await useDevModeStore.getState().hydrate("u1");
    expect(useDevModeStore.getState()).toMatchObject({ devMode: false, debugCapture: true });

    useDevModeStore.getState().setDebugCapture("u1", false);
    expect(useDevModeStore.getState().debugCapture).toBe(false);
    expect(setItem).toHaveBeenCalledWith(debugCaptureKey("u1"), "off");
  });

  it("is active only while Developer mode is on as well", () => {
    useDevModeStore.setState({ devMode: false, debugCapture: true });
    expect(isDebugCaptureActive()).toBe(false);
    useDevModeStore.setState({ devMode: true, debugCapture: false });
    expect(isDebugCaptureActive()).toBe(false);
    useDevModeStore.setState({ devMode: true, debugCapture: true });
    expect(isDebugCaptureActive()).toBe(true);
  });
});
