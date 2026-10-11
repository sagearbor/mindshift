import * as SecureStore from "expo-secure-store";
import {
  DEFAULT_EARLY_TURN_LOCAL,
  EARLY_TURN_LOCAL_KEY,
  SPEED_TEST_AUTO_SEND,
  loadEarlyTurnLocal,
  saveEarlyTurnLocal,
  useSpeedTestStore,
} from "../src/live/speedTestPrefs";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
  useSpeedTestStore.setState({ earlyTurnLocal: DEFAULT_EARLY_TURN_LOCAL, hydrated: false });
});

describe("speed test switch (send turn to cloud immediately)", () => {
  it("is ON by default in the speed-test build, and every session's timing is auto-sent", () => {
    expect(DEFAULT_EARLY_TURN_LOCAL).toBe(true);
    expect(SPEED_TEST_AUTO_SEND).toBe(true);
    expect(useSpeedTestStore.getState().earlyTurnLocal).toBe(true);
  });

  it("is per device (one key, no account in it)", async () => {
    await loadEarlyTurnLocal();
    expect(getItem).toHaveBeenLastCalledWith(EARLY_TURN_LOCAL_KEY);
    expect(EARLY_TURN_LOCAL_KEY).toBe("mindshift.earlyTurnLocal.device.v1");
  });

  it("loads the remembered choice and fails open to ON", async () => {
    getItem.mockResolvedValueOnce("off");
    expect(await loadEarlyTurnLocal()).toBe(false);
    getItem.mockResolvedValueOnce("on");
    expect(await loadEarlyTurnLocal()).toBe(true);
    getItem.mockResolvedValueOnce("garbage");
    expect(await loadEarlyTurnLocal()).toBe(true);
    getItem.mockRejectedValueOnce(new Error("keychain locked"));
    expect(await loadEarlyTurnLocal()).toBe(true);
  });

  it("saves and swallows storage errors", async () => {
    await saveEarlyTurnLocal(false);
    expect(setItem).toHaveBeenCalledWith(EARLY_TURN_LOCAL_KEY, "off");
    setItem.mockRejectedValueOnce(new Error("full"));
    await expect(saveEarlyTurnLocal(true)).resolves.toBeUndefined();
  });

  it("store: hydrate reads the stored OFF; a flip persists", async () => {
    getItem.mockResolvedValueOnce("off");
    await useSpeedTestStore.getState().hydrate();
    expect(useSpeedTestStore.getState().earlyTurnLocal).toBe(false);
    useSpeedTestStore.getState().setEarlyTurnLocal(true);
    expect(useSpeedTestStore.getState().earlyTurnLocal).toBe(true);
    expect(setItem).toHaveBeenCalledWith(EARLY_TURN_LOCAL_KEY, "on");
  });

  it("store: a flip made while hydrate is in flight wins", async () => {
    let resolve: (v: string | null) => void = () => {};
    getItem.mockReturnValueOnce(new Promise((r) => (resolve = r)));
    const p = useSpeedTestStore.getState().hydrate();
    useSpeedTestStore.getState().setEarlyTurnLocal(false);
    resolve("on");
    await p;
    expect(useSpeedTestStore.getState().earlyTurnLocal).toBe(false);
  });
});
