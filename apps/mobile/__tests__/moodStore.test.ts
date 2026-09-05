/**
 * src/store/moodStore.ts — the outcome-engine mood check (CANDOR's single
 * 1-9 item) held for the current session, persisted per session id via
 * SecureStore (native) / localStorage (web), fail-open.
 */
import * as SecureStore from "expo-secure-store";
import { loadMoodPair, moodKey, saveMoodPair, useMoodStore } from "../src/store/moodStore";

const getItem = SecureStore.getItemAsync as jest.Mock;
const setItem = SecureStore.setItemAsync as jest.Mock;

beforeEach(() => {
  getItem.mockReset().mockResolvedValue(null);
  setItem.mockReset().mockResolvedValue(undefined);
  useMoodStore.setState({ before: null, after: null });
});

describe("moodKey", () => {
  it("is per session and safe for storage backends", () => {
    expect(moodKey("live-123")).toBe("mindshift.moodCheck.v1.live-123");
    expect(moodKey("a b/c@d")).toBe("mindshift.moodCheck.v1.a_b_c_d");
  });
});

describe("loadMoodPair / saveMoodPair", () => {
  it("null on nothing stored, garbage, or storage throwing", async () => {
    expect(await loadMoodPair("s1")).toBeNull();
    getItem.mockResolvedValue("not json");
    expect(await loadMoodPair("s1")).toBeNull();
    getItem.mockResolvedValue(JSON.stringify({ before: null, after: null }));
    expect(await loadMoodPair("s1")).toBeNull();
    getItem.mockResolvedValue(JSON.stringify({ before: 11, after: -1 })); // out of 1-9 range
    expect(await loadMoodPair("s1")).toBeNull();
    getItem.mockRejectedValue(new Error("keystore locked"));
    expect(await loadMoodPair("s1")).toBeNull();
  });

  it("round-trips a completed pair", async () => {
    await saveMoodPair("s1", { before: 4, after: 7 });
    expect(setItem).toHaveBeenCalledWith(moodKey("s1"), JSON.stringify({ before: 4, after: 7 }));
    getItem.mockResolvedValue(JSON.stringify({ before: 4, after: 7 }));
    expect(await loadMoodPair("s1")).toEqual({ before: 4, after: 7 });
  });

  it("a save failure never throws", async () => {
    setItem.mockRejectedValue(new Error("keystore locked"));
    await expect(saveMoodPair("s1", { before: 4, after: null })).resolves.toBeUndefined();
  });
});

describe("useMoodStore", () => {
  it("setBefore/setAfter update the current values", () => {
    useMoodStore.getState().setBefore(null, 3);
    expect(useMoodStore.getState().before).toBe(3);
    useMoodStore.getState().setAfter(null, 8);
    expect(useMoodStore.getState().after).toBe(8);
  });

  it("setBefore with no session id updates in-memory only — nothing persisted", () => {
    useMoodStore.getState().setBefore(null, 5);
    expect(setItem).not.toHaveBeenCalled();
  });

  it("setAfter with a session id persists the completed pair (before + after together)", () => {
    useMoodStore.getState().setBefore(null, 4);
    useMoodStore.getState().setAfter("ep-1", 8);
    expect(setItem).toHaveBeenCalledWith(
      moodKey("ep-1"), JSON.stringify({ before: 4, after: 8 }),
    );
  });

  it("reset clears both", () => {
    useMoodStore.getState().setBefore(null, 3);
    useMoodStore.getState().setAfter(null, 6);
    useMoodStore.getState().reset();
    expect(useMoodStore.getState().before).toBeNull();
    expect(useMoodStore.getState().after).toBeNull();
  });

  it("skip (null) is a valid answer and persists as null", () => {
    useMoodStore.getState().setBefore("s1", null);
    expect(setItem).toHaveBeenCalledWith(moodKey("s1"), JSON.stringify({ before: null, after: null }));
  });
});
