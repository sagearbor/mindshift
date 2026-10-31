import React from "react";
import renderer, { act } from "react-test-renderer";
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

import {
  applyRoomFrame,
  displaySourceTitle,
  IDLE_ROOM_STATE,
  MAX_ROOM_CHECKS,
  MAX_ROOM_HEADSUPS,
  parseRoomFrame,
  type RoomViewState,
} from "../src/live/roomMode";
import { headsUpKey, loadHeadsUpCues, saveHeadsUpCues } from "../src/live/modePrefs";
import RoomPanel from "../src/components/RoomPanel";

const headsUp = (id: string) => ({
  type: "room_headsup",
  id,
  headline: "You mentioned profit",
  title: "Profit plan",
  snippet: "Our profit target for next year is 18% net margin.",
  passage: "Our profit target for next year is 18% net margin. Pricing changes ship in March.",
  source_item_id: "plan2",
  source_title: "FY27 business plan",
  t: 5,
});
const check = (id: string) => ({
  type: "room_check",
  id,
  statement: "we can use dangerously skip permissions",
  title: "Permissions",
  passage: "Dangerous permission bypass should be asked for, not skipped.",
  reason: "The note says to ask first.",
  source_item_id: "plan1",
  source_title: "Claude working rules",
  t: 9,
});

function fold(frames: unknown[]): RoomViewState {
  return frames.reduce<RoomViewState>((s, f) => {
    const p = parseRoomFrame(f);
    return p ? applyRoomFrame(s, p) : s;
  }, IDLE_ROOM_STATE);
}

describe("heads-up frames", () => {
  it("parses and folds, deduping and capping", () => {
    const s = fold([headsUp("a"), headsUp("a"), headsUp("b"), headsUp("c"), headsUp("d"), check("x"), check("y"), check("z")]);
    expect(s.headsUps.map((h) => h.id)).toEqual(["d", "c", "b"].slice(0, MAX_ROOM_HEADSUPS));
    expect(s.checks).toHaveLength(MAX_ROOM_CHECKS);
    expect(s.headsUps[0].passage).toContain("Pricing changes");
  });

  it("rejects malformed frames", () => {
    expect(parseRoomFrame({ type: "room_headsup", id: "", snippet: "x" })).toBeNull();
    expect(parseRoomFrame({ type: "room_check", id: "a", passage: "" })).toBeNull();
  });

  it("never shows a uuid as the source", () => {
    expect(displaySourceTitle("3f2b8c1e-1111-4222-8333-444455556666")).toBe("your library");
    expect(displaySourceTitle("FY27 plan")).toBe("FY27 plan");
  });
});

describe("heads-up pref", () => {
  const getItem = SecureStore.getItemAsync as jest.Mock;
  const setItem = SecureStore.setItemAsync as jest.Mock;
  beforeEach(() => {
    getItem.mockReset().mockResolvedValue(null);
    setItem.mockReset().mockResolvedValue(undefined);
  });
  it("defaults OFF and persists per account", async () => {
    expect(Platform.OS).not.toBe("web");
    expect(await loadHeadsUpCues("sage")).toBe(false);
    getItem.mockResolvedValueOnce("1");
    expect(await loadHeadsUpCues("sage")).toBe(true);
    expect(getItem).toHaveBeenLastCalledWith(headsUpKey("sage"));
    await saveHeadsUpCues("sage", true);
    expect(setItem).toHaveBeenCalledWith(headsUpKey("sage"), "1");
    getItem.mockRejectedValueOnce(new Error("locked"));
    expect(await loadHeadsUpCues("sage")).toBe(false);
  });
});

describe("RoomPanel heads-up", () => {
  const state = fold([headsUp("a"), check("x")]);
  const render = (on: boolean, onChange = jest.fn()) => {
    let r!: renderer.ReactTestRenderer;
    act(() => {
      r = renderer.create(
        <RoomPanel
          state={state}
          sessionActive
          speakAloud={false}
          onSpeakAloudChange={jest.fn()}
          librarySelected={1}
          headsUpCues={on}
          onHeadsUpCuesChange={onChange}
        />,
      );
    });
    return r;
  };

  it("shows nothing from heads-up frames while the toggle is off", () => {
    const r = render(false);
    expect(r.root.findAllByProps({ testID: "room-headsups" })).toHaveLength(0);
    expect(r.root.findAllByProps({ testID: "room-checks" })).toHaveLength(0);
  });

  it("toggle calls back", () => {
    const onChange = jest.fn();
    const r = render(false, onChange);
    act(() => r.root.findByProps({ testID: "room-headsup-switch" }).props.onValueChange(true));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("renders titled, sourced cards and expands to the full passage on tap", () => {
    const r = render(true);
    const card = r.root.findAll(
      (n) => n.props.testID === "room-headsup-a" && typeof n.props.onPress === "function",
    )[0];
    const text = () => JSON.stringify(r.toJSON());
    expect(text()).toContain("Profit plan");
    expect(text()).toContain("FY27 business plan");
    expect(text()).toContain("You mentioned profit");
    expect(text()).not.toContain("Pricing changes ship in March");
    act(() => card.props.onPress());
    expect(text()).toContain("Pricing changes ship in March");
    expect(text()).toContain("Check");
    expect(text()).toContain("asked for, not skipped");
  });
});
