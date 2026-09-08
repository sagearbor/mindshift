import React from "react";
import renderer, { act } from "react-test-renderer";
import LiveModePicker, { LIVE_MODE_OPTIONS } from "../src/components/LiveModePicker";

/**
 * Unmount every tree this file creates.
 *
 * react-test-renderer keeps a mounted tree scheduled, and a state update that
 * lands AFTER Jest tears the environment down throws "You are trying to
 * `import` a file after the Jest environment has been torn down" — which is
 * not a test failure but a WORKER CRASH, taking every unrelated suite sharing
 * that worker with it. That is why running the whole suite at once used to
 * fail a handful of random files that each passed on their own.
 */
const __trees: renderer.ReactTestRenderer[] = [];
function track<T extends renderer.ReactTestRenderer>(t: T): T {
  __trees.push(t);
  return t;
}
afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // A tree a test already unmounted, or one whose teardown throws, must
        // not fail the test that otherwise passed.
      }
    }
  });
});


describe("LiveModePicker", () => {
  it("offers the five modes with a one-line hint for the selected one", () => {
    const onChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveModePicker value="speaker" onChange={onChange} />));
    });
    expect(LIVE_MODE_OPTIONS.map((o) => o.mode)).toEqual(["earpiece", "speaker", "therapist", "call", "journal"]);
    // "speaker" is shown as "In person" (the wire value is unchanged).
    expect(LIVE_MODE_OPTIONS.find((o) => o.mode === "speaker")?.label).toBe("In person");
    expect(JSON.stringify(root!.toJSON())).not.toContain("Speaker-phone");
    const hint = root!.root.findByProps({ testID: "session-mode-hint" });
    expect(JSON.stringify(hint.props.children)).toContain("Both of you in the room");
    const selected = root!.root.findByProps({ testID: "session-mode-speaker" });
    expect(selected.props.accessibilityState.selected).toBe(true);
    act(() => {
      root!.root.findByProps({ testID: "session-mode-therapist" }).props.onPress();
    });
    expect(onChange).toHaveBeenCalledWith("therapist");
  });

  it("locks every chip while disabled", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<LiveModePicker value="earpiece" onChange={jest.fn()} disabled />));
    });
    for (const o of LIVE_MODE_OPTIONS) {
      expect(root!.root.findByProps({ testID: `session-mode-${o.mode}` }).props.disabled).toBe(true);
    }
  });
});
