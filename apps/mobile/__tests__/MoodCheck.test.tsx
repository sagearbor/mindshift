import React from "react";
import renderer, { act } from "react-test-renderer";
import MoodCheck from "../src/components/MoodCheck";

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


describe("MoodCheck", () => {
  it("renders all 9 options plus a skip affordance, with the phase's title", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<MoodCheck phase="before" value={null} onChange={jest.fn()} />));
    });
    for (let n = 1; n <= 9; n += 1) {
      expect(root!.root.findByProps({ testID: `mood-check-option-${n}` })).toBeTruthy();
    }
    expect(root!.root.findByProps({ testID: "mood-check-skip" })).toBeTruthy();
    const json = JSON.stringify(root!.toJSON());
    expect(json).toContain("How are you feeling right now?");
  });

  it("the after phase uses its own title", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<MoodCheck phase="after" value={null} onChange={jest.fn()} />));
    });
    const json = JSON.stringify(root!.toJSON());
    expect(json).toContain("How are you feeling now?");
  });

  it("tapping a number reports it and marks it selected", () => {
    const onChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<MoodCheck phase="before" value={null} onChange={onChange} />));
    });
    act(() => {
      root!.root.findByProps({ testID: "mood-check-option-7" }).props.onPress();
    });
    expect(onChange).toHaveBeenCalledWith(7);

    act(() => {
      root!.update(<MoodCheck phase="before" value={7} onChange={onChange} />);
    });
    expect(
      root!.root.findByProps({ testID: "mood-check-option-7" }).props.accessibilityState,
    ).toEqual({ selected: true });
    expect(
      root!.root.findByProps({ testID: "mood-check-option-3" }).props.accessibilityState,
    ).toEqual({ selected: false });
  });

  it("tapping skip reports null", () => {
    const onChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<MoodCheck phase="after" value={5} onChange={onChange} />));
    });
    act(() => {
      root!.root.findByProps({ testID: "mood-check-skip" }).props.onPress();
    });
    expect(onChange).toHaveBeenCalledWith(null);
  });
});
