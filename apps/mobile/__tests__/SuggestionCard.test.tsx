import React from "react";
import renderer, { act } from "react-test-renderer";
import SuggestionCard, { getToneColor } from "../src/components/SuggestionCard";

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


describe("SuggestionCard", () => {
  it("renders correctly with empathetic tone", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<SuggestionCard
          text="I hear what you're saying and that sounds really difficult."
          tone="empathetic"
        />,));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly with assertive tone", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<SuggestionCard
          text="I understand, but I need to express my perspective too."
          tone="assertive"
        />,));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly with balanced tone", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<SuggestionCard
          text="I see your point. Let's find a middle ground."
          tone="balanced"
        />,));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  describe("getToneColor", () => {
    it("returns correct color for known tones", () => {
      expect(getToneColor("empathetic")).toBe("#10B981");
      expect(getToneColor("assertive")).toBe("#EF4444");
      expect(getToneColor("balanced")).toBe("#F59E0B");
    });

    it("returns neutral color for unknown tones", () => {
      expect(getToneColor("unknown")).toBe("#6B7280");
    });

    it("is case insensitive", () => {
      expect(getToneColor("Empathetic")).toBe("#10B981");
    });
  });
});
