import React from "react";
import renderer, { act } from "react-test-renderer";
import TherapistTranscript, { columnOf } from "../src/components/TherapistTranscript";

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


const entries = [
  { speaker: "Sage", text: "I felt ignored.", timestamp: 1 },
  { speaker: "Mom", text: "I didn't mean to.", timestamp: 2 },
  { speaker: "Sage", text: "I know.", timestamp: 3 },
  { speaker: "Dad", text: "Can I say something?", timestamp: 4 },
];

describe("TherapistTranscript", () => {
  it("columnOf: first voice left, second right, a third joins the right", () => {
    const c = columnOf(entries);
    expect(c.left).toBe("Sage");
    expect(c.right).toBe("Mom");
    expect(c.side("Sage")).toBe("left");
    expect(c.side("Mom")).toBe("right");
    expect(c.side("Dad")).toBe("right");
    expect(columnOf([]).left).toBeNull();
  });

  it("renders the empty state, then two labelled columns with bubbles per side", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(<TherapistTranscript entries={[]} />));
    });
    expect(root!.root.findByProps({ testID: "therapist-transcript-empty" })).toBeTruthy();
    act(() => {
      root!.update(<TherapistTranscript entries={entries} />);
    });
    expect(JSON.stringify(root!.root.findByProps({ testID: "therapist-column-left" }).props.children)).toContain("Sage");
    expect(JSON.stringify(root!.root.findByProps({ testID: "therapist-column-right" }).props.children)).toContain("Mom");
    expect(root!.root.findByProps({ testID: "therapist-turn-0-left" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "therapist-turn-1-right" })).toBeTruthy();
    expect(root!.root.findByProps({ testID: "therapist-turn-3-right" })).toBeTruthy();
    expect(JSON.stringify(root!.toJSON())).toContain("Can I say something?");
  });
});
