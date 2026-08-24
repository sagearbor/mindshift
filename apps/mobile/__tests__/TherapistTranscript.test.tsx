import React from "react";
import renderer, { act } from "react-test-renderer";
import TherapistTranscript, { columnOf } from "../src/components/TherapistTranscript";

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
      root = renderer.create(<TherapistTranscript entries={[]} />);
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
