import React from "react";
import renderer from "react-test-renderer";
import SuggestionCard, { getToneColor } from "../src/components/SuggestionCard";

describe("SuggestionCard", () => {
  it("renders correctly with empathetic tone", () => {
    const tree = renderer
      .create(
        <SuggestionCard
          text="I hear what you're saying and that sounds really difficult."
          tone="empathetic"
        />,
      )
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("renders correctly with assertive tone", () => {
    const tree = renderer
      .create(
        <SuggestionCard
          text="I understand, but I need to express my perspective too."
          tone="assertive"
        />,
      )
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("renders correctly with balanced tone", () => {
    const tree = renderer
      .create(
        <SuggestionCard
          text="I see your point. Let's find a middle ground."
          tone="balanced"
        />,
      )
      .toJSON();
    expect(tree).toMatchSnapshot();
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
