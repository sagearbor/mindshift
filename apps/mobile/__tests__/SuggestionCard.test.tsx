import React from "react";
import renderer, { act } from "react-test-renderer";
import SuggestionCard, { getToneColor } from "../src/components/SuggestionCard";

describe("SuggestionCard", () => {
  it("renders correctly with empathetic tone", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SuggestionCard
          text="I hear what you're saying and that sounds really difficult."
          tone="empathetic"
        />,);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly with assertive tone", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SuggestionCard
          text="I understand, but I need to express my perspective too."
          tone="assertive"
        />,);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly with balanced tone", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<SuggestionCard
          text="I see your point. Let's find a middle ground."
          tone="balanced"
        />,);
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
