import React from "react";
import renderer, { act } from "react-test-renderer";
import MoodCheck from "../src/components/MoodCheck";

describe("MoodCheck", () => {
  it("renders all 9 options plus a skip affordance, with the phase's title", () => {
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<MoodCheck phase="before" value={null} onChange={jest.fn()} />);
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
      root = renderer.create(<MoodCheck phase="after" value={null} onChange={jest.fn()} />);
    });
    const json = JSON.stringify(root!.toJSON());
    expect(json).toContain("How are you feeling now?");
  });

  it("tapping a number reports it and marks it selected", () => {
    const onChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<MoodCheck phase="before" value={null} onChange={onChange} />);
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
      root = renderer.create(<MoodCheck phase="after" value={5} onChange={onChange} />);
    });
    act(() => {
      root!.root.findByProps({ testID: "mood-check-skip" }).props.onPress();
    });
    expect(onChange).toHaveBeenCalledWith(null);
  });
});
