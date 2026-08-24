import React from "react";
import renderer, { act } from "react-test-renderer";
import LiveModePicker, { LIVE_MODE_OPTIONS } from "../src/components/LiveModePicker";

describe("LiveModePicker", () => {
  it("offers the three modes with a one-line hint for the selected one", () => {
    const onChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(<LiveModePicker value="speaker" onChange={onChange} />);
    });
    expect(LIVE_MODE_OPTIONS.map((o) => o.mode)).toEqual(["earpiece", "speaker", "therapist"]);
    const hint = root!.root.findByProps({ testID: "session-mode-hint" });
    expect(JSON.stringify(hint.props.children)).toContain("Both voices on one mic");
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
      root = renderer.create(<LiveModePicker value="earpiece" onChange={jest.fn()} disabled />);
    });
    for (const o of LIVE_MODE_OPTIONS) {
      expect(root!.root.findByProps({ testID: `session-mode-${o.mode}` }).props.disabled).toBe(true);
    }
  });
});
