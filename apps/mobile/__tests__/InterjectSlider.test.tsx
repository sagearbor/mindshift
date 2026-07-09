import React from "react";
import renderer, { act } from "react-test-renderer";
import InterjectSlider, {
  getInterjectLabel,
} from "../src/components/InterjectSlider";

describe("InterjectSlider", () => {
  it("renders correctly at default value", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(
        <InterjectSlider value={0} onValueChange={jest.fn()} />,
      );
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly at maximum", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(
        <InterjectSlider value={100} onValueChange={jest.fn()} />,
      );
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("fires onValueChange when the slider moves", () => {
    const onValueChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = renderer.create(
        <InterjectSlider value={0} onValueChange={onValueChange} />,
      );
    });
    act(() => {
      root!.root.findByProps({ testID: "interject-slider" }).props.onValueChange(70);
    });
    expect(onValueChange).toHaveBeenCalledWith(70);
  });

  describe("getInterjectLabel", () => {
    it("returns Every turn for 0-20", () => {
      expect(getInterjectLabel(0)).toBe("Every turn");
      expect(getInterjectLabel(20)).toBe("Every turn");
    });

    it("returns Most turns for 21-50", () => {
      expect(getInterjectLabel(21)).toBe("Most turns");
      expect(getInterjectLabel(50)).toBe("Most turns");
    });

    it("returns Key moments for 51-80", () => {
      expect(getInterjectLabel(51)).toBe("Key moments");
      expect(getInterjectLabel(80)).toBe("Key moments");
    });

    it("returns Critical only for 81-100", () => {
      expect(getInterjectLabel(81)).toBe("Critical only");
      expect(getInterjectLabel(100)).toBe("Critical only");
    });
  });
});
