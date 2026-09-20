import React from "react";
import renderer, { act } from "react-test-renderer";
import InterjectSlider, {
  getInterjectLabel,
} from "../src/components/InterjectSlider";

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


describe("InterjectSlider", () => {
  it("renders correctly at default value", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(
        <InterjectSlider value={0} onValueChange={jest.fn()} />,
      ));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly at maximum", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(
        <InterjectSlider value={100} onValueChange={jest.fn()} />,
      ));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("fires onValueChange when the slider moves", () => {
    const onValueChange = jest.fn();
    let root: renderer.ReactTestRenderer;
    act(() => {
      root = track(renderer.create(
        <InterjectSlider value={0} onValueChange={onValueChange} />,
      ));
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
