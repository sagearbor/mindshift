import React from "react";
import renderer, { act } from "react-test-renderer";
import EmpathySlider, {
  getEmpathyLabel,
} from "../src/components/EmpathySlider";

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


describe("EmpathySlider", () => {
  it("renders correctly at default value", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<EmpathySlider value={50} onValueChange={jest.fn()} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly at minimum", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<EmpathySlider value={0} onValueChange={jest.fn()} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly at maximum", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<EmpathySlider value={100} onValueChange={jest.fn()} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  describe("getEmpathyLabel", () => {
    it("returns Assertive for 0-20", () => {
      expect(getEmpathyLabel(0)).toBe("Assertive");
      expect(getEmpathyLabel(20)).toBe("Assertive");
    });

    it("returns Balanced for 21-50", () => {
      expect(getEmpathyLabel(21)).toBe("Balanced");
      expect(getEmpathyLabel(50)).toBe("Balanced");
    });

    it("returns Empathetic for 51-80", () => {
      expect(getEmpathyLabel(51)).toBe("Empathetic");
      expect(getEmpathyLabel(80)).toBe("Empathetic");
    });

    it("returns Full Empathy for 81-100", () => {
      expect(getEmpathyLabel(81)).toBe("Full Empathy");
      expect(getEmpathyLabel(100)).toBe("Full Empathy");
    });
  });
});
