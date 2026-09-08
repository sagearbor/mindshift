import React from "react";
import renderer, { act } from "react-test-renderer";
import ToneSparkline from "../src/components/ToneSparkline";

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


describe("ToneSparkline", () => {
  it("renders with scores", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer
        .create(
          <ToneSparkline scores={[30, 55, 70, 45, 80]} width={120} height={40} />
        )
        ;
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders empty state when no scores", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<ToneSparkline scores={[]} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders single score", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer
        .create(<ToneSparkline scores={[65]} width={100} height={30} color="#10B981" />)
        ;
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders with custom color", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer
        .create(
          <ToneSparkline scores={[20, 40, 60]} color="#EF4444" />
        )
        ;
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });
});
