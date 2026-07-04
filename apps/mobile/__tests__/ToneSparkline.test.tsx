import React from "react";
import renderer, { act } from "react-test-renderer";
import ToneSparkline from "../src/components/ToneSparkline";

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
      component = renderer.create(<ToneSparkline scores={[]} />);
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
