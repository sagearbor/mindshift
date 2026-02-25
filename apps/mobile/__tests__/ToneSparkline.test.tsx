import React from "react";
import renderer, { act } from "react-test-renderer";
import ToneSparkline from "../src/components/ToneSparkline";

describe("ToneSparkline", () => {
  it("renders with scores", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(
          <ToneSparkline scores={[30, 55, 70, 45, 80]} width={120} height={40} />
        )
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders empty state when no scores", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<ToneSparkline scores={[]} />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders single score", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(<ToneSparkline scores={[65]} width={100} height={30} color="#10B981" />)
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders with custom color", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer
        .create(
          <ToneSparkline scores={[20, 40, 60]} color="#EF4444" />
        )
        .toJSON();
    });
    expect(tree).toMatchSnapshot();
  });
});
