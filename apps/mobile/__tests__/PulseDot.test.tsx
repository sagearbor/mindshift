import React from "react";
import renderer, { act } from "react-test-renderer";
import PulseDot from "../src/components/PulseDot";

function queryId(comp: renderer.ReactTestRenderer, id: string) {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

describe("PulseDot", () => {
  it("renders a static dot with no halo when animation is disabled (reduced motion)", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <PulseDot testID="pd" animate={false} color="#4A90D9" />,
      );
    });
    await act(async () => {});
    expect(queryId(comp, "pd")).toBeTruthy();
    // No animated halo node when motion is disabled.
    expect(queryId(comp, "pd-halo")).toBeNull();
    act(() => comp.unmount());
  });

  it("mounts the animated halo when motion is allowed", async () => {
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(<PulseDot testID="pd" />);
    });
    // Flush the reduce-motion query so the halo mounts.
    await act(async () => {});
    expect(queryId(comp, "pd")).toBeTruthy();
    expect(queryId(comp, "pd-halo")).toBeTruthy();
    act(() => comp.unmount());
  });
});
