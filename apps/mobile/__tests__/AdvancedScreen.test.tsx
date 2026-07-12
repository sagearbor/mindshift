import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import AdvancedScreen from "../src/screens/AdvancedScreen";

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

describe("AdvancedScreen", () => {
  it("renders the dashboard entry, sign out, and back — and wires each press", () => {
    const onBack = jest.fn();
    const onOpenDashboard = jest.fn();
    const onSignOut = jest.fn();

    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <AdvancedScreen
          onBack={onBack}
          onOpenDashboard={onOpenDashboard}
          onSignOut={onSignOut}
        />,
      );
    });

    act(() => queryId(comp, "advanced-dashboard")!.props.onPress());
    expect(onOpenDashboard).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "advanced-sign-out")!.props.onPress());
    expect(onSignOut).toHaveBeenCalledTimes(1);

    act(() => queryId(comp, "advanced-back")!.props.onPress());
    expect(onBack).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });
});
