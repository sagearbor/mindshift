import React from "react";
import { Text } from "react-native";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import PushedScreenChrome from "../src/components/PushedScreenChrome";

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

describe("PushedScreenChrome", () => {
  it("renders a Home tap target above the wrapped screen content", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <PushedScreenChrome onGoHome={jest.fn()}>
          <Text testID="pushed-child">screen content</Text>
        </PushedScreenChrome>,
      );
    });
    expect(queryId(comp, "pushed-chrome-home-button")).toBeTruthy();
    expect(queryId(comp, "pushed-child")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("tapping Home calls onGoHome exactly once", () => {
    const onGoHome = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <PushedScreenChrome onGoHome={onGoHome}>
          <Text>content</Text>
        </PushedScreenChrome>,
      );
    });
    act(() => queryId(comp, "pushed-chrome-home-button")!.props.onPress());
    expect(onGoHome).toHaveBeenCalledTimes(1);
    act(() => comp.unmount());
  });

  it("exposes an accessible label for the Home tap target", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <PushedScreenChrome onGoHome={jest.fn()}>
          <Text>content</Text>
        </PushedScreenChrome>,
      );
    });
    const button = queryId(comp, "pushed-chrome-home-button")!;
    expect(button.props.accessibilityRole).toBe("button");
    expect(button.props.accessibilityLabel).toBe("Home");
    act(() => comp.unmount());
  });
});
