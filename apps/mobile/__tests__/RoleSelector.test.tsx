import React from "react";
import renderer, { act } from "react-test-renderer";
import RoleSelector, { ROLES } from "../src/components/RoleSelector";

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


describe("RoleSelector", () => {
  it("renders correctly with no selection", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<RoleSelector selectedRole="" onSelect={jest.fn()} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly with a selected role", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<RoleSelector selectedRole="Husband / Wife" onSelect={jest.fn()} />,));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("calls onSelect when a role is pressed", () => {
    const onSelect = jest.fn();
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(
        <RoleSelector selectedRole="" onSelect={onSelect} />,
      ));
    });

    const firstButton = component!.root.findByProps({ testID: `role-${ROLES[0]}` });
    firstButton.props.onPress();
    expect(onSelect).toHaveBeenCalledWith(ROLES[0]);
  });
});
