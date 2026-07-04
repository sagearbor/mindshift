import React from "react";
import renderer, { act } from "react-test-renderer";
import RoleSelector, { ROLES } from "../src/components/RoleSelector";

describe("RoleSelector", () => {
  it("renders correctly with no selection", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<RoleSelector selectedRole="" onSelect={jest.fn()} />);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("renders correctly with a selected role", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(<RoleSelector selectedRole="Husband / Wife" onSelect={jest.fn()} />,);
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("calls onSelect when a role is pressed", () => {
    const onSelect = jest.fn();
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = renderer.create(
        <RoleSelector selectedRole="" onSelect={onSelect} />,
      );
    });

    const firstButton = component!.root.findByProps({ testID: `role-${ROLES[0]}` });
    firstButton.props.onPress();
    expect(onSelect).toHaveBeenCalledWith(ROLES[0]);
  });
});
