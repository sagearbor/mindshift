import React from "react";
import renderer from "react-test-renderer";
import RoleSelector, { ROLES } from "../src/components/RoleSelector";

describe("RoleSelector", () => {
  it("renders correctly with no selection", () => {
    const tree = renderer
      .create(<RoleSelector selectedRole="" onSelect={jest.fn()} />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("renders correctly with a selected role", () => {
    const tree = renderer
      .create(
        <RoleSelector selectedRole="Husband / Wife" onSelect={jest.fn()} />,
      )
      .toJSON();
    expect(tree).toMatchSnapshot();
  });

  it("calls onSelect when a role is pressed", () => {
    const onSelect = jest.fn();
    const component = renderer.create(
      <RoleSelector selectedRole="" onSelect={onSelect} />,
    );

    const firstButton = component.root.findByProps({ testID: `role-${ROLES[0]}` });
    firstButton.props.onPress();
    expect(onSelect).toHaveBeenCalledWith(ROLES[0]);
  });
});
