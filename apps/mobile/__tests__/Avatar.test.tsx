import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import Avatar, { accountInitial } from "../src/components/Avatar";

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

describe("accountInitial", () => {
  it("prefers displayName over email", () => {
    expect(
      accountInitial({ displayName: "Sophie", email: "s@example.com" }),
    ).toBe("S");
  });

  it("falls back to email when there's no displayName", () => {
    expect(accountInitial({ displayName: null, email: "a@example.com" })).toBe(
      "A",
    );
  });

  it("falls back to '?' when there's neither", () => {
    expect(accountInitial({ displayName: null, email: null })).toBe("?");
    expect(accountInitial(null)).toBe("?");
  });

  it("uppercases a lowercase source", () => {
    expect(accountInitial({ displayName: null, email: "zed@example.com" })).toBe(
      "Z",
    );
  });
});

describe("Avatar", () => {
  it("renders the initial circle when no photo is set (the honest empty state)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <Avatar user={{ displayName: "Sophie", email: null }} testID="avatar" />,
      );
    });
    expect(queryId(comp, "avatar-initial")).toBeTruthy();
    expect(queryId(comp, "avatar-photo")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders a photo when photoUri is set (Task N6's slot)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <Avatar
          user={{ displayName: "Sophie", email: null }}
          photoUri="file:///selfie.jpg"
          testID="avatar"
        />,
      );
    });
    expect(queryId(comp, "avatar-photo")).toBeTruthy();
    expect(queryId(comp, "avatar-initial")).toBeNull();
    act(() => comp.unmount());
  });
});
