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

  // N7 fix round 1 (IMPORTANT 4): a persisted absolute file:// uri can go
  // stale (e.g. an iOS app-container path change across reinstalls/updates)
  // with no fallback — Avatar branched purely on photoUri truthiness, so a
  // dead uri rendered an <Image> that silently failed: a blank circle
  // instead of the honest initial-letter fallback the component otherwise
  // guarantees.
  describe("stale photo fallback", () => {
    it("falls back to the honest initial-letter circle after the image fails to load", () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(
          <Avatar
            user={{ displayName: "Sophie", email: null }}
            photoUri="file:///stale.jpg"
            testID="avatar"
          />,
        );
      });
      expect(queryId(comp, "avatar-photo")).toBeTruthy();
      expect(queryId(comp, "avatar-initial")).toBeNull();

      act(() => {
        queryId(comp, "avatar-photo")!.props.onError();
      });

      expect(queryId(comp, "avatar-photo")).toBeNull();
      expect(queryId(comp, "avatar-initial")).toBeTruthy();
      act(() => comp.unmount());
    });

    it("resets the fallback and retries the photo once a new photoUri is set", () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = renderer.create(
          <Avatar
            user={{ displayName: "Sophie", email: null }}
            photoUri="file:///stale.jpg"
            testID="avatar"
          />,
        );
      });
      act(() => {
        queryId(comp, "avatar-photo")!.props.onError();
      });
      expect(queryId(comp, "avatar-initial")).toBeTruthy();

      // A fresh capture (a new, different uri) should get its own chance to
      // load rather than staying stuck on the fallback forever.
      act(() => {
        comp.update(
          <Avatar
            user={{ displayName: "Sophie", email: null }}
            photoUri="file:///fresh.jpg"
            testID="avatar"
          />,
        );
      });
      expect(queryId(comp, "avatar-photo")).toBeTruthy();
      expect(queryId(comp, "avatar-initial")).toBeNull();
      act(() => comp.unmount());
    });
  });
});
