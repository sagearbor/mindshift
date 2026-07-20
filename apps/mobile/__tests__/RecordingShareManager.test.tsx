import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import RecordingShareManager from "../src/components/RecordingShareManager";
import { postShare, deleteShare } from "../src/api/client";
import type { RecordingShare } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  postShare: jest.fn(),
  deleteShare: jest.fn(),
}));
const mockPost = postShare as jest.Mock;
const mockDelete = deleteShare as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function setText(comp: renderer.ReactTestRenderer, id: string, text: string) {
  act(() => comp.root.find((n) => n.props?.testID === id).props.onChangeText(text));
}

function press(comp: renderer.ReactTestRenderer, id: string) {
  return comp.root.find((n) => n.props?.testID === id).props.onPress();
}

const existing: RecordingShare[] = [
  { uid: "u-sage", email: "sage@example.com", created_at: "2026-07-03T00:00:00Z" },
];

beforeEach(() => {
  mockPost.mockReset();
  mockDelete.mockReset();
});

describe("RecordingShareManager", () => {
  it("shares an email and shows the success state + updated list", async () => {
    mockPost.mockResolvedValueOnce({
      shares: [
        { uid: "u-ari", email: "ari@example.com", created_at: "2026-07-05T00:00:00Z" },
      ],
    });
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingShareManager recordingId="r1" initialShares={[]} />,
      );
    });

    act(() => press(comp, "share-open-button"));
    setText(comp, "share-email-input", "ari@example.com");
    await act(async () => {
      press(comp, "share-submit");
    });
    await act(async () => {});

    expect(mockPost).toHaveBeenCalledWith("r1", "ari@example.com");
    const success = queryId(comp, "share-success");
    expect(success).toBeTruthy();
    expect(JSON.stringify(success!.props.children)).toContain("ari@example.com");
    // The returned share now appears in the list.
    expect(queryId(comp, "share-row-u-ari")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("surfaces the no-account error verbatim (never fabricates success)", async () => {
    const err = Object.assign(new Error("no MindShift account with that email"), {
      status: 404,
      detail: "no MindShift account with that email",
    });
    mockPost.mockRejectedValueOnce(err);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingShareManager recordingId="r1" initialShares={[]} />,
      );
    });

    act(() => press(comp, "share-open-button"));
    setText(comp, "share-email-input", "ghost@example.com");
    await act(async () => {
      press(comp, "share-submit");
    });
    await act(async () => {});

    const error = queryId(comp, "share-error");
    expect(error).toBeTruthy();
    expect(JSON.stringify(error!.props.children)).toContain(
      "no MindShift account with that email",
    );
    expect(queryId(comp, "share-success")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders existing shares and removes one with ✕", async () => {
    mockDelete.mockResolvedValueOnce(undefined);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingShareManager recordingId="r1" initialShares={existing} />,
      );
    });

    expect(queryId(comp, "share-row-u-sage")).toBeTruthy();
    await act(async () => {
      press(comp, "share-remove-u-sage");
    });
    await act(async () => {});

    expect(mockDelete).toHaveBeenCalledWith("r1", "u-sage");
    expect(queryId(comp, "share-row-u-sage")).toBeNull();
    act(() => comp.unmount());
  });

  it("keeps the row and shows an error when removal fails", async () => {
    mockDelete.mockRejectedValueOnce(new Error("API error: 503"));
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingShareManager recordingId="r1" initialShares={existing} />,
      );
    });

    await act(async () => {
      press(comp, "share-remove-u-sage");
    });
    await act(async () => {});

    expect(queryId(comp, "share-row-u-sage")).toBeTruthy();
    expect(queryId(comp, "share-remove-error")).toBeTruthy();
    act(() => comp.unmount());
  });
});
