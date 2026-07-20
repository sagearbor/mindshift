import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import RecordingsScreen, {
  formatParticipants,
} from "../src/screens/RecordingsScreen";
import { listRecordingsAndShared, deleteRecording } from "../src/api/client";
import type {
  RecordingSummary,
  SharedRecordingSummary,
} from "../src/api/client";

jest.mock("../src/api/client", () => ({
  listRecordingsAndShared: jest.fn(),
  deleteRecording: jest.fn(),
}));
const mockList = listRecordingsAndShared as jest.Mock;
const mockDelete = deleteRecording as jest.Mock;

/** Resolve the list call with owned + (optional) shared recordings. */
function resolveList(
  own: RecordingSummary[],
  shared: SharedRecordingSummary[] = [],
) {
  mockList.mockResolvedValueOnce({ recordings: own, sharedWithMe: shared });
}

const recordings: RecordingSummary[] = [
  {
    id: "r1",
    created_at: "2026-07-01T10:00:00Z",
    filename: "kitchen-fight.m4a",
    media_type: "audio",
    duration_seconds: 182,
    has_analysis: true,
  },
  {
    id: "r2",
    created_at: "2026-07-02T10:00:00Z",
    filename: "living-room.mp4",
    media_type: "video",
    duration_seconds: 95,
    has_analysis: false,
  },
];

const shared: SharedRecordingSummary[] = [
  {
    id: "s1",
    created_at: "2026-07-03T10:00:00Z",
    filename: "moms-call.m4a",
    title: "Sunday call",
    media_type: "audio",
    duration_seconds: 240,
    has_analysis: true,
    owner_email: "linda@example.com",
    shared: true,
  },
];

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

beforeEach(() => {
  mockList.mockReset();
  mockDelete.mockReset();
});

describe("RecordingsScreen", () => {
  it("lists recordings and opens the replay on tap", async () => {
    resolveList(recordings);
    const onSelect = jest.fn();

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={onSelect} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "recordings-list")).toBeTruthy();
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    expect(queryId(comp, "recording-r2")).toBeTruthy();

    act(() => comp.root.find((n) => n.props?.testID === "recording-open-r1").props.onPress());
    expect(onSelect).toHaveBeenCalledWith("r1");
    act(() => comp.unmount());
  });

  it("shows the honest empty state", async () => {
    resolveList([]);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "recordings-empty")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("No stored recordings yet");
    act(() => comp.unmount());
  });

  it("shows the honest 503 error state", async () => {
    mockList.mockRejectedValueOnce(new Error("API error: 503"));
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "recordings-error")).toBeTruthy();
    expect(JSON.stringify(comp.toJSON())).toContain("Replay storage");
    act(() => comp.unmount());
  });

  it("deletes a recording through the inline confirm flow", async () => {
    resolveList(recordings);
    mockDelete.mockResolvedValueOnce(undefined);

    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    // First tap on Delete reveals the confirm row (no network yet).
    act(() => comp.root.find((n) => n.props?.testID === "recording-delete-r1").props.onPress());
    expect(queryId(comp, "confirm-r1")).toBeTruthy();
    expect(mockDelete).not.toHaveBeenCalled();

    // Confirm → DELETE fires and the row is removed.
    await act(async () => {
      comp.root.find((n) => n.props?.testID === "confirm-yes-r1").props.onPress();
    });
    await act(async () => {});

    expect(mockDelete).toHaveBeenCalledWith("r1");
    expect(queryId(comp, "recording-r1")).toBeNull();
    // The other recording remains.
    expect(queryId(comp, "recording-r2")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("cancels the delete without calling the API", async () => {
    resolveList(recordings);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    act(() => comp.root.find((n) => n.props?.testID === "recording-delete-r1").props.onPress());
    act(() => comp.root.find((n) => n.props?.testID === "confirm-no-r1").props.onPress());
    expect(queryId(comp, "confirm-r1")).toBeNull();
    expect(mockDelete).not.toHaveBeenCalled();
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("shows named participants from the list's manual_speaker_labels, and nothing when none", async () => {
    resolveList([
      {
        ...recordings[0],
        manual_speaker_labels: { "Speaker A": "Linda", "Speaker B": "Sage" },
      },
      // r2 has an empty manual map → no participant line (never fabricated).
      { ...recordings[1], manual_speaker_labels: {} },
    ]);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    const parts = queryId(comp, "recording-participants-r1");
    expect(parts).toBeTruthy();
    expect(JSON.stringify(parts!.props.children)).toContain("Linda & Sage");
    // The unnamed recording shows no participant line.
    expect(queryId(comp, "recording-participants-r2")).toBeNull();
    act(() => comp.unmount());
  });

  it("renders the Shared with me section and opens a shared recording in replay", async () => {
    resolveList(recordings, shared);
    const onSelect = jest.fn();
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={onSelect} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "shared-with-me-section")).toBeTruthy();
    // The from-line shows the owner's email, verbatim.
    const from = queryId(comp, "shared-from-s1");
    expect(from).toBeTruthy();
    expect(JSON.stringify(from!.props.children)).toContain("linda@example.com");
    // Tapping opens the normal replay for that id (read-only handled in Replay).
    act(() => comp.root.find((n) => n.props?.testID === "shared-open-s1").props.onPress());
    expect(onSelect).toHaveBeenCalledWith("s1");
    act(() => comp.unmount());
  });

  it("shows only the shared section when the user owns nothing", async () => {
    resolveList([], shared);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    // Not the empty state — there IS something to show.
    expect(queryId(comp, "recordings-empty")).toBeNull();
    expect(queryId(comp, "shared-with-me-section")).toBeTruthy();
    expect(queryId(comp, "shared-open-s1")).toBeTruthy();
    act(() => comp.unmount());
  });

  it("defensively renders no shared section when the server omits it", async () => {
    // Older server: sharedWithMe is [] (client already normalized absent → []).
    resolveList(recordings, []);
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <RecordingsScreen onSelectRecording={() => {}} onBack={() => {}} />,
      );
    });
    await act(async () => {});

    expect(queryId(comp, "shared-with-me-section")).toBeNull();
    expect(queryId(comp, "recording-r1")).toBeTruthy();
    act(() => comp.unmount());
  });
});

describe("formatParticipants", () => {
  it("returns null when there are no names (absent, empty, or blank)", () => {
    expect(formatParticipants(undefined)).toBeNull();
    expect(formatParticipants({})).toBeNull();
    expect(formatParticipants({ "Speaker A": "  " })).toBeNull();
  });

  it("joins names honestly by count", () => {
    expect(formatParticipants({ a: "Linda" })).toBe("Linda");
    expect(formatParticipants({ a: "Linda", b: "Sage" })).toBe("Linda & Sage");
    expect(formatParticipants({ a: "Linda", b: "Sage", c: "Ari" })).toBe(
      "Linda, Sage & Ari",
    );
    expect(
      formatParticipants({ a: "Linda", b: "Sage", c: "Ari", d: "Bo" }),
    ).toBe("Linda, Sage & 2 more");
  });
});
