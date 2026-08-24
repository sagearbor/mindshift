import React from "react";
import { Alert } from "react-native";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import PeopleScreen from "../src/screens/PeopleScreen";
import {
  deleteVoicePerson,
  enrollPersonFromRecording,
  getRecording,
  listRecordings,
  listVoicePeople,
  renameVoicePerson,
} from "../src/api/client";

jest.mock("../src/api/client", () => ({
  listVoicePeople: jest.fn(),
  renameVoicePerson: jest.fn(),
  deleteVoicePerson: jest.fn(),
  enrollPersonFromRecording: jest.fn(),
  listRecordings: jest.fn(),
  getRecording: jest.fn(),
}));

// The guided flow is its own tested component; here a stub that records the
// props PeopleScreen wires (the person + the completion callback).
let mockVtProps: { onDone: (n: number) => void; person?: { personId: string; displayName: string } } | null = null;
jest.mock("../src/components/VoiceTrainingFlow", () => {
  const React = require("react");
  const { View } = require("react-native");
  return {
    __esModule: true,
    default: (props: { onDone: (n: number) => void; person?: { personId: string; displayName: string } }) => {
      mockVtProps = props;
      return React.createElement(View, { testID: "voice-training-flow" });
    },
  };
});

const mockList = listVoicePeople as jest.Mock;
const mockRename = renameVoicePerson as jest.Mock;
const mockDelete = deleteVoicePerson as jest.Mock;
const mockEnroll = enrollPersonFromRecording as jest.Mock;
const mockRecordings = listRecordings as jest.Mock;
const mockGetRecording = getRecording as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance | null): string {
  if (!node) return "";
  const out: string[] = [];
  for (const n of node.findAll((n) => typeof n.type === "string")) {
    const c = n.props?.children;
    if (typeof c === "string") out.push(c);
    else if (Array.isArray(c)) out.push(...c.filter((x): x is string => typeof x === "string"));
  }
  return out.join(" ");
}

const SELF = {
  available: true, storage_enabled: true, enrolled: true, enroll_count: 2,
  person_id: "self", display_name: "You", is_self: true,
  samples: [{ id: "a", recording_id: null, speaker: null, at: null, note: "guided enrollment" }],
};
const MOM = {
  available: true, storage_enabled: true, enrolled: true, enroll_count: 1,
  person_id: "mom", display_name: "Mom", is_self: false,
  samples: [{ id: "b", recording_id: "r1", speaker: "Speaker B", at: null, seconds: 12.4 }],
};

async function mount() {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<PeopleScreen onBack={jest.fn()} />);
  });
  return comp;
}

beforeEach(() => {
  mockList.mockReset();
  mockRename.mockReset();
  mockDelete.mockReset();
  mockEnroll.mockReset();
  mockRecordings.mockReset();
  mockGetRecording.mockReset();
  mockVtProps = null;
  mockList.mockResolvedValue({ available: true, storage_enabled: true, people: [MOM, SELF] });
});

describe("PeopleScreen", () => {
  it("lists people with You pinned first and a samples/seconds summary", async () => {
    const comp = await mount();
    const rows = comp.root.findAll(
      (n) => typeof n.type === "string" && typeof n.props?.testID === "string" && n.props.testID.startsWith("people-row-"),
    );
    expect(rows.map((r) => r.props.testID)).toEqual(["people-row-self", "people-row-mom"]);
    expect(textOf(queryId(comp, "people-summary-self"))).toContain("2 samples");
    expect(textOf(queryId(comp, "people-summary-self"))).toContain("that’s you");
    expect(textOf(queryId(comp, "people-summary-mom")).trim()).toBe("1 sample · 12 s of speech");
    // The owner is always "You" — no rename for self.
    expect(queryId(comp, "people-rename-self")).toBeNull();
    expect(queryId(comp, "people-rename-mom")).toBeTruthy();
  });

  it("says honestly when the server can't do voice ID", async () => {
    mockList.mockResolvedValue({ available: false, storage_enabled: true, people: [] });
    const comp = await mount();
    expect(textOf(queryId(comp, "people-unavailable"))).toContain("isn’t available on this server");
    expect(queryId(comp, "people-add-person")).toBeNull();
  });

  it("renames a partner through the server and shows the new name", async () => {
    mockRename.mockResolvedValue({ ...MOM, display_name: "Mum" });
    const comp = await mount();
    act(() => queryId(comp, "people-rename-mom")!.props.onPress());
    act(() => queryId(comp, "people-rename-input-mom")!.props.onChangeText("Mum"));
    await act(async () => {
      queryId(comp, "people-rename-save-mom")!.props.onPress();
    });
    expect(mockRename).toHaveBeenCalledWith("mom", "Mum");
    expect(textOf(queryId(comp, "people-name-mom"))).toBe("Mum");
  });

  it("deletes only after confirmation, then drops the row", async () => {
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    mockDelete.mockResolvedValue({ deleted: true, person_id: "mom" });
    const comp = await mount();
    act(() => queryId(comp, "people-delete-mom")!.props.onPress());
    expect(mockDelete).not.toHaveBeenCalled();
    const buttons = alertSpy.mock.calls[0][2] as { text: string; onPress?: () => void }[];
    await act(async () => {
      buttons.find((b) => b.text === "Forget")!.onPress!();
    });
    expect(mockDelete).toHaveBeenCalledWith("mom");
    expect(queryId(comp, "people-row-mom")).toBeNull();
    expect(queryId(comp, "people-row-self")).toBeTruthy();
    alertSpy.mockRestore();
  });

  it("Add person → name → record hands the guided flow the new person, and refreshes on done", async () => {
    const comp = await mount();
    act(() => queryId(comp, "people-add-person")!.props.onPress());
    act(() => queryId(comp, "people-new-name")!.props.onChangeText("Dad"));
    await act(async () => {
      queryId(comp, "people-method-record")!.props.onPress();
    });
    expect(queryId(comp, "voice-training-flow")).toBeTruthy();
    expect(mockVtProps?.person).toEqual({ personId: "dad", displayName: "Dad" });
    mockList.mockResolvedValue({ available: true, storage_enabled: true, people: [MOM, SELF, { ...MOM, person_id: "dad", display_name: "Dad" }] });
    await act(async () => {
      mockVtProps!.onDone(1);
    });
    expect(textOf(queryId(comp, "people-add-success"))).toContain("Dad’s voice profile now blends 1 sample");
    expect(queryId(comp, "people-row-dad")).toBeTruthy();
  });

  it("Add person → from a recording lists analyzed recordings with audio, then the speakers, then enrolls", async () => {
    mockRecordings.mockResolvedValue([
      { id: "r1", created_at: "2026-08-01T00:00:00Z", filename: "a.m4a", title: "Dinner", media_type: "audio", duration_seconds: 60, has_analysis: true },
      { id: "live", created_at: "2026-08-02T00:00:00Z", filename: "live-session", title: "Live", media_type: "none", duration_seconds: 60, has_analysis: true },
      { id: "raw", created_at: "2026-08-03T00:00:00Z", filename: "b.m4a", title: "Unanalyzed", media_type: "audio", duration_seconds: 60, has_analysis: false },
    ]);
    mockGetRecording.mockResolvedValue({
      id: "r1", turns: [
        { speaker: "Speaker A", text: "hi", start_time: 0, end_time: 1 },
        { speaker: "Speaker B", text: "yo", start_time: 1, end_time: 2 },
        { speaker: "Speaker A", text: "ok", start_time: 2, end_time: 3 },
      ],
    });
    mockEnroll.mockResolvedValue({
      enrolled: true, person_id: "dad", display_name: "Dad", is_self: false, created: true,
      enroll_count: 1, seconds: 9.6, dim: 192, updated_at: "t", speaker_labels: {}, stored: "…",
    });
    const comp = await mount();
    act(() => queryId(comp, "people-add-person")!.props.onPress());
    act(() => queryId(comp, "people-new-name")!.props.onChangeText("Dad"));
    await act(async () => {
      queryId(comp, "people-method-recording")!.props.onPress();
    });
    // Only analyzed recordings that kept audio are offered.
    expect(queryId(comp, "people-recording-r1")).toBeTruthy();
    expect(queryId(comp, "people-recording-live")).toBeNull();
    expect(queryId(comp, "people-recording-raw")).toBeNull();
    await act(async () => {
      queryId(comp, "people-recording-r1")!.props.onPress();
    });
    expect(queryId(comp, "people-speaker-Speaker A")).toBeTruthy();
    expect(queryId(comp, "people-speaker-Speaker B")).toBeTruthy();
    await act(async () => {
      queryId(comp, "people-speaker-Speaker B")!.props.onPress();
    });
    expect(mockEnroll).toHaveBeenCalledWith("dad", "r1", "Speaker B", "Dad");
    expect(textOf(queryId(comp, "people-add-success"))).toContain("Learned 10 s of Dad’s voice from “Dinner”");
  });

  it("shows the server's refusal inline when the picked speaker has too little speech", async () => {
    mockRecordings.mockResolvedValue([
      { id: "r1", created_at: "2026-08-01T00:00:00Z", filename: "a.m4a", title: "Dinner", media_type: "audio", duration_seconds: 60, has_analysis: true },
    ]);
    mockGetRecording.mockResolvedValue({ id: "r1", turns: [{ speaker: "Speaker C", text: "hm", start_time: 0, end_time: 1 }] });
    const refusal = new Error("[too-little-speech] only 1.0s of that speaker's voice is in this recording") as Error & { status?: number; detail?: string };
    refusal.status = 422;
    refusal.detail = refusal.message;
    mockEnroll.mockRejectedValue(refusal);
    const comp = await mount();
    // Add a sample for an EXISTING person (no name step).
    act(() => queryId(comp, "people-add-sample-mom")!.props.onPress());
    await act(async () => {
      queryId(comp, "people-method-recording")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "people-recording-r1")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "people-speaker-Speaker C")!.props.onPress();
    });
    expect(mockEnroll).toHaveBeenCalledWith("mom", "r1", "Speaker C", undefined);
    const error = textOf(queryId(comp, "people-add-error"));
    expect(error).toContain("Not enough of their voice here");
    expect(error).toContain("only 1.0s");
    // Still on the speaker picker — the user can pick another speaker.
    expect(queryId(comp, "people-pick-speaker")).toBeTruthy();
  });
});
