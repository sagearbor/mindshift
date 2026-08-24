import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import WhoIsThisSheet from "../src/components/WhoIsThisSheet";
import { enrollPersonFromRecording, patchSpeakerLabels } from "../src/api/client";
import type { VoicePerson } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  patchSpeakerLabels: jest.fn(),
  enrollPersonFromRecording: jest.fn(),
}));

const mockPatch = patchSpeakerLabels as jest.Mock;
const mockEnroll = enrollPersonFromRecording as jest.Mock;

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

function person(over: Partial<VoicePerson>): VoicePerson {
  return {
    available: true, storage_enabled: true, enrolled: true, enroll_count: 2,
    person_id: "mom", display_name: "Mom", is_self: false, samples: [], ...over,
  };
}

const PEOPLE = [
  person({ person_id: "mom", display_name: "Mom" }),
  person({ person_id: "self", display_name: null, is_self: true }),
];

function labelsResult(speaker: string, name: string, pid?: string) {
  return {
    id: "r1",
    manual_speaker_labels: { [speaker]: name },
    manual_speaker_people: pid ? { [speaker]: pid } : {},
    speaker_labels: {
      [speaker]: pid
        ? { display_label: name, label_source: "manual-person", person_id: pid }
        : { display_label: name, label_source: "manual" },
    },
  };
}

function mount(over: Partial<React.ComponentProps<typeof WhoIsThisSheet>> = {}) {
  const props = {
    visible: true,
    recordingId: "r1",
    speaker: "Speaker B",
    currentLabel: "Speaker B",
    people: PEOPLE,
    hasAudio: true,
    onClose: jest.fn(),
    onLabeled: jest.fn(),
    onEnrolled: jest.fn(),
    ...over,
  };
  let comp!: renderer.ReactTestRenderer;
  act(() => {
    comp = renderer.create(<WhoIsThisSheet {...props} />);
  });
  return { comp, props };
}

beforeEach(() => {
  mockPatch.mockReset();
  mockEnroll.mockReset();
});

describe("WhoIsThisSheet", () => {
  it("lists enrolled people with You pinned first, plus New person…", () => {
    const { comp } = mount();
    const rows = comp.root.findAll(
      (n) => typeof n.type === "string" && typeof n.props?.testID === "string" && n.props.testID.startsWith("who-person-"),
    );
    expect(rows.map((r) => r.props.testID)).toEqual(["who-person-self", "who-person-mom"]);
    expect(queryId(comp, "who-new-person")).toBeTruthy();
    // No manual name yet → no "clear" row.
    expect(queryId(comp, "who-clear")).toBeNull();
  });

  it("picking a person relabels with the person id, then offers Remember this voice, and enrolls", async () => {
    mockPatch.mockResolvedValue(labelsResult("Speaker B", "Mom", "mom"));
    mockEnroll.mockResolvedValue({
      enrolled: true, person_id: "mom", display_name: "Mom", is_self: false, created: false,
      enroll_count: 3, seconds: 12.4, dim: 192, updated_at: "t", speaker_labels: {}, stored: "…",
    });
    const { comp, props } = mount();

    await act(async () => {
      queryId(comp, "who-person-mom")!.props.onPress();
    });
    expect(mockPatch).toHaveBeenCalledWith("r1", { "Speaker B": "Mom" }, { "Speaker B": "mom" });
    expect(props.onLabeled).toHaveBeenCalledTimes(1);
    expect(queryId(comp, "who-remember-stage")).toBeTruthy();

    await act(async () => {
      queryId(comp, "who-remember")!.props.onPress();
    });
    expect(mockEnroll).toHaveBeenCalledWith("mom", "r1", "Speaker B", undefined);
    expect(props.onEnrolled).toHaveBeenCalledTimes(1);
    const done = textOf(queryId(comp, "who-done-text"));
    expect(done).toContain("Learned 12 s of Mom’s voice");
    expect(done).toContain("3 samples");
    act(() => queryId(comp, "who-done")!.props.onPress());
    expect(props.onClose).toHaveBeenCalled();
  });

  it("New person… saves a free-text name first, then remembers it by creating the person", async () => {
    mockPatch.mockResolvedValue(labelsResult("Speaker B", "Dad"));
    mockEnroll.mockResolvedValue({
      enrolled: true, person_id: "dad", display_name: "Dad", is_self: false, created: true,
      enroll_count: 1, seconds: 8, dim: 192, updated_at: "t", speaker_labels: {}, stored: "…",
    });
    const { comp, props } = mount();
    act(() => queryId(comp, "who-new-person")!.props.onPress());
    act(() => queryId(comp, "who-name-input")!.props.onChangeText("Dad"));
    await act(async () => {
      queryId(comp, "who-save-name")!.props.onPress();
    });
    // Name only — the person doesn't exist yet (a person is a voiceprint).
    expect(mockPatch).toHaveBeenLastCalledWith("r1", { "Speaker B": "Dad" }, undefined);
    expect(queryId(comp, "who-remember-stage")).toBeTruthy();

    await act(async () => {
      queryId(comp, "who-remember")!.props.onPress();
    });
    // Created with the display name, then the new id is attached to the label.
    expect(mockEnroll).toHaveBeenCalledWith("dad", "r1", "Speaker B", "Dad");
    expect(mockPatch).toHaveBeenLastCalledWith("r1", { "Speaker B": "Dad" }, { "Speaker B": "dad" });
    expect(props.onLabeled).toHaveBeenCalledTimes(2);
    expect(props.onEnrolled).toHaveBeenCalledTimes(1);
  });

  it("shows the server's honest 422 reasons inline and keeps the name", async () => {
    mockPatch.mockResolvedValue(labelsResult("Speaker B", "Mom", "mom"));
    const refusal = new Error("[sounds-like-someone-else] that voice sounds like You (similarity 0.81)") as Error & {
      status?: number; detail?: string;
    };
    refusal.status = 422;
    refusal.detail = refusal.message;
    mockEnroll.mockRejectedValue(refusal);
    const { comp, props } = mount();
    await act(async () => {
      queryId(comp, "who-person-mom")!.props.onPress();
    });
    await act(async () => {
      queryId(comp, "who-remember")!.props.onPress();
    });
    const error = textOf(queryId(comp, "who-error"));
    expect(error).toContain("That sounds like someone else");
    expect(error).toContain("that voice sounds like You (similarity 0.81)");
    expect(props.onEnrolled).not.toHaveBeenCalled();
    // The label stays; the user can keep it without a voiceprint.
    expect(queryId(comp, "who-skip-remember")).toBeTruthy();
    expect(textOf(queryId(comp, "who-skip-remember"))).toContain("Keep the name anyway");
    act(() => queryId(comp, "who-skip-remember")!.props.onPress());
    expect(props.onClose).toHaveBeenCalled();
  });

  it("a live session (no audio) labels without offering to remember, and says why", async () => {
    mockPatch.mockResolvedValue(labelsResult("Speaker B", "Dad"));
    const { comp } = mount({ hasAudio: false });
    act(() => queryId(comp, "who-new-person")!.props.onPress());
    act(() => queryId(comp, "who-name-input")!.props.onChangeText("Dad"));
    await act(async () => {
      queryId(comp, "who-save-name")!.props.onPress();
    });
    expect(queryId(comp, "who-remember")).toBeNull();
    expect(textOf(queryId(comp, "who-done-text"))).toContain("Live sessions keep no audio");
    expect(mockEnroll).not.toHaveBeenCalled();
  });

  it("offers to clear a name that was set, and closes on success", async () => {
    mockPatch.mockResolvedValue({ id: "r1", manual_speaker_labels: {}, speaker_labels: {} });
    const { comp, props } = mount({ currentLabel: "Mom", currentPersonId: "mom" });
    await act(async () => {
      queryId(comp, "who-clear")!.props.onPress();
    });
    expect(mockPatch).toHaveBeenCalledWith("r1", { "Speaker B": "" }, undefined);
    expect(props.onClose).toHaveBeenCalled();
  });
});
