import React from "react";
import renderer, { act, type ReactTestInstance } from "react-test-renderer";
import SpeakerNaming from "../src/components/SpeakerNaming";
import type { RecordingTurn } from "../src/api/client";
import { getSpeakerColor } from "../src/utils/speakerColors";

jest.mock("../src/api/client", () => ({
  patchSpeakerLabels: jest.fn(),
}));

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

// style is [styles.dot, { backgroundColor }] — an array, not a flat object —
// same convention HeatChart's legend swatches use.
function swatchBg(n: ReactTestInstance): string {
  return (n.props.style as { backgroundColor?: string }[]).find((s) => s?.backgroundColor)!
    .backgroundColor!;
}

// Exact turns from the project's real fixture
// (server/tests/fixtures/audio/test_recording_family_real_meta.json) whose raw
// speaker ids "Sage" and "Asher" are the confirmed getSpeakerColor collision
// ("Sage"/"Asher" both hash to "#8B5CF6") that HeatChart's resolveSpeakerColors
// already fixes for the chart's own swatches/dashes.
const realTurns: RecordingTurn[] = [
  { speaker: "Sage", text: "Hey, how was your day?", start_time: 0.96, end_time: 6.08 },
  { speaker: "Asher", text: "It was fine, thanks.", start_time: 6.08, end_time: 8.82 },
];

describe("SpeakerNaming color/speaker-mismatch fix (matches HeatChart's resolveSpeakerColors)", () => {
  it("BUG (if unfixed): plain getSpeakerColor collides for 'Sage'/'Asher' — documents the raw defect this component must not reproduce", () => {
    expect(getSpeakerColor("Sage")).toBe(getSpeakerColor("Asher"));
    expect(getSpeakerColor("Sage")).toBe("#8B5CF6");
  });

  it("renders Sage's and Asher's swatches in genuinely distinct colors — the regression test for the cross-screen collision gap", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <SpeakerNaming
          recordingId="r1"
          turns={realTurns}
          manualLabels={{}}
          onSaved={jest.fn()}
        />,
      );
    });

    const sageSwatch = queryId(comp, "name-swatch-Sage");
    const asherSwatch = queryId(comp, "name-swatch-Asher");
    expect(sageSwatch).toBeTruthy();
    expect(asherSwatch).toBeTruthy();

    const sageColor = swatchBg(sageSwatch!);
    const asherColor = swatchBg(asherSwatch!);
    expect(sageColor).toBeTruthy();
    expect(asherColor).toBeTruthy();
    // Before the fix these were both "#8B5CF6" — the confirmed collision.
    expect(sageColor).not.toBe(asherColor);

    act(() => comp.unmount());
  });

  it("keeps the swatch color stable across the editing row for the same speaker (name-edit toggles the row, not the color)", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <SpeakerNaming
          recordingId="r1"
          turns={realTurns}
          manualLabels={{}}
          onSaved={jest.fn()}
        />,
      );
    });

    const beforeColor = swatchBg(queryId(comp, "name-swatch-Sage")!);
    act(() => {
      queryId(comp, "name-edit-Sage")!.props.onPress();
    });
    const editingColor = swatchBg(queryId(comp, "name-swatch-Sage")!);
    expect(editingColor).toBe(beforeColor);

    act(() => comp.unmount());
  });
});

// ---------------------------------------------------------------------------
// "This is me" inline — one speaker list, not two cards
// ---------------------------------------------------------------------------

describe("SpeakerNaming — inline enrollment", () => {
  const turns: RecordingTurn[] = [
    { speaker: "Speaker A", text: "hi", start_time: 0, end_time: 2 },
    { speaker: "Speaker B", text: "hey", start_time: 2, end_time: 4 },
  ];
  const base = {
    recordingId: "rec-1",
    turns,
    manualLabels: {},
    onSaved: jest.fn(),
  };
  function enrollmentState(over: Partial<import("../src/components/SpeakerEnrollment").SpeakerEnrollmentState> = {}) {
    return {
      available: true,
      profile: { available: true, storage_enabled: true, enrolled: true, enroll_count: 5 },
      enrollingSpeaker: null,
      enrolledSpeaker: null,
      error: null,
      enroll: jest.fn().mockResolvedValue(undefined),
      ...over,
    } as import("../src/components/SpeakerEnrollment").SpeakerEnrollmentState;
  }

  it("puts 'This is me' on every speaker row and says the print will be refined", () => {
    const st = enrollmentState();
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(<SpeakerNaming {...base} enrollment={st} />);
    });
    expect(queryId(comp, "enroll-Speaker A")).not.toBeNull();
    expect(queryId(comp, "enroll-Speaker B")).not.toBeNull();
    expect(queryId(comp, "speaker-enrollment-note")).not.toBeNull();
    act(() => queryId(comp, "enroll-Speaker B")!.props.onPress());
    expect(st.enroll).toHaveBeenCalledWith("Speaker B");
  });

  it("shows no 'This is me' when the server can't do voice ID, and no second card is needed", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <SpeakerNaming {...base} enrollment={enrollmentState({ available: false, profile: null })} />,
      );
    });
    expect(queryId(comp, "enroll-Speaker A")).toBeNull();
    expect(queryId(comp, "name-edit-Speaker A")).not.toBeNull();
  });

  it("confirms an enrollment inline and surfaces enrollment errors under the list", () => {
    let comp!: renderer.ReactTestRenderer;
    act(() => {
      comp = renderer.create(
        <SpeakerNaming {...base} enrollment={enrollmentState({ enrolledSpeaker: "Speaker A" })} />,
      );
    });
    expect(queryId(comp, "speaker-enrollment-manage-hint")).not.toBeNull();
    expect(queryId(comp, "speaker-enrollment-note")).toBeNull();
    act(() => {
      comp = renderer.create(
        <SpeakerNaming {...base} enrollment={enrollmentState({ error: "Couldn’t save your voice. Please try again." })} />,
      );
    });
    expect(queryId(comp, "enroll-error")).not.toBeNull();
  });
});
