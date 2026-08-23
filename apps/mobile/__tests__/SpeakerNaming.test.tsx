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
