import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import SpeakerEnrollment from "../src/components/SpeakerEnrollment";
import { getVoiceProfile, enrollVoice } from "../src/api/client";
import type { RecordingTurn } from "../src/api/client";

jest.mock("../src/api/client", () => ({
  getVoiceProfile: jest.fn(),
  enrollVoice: jest.fn(),
}));
const mockProfile = getVoiceProfile as jest.Mock;
const mockEnroll = enrollVoice as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

const turns: RecordingTurn[] = [
  { speaker: "SPEAKER_00", text: "You never listen.", start_time: 0, end_time: 3 },
  { speaker: "SPEAKER_01", text: "That's not fair.", start_time: 3, end_time: 6 },
];

// The analysis resolved a friendly label for one speaker but not the other.
const speakerLabels = {
  SPEAKER_00: { display_label: "Deeper voice", label_source: "voice" },
};

beforeEach(() => {
  mockProfile.mockReset();
  mockEnroll.mockReset();
});

describe("SpeakerEnrollment", () => {
  it("renders nothing until voice ID is confirmed available", async () => {
    mockProfile.mockResolvedValueOnce({
      available: false,
      storage_enabled: false,
      enrolled: false,
      enroll_count: 0,
    });
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <SpeakerEnrollment recordingId="r1" turns={turns} />,
      );
    });
    await act(async () => {});
    expect(queryId(comp, "speaker-enrollment")).toBeNull();
    act(() => comp.unmount());
  });

  it("shows the friendly display label instead of the raw diarization id", async () => {
    mockProfile.mockResolvedValueOnce({
      available: true,
      storage_enabled: true,
      enrolled: false,
      enroll_count: 0,
    });
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <SpeakerEnrollment
          recordingId="r1"
          turns={turns}
          speakerLabels={speakerLabels}
        />,
      );
    });
    await act(async () => {});

    const json = JSON.stringify(comp.toJSON());
    // The labeled speaker reads as "Deeper voice", not "SPEAKER_00".
    expect(json).toContain("Deeper voice");
    // The unlabeled speaker still falls back to its raw id (honest, unchanged).
    expect(json).toContain("SPEAKER_01");
    act(() => comp.unmount());
  });

  it("enrolls using the raw speaker id (not the display label) and confirms", async () => {
    mockProfile.mockResolvedValueOnce({
      available: true,
      storage_enabled: true,
      enrolled: false,
      enroll_count: 0,
    });
    mockEnroll.mockResolvedValueOnce({ enroll_count: 1 });
    let comp!: renderer.ReactTestRenderer;
    await act(async () => {
      comp = renderer.create(
        <SpeakerEnrollment
          recordingId="r1"
          turns={turns}
          speakerLabels={speakerLabels}
        />,
      );
    });
    await act(async () => {});

    // The tap target keys off the raw id even though the label reads friendly.
    const btn = queryId(comp, "enroll-SPEAKER_00");
    expect(btn).toBeTruthy();
    await act(async () => {
      btn!.props.onPress();
    });
    await act(async () => {});

    expect(mockEnroll).toHaveBeenCalledWith("r1", "SPEAKER_00");
    expect(JSON.stringify(comp.toJSON())).toContain("Voice saved");
    act(() => comp.unmount());
  });
});
