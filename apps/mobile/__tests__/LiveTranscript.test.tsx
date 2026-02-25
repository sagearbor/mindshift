import React from "react";
import renderer, { act } from "react-test-renderer";
import LiveTranscript from "../src/components/LiveTranscript";
import type { TranscriptEntry } from "../src/hooks/useAudioStream";

describe("LiveTranscript", () => {
  it("renders empty state", () => {
    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<LiveTranscript entries={[]} />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("renders transcript entries with speaker colors", () => {
    const entries: TranscriptEntry[] = [
      { speaker: "Speaker A", text: "I feel frustrated.", timestamp: 1000 },
      {
        speaker: "Speaker B",
        text: "I understand, tell me more.",
        timestamp: 2000,
      },
    ];

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<LiveTranscript entries={entries} />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });

  it("highlights the most recent entry", () => {
    const entries: TranscriptEntry[] = [
      { speaker: "Speaker A", text: "First message.", timestamp: 1000 },
      { speaker: "Speaker B", text: "Second message.", timestamp: 2000 },
      { speaker: "Speaker A", text: "Latest message.", timestamp: 3000 },
    ];

    let tree: renderer.ReactTestRendererJSON | null = null;
    act(() => {
      tree = renderer.create(<LiveTranscript entries={entries} />).toJSON();
    });
    expect(tree).toMatchSnapshot();
  });
});
