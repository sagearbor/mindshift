import React from "react";
import renderer, { act } from "react-test-renderer";
import LiveTranscript from "../src/components/LiveTranscript";
import type { TranscriptEntry } from "../src/hooks/useAudioStream";

/**
 * Unmount every tree this file creates.
 *
 * react-test-renderer keeps a mounted tree scheduled, and a state update that
 * lands AFTER Jest tears the environment down throws "You are trying to
 * `import` a file after the Jest environment has been torn down" — which is
 * not a test failure but a WORKER CRASH, taking every unrelated suite sharing
 * that worker with it. That is why running the whole suite at once used to
 * fail a handful of random files that each passed on their own.
 */
const __trees: renderer.ReactTestRenderer[] = [];
function track<T extends renderer.ReactTestRenderer>(t: T): T {
  __trees.push(t);
  return t;
}
afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // A tree a test already unmounted, or one whose teardown throws, must
        // not fail the test that otherwise passed.
      }
    }
  });
});


describe("LiveTranscript", () => {
  it("renders empty state", () => {
    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<LiveTranscript entries={[]} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
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

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<LiveTranscript entries={entries} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });

  it("highlights the most recent entry", () => {
    const entries: TranscriptEntry[] = [
      { speaker: "Speaker A", text: "First message.", timestamp: 1000 },
      { speaker: "Speaker B", text: "Second message.", timestamp: 2000 },
      { speaker: "Speaker A", text: "Latest message.", timestamp: 3000 },
    ];

    let component: renderer.ReactTestRenderer;
    act(() => {
      component = track(renderer.create(<LiveTranscript entries={entries} />));
    });
    expect(component!.toJSON()).toMatchSnapshot();
  });
});
