import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";

// First-render transform of react-native + expo modules can exceed the global
// 30s allowance on cold CI/sandbox workers (the suite's first component test
// pays it all); the work itself is fast. Same de-flake rationale as jest-setup.
jest.setTimeout(120000);
import VoiceTrainingFlow, {
  MIN_TAKE_MS,
  PHRASES,
  type VoiceTrainingDeps,
} from "../src/components/VoiceTrainingFlow";
import type { PcmFrame, PcmSource } from "../src/recorder/pcmSource";

/** Controllable PcmSource: the test pushes frames through `emit`. */
class FakeSource implements PcmSource {
  onFrame: ((f: PcmFrame) => void) | null = null;
  started = false;
  failStart = false;

  async start(onFrame: (f: PcmFrame) => void): Promise<void> {
    if (this.failStart) throw new Error("mic busy");
    this.onFrame = onFrame;
    this.started = true;
  }

  stop(): void {
    this.started = false;
  }

  isCapturing(): boolean {
    return this.started;
  }

  emit(seconds: number, sampleRate = 16000): void {
    this.onFrame?.({
      samples: new Int16Array(Math.round(seconds * sampleRate)),
      sampleRate,
    });
  }
}

function makeDeps(overrides: Partial<VoiceTrainingDeps> = {}) {
  const sources: FakeSource[] = [];
  const saved: Uint8Array[] = [];
  const deps: VoiceTrainingDeps = {
    makeSource: () => {
      const s = new FakeSource();
      sources.push(s);
      return s;
    },
    saveWav: jest.fn(async (bytes: Uint8Array) => {
      saved.push(bytes);
      return "file:///cache/guided-enrollment.wav";
    }),
    enroll: jest.fn(async () => ({
      enrolled: true,
      enroll_count: 3,
      dim: 192,
      updated_at: "2026-08-15T10:00:00+00:00",
      stored: "a numeric voice signature (192 numbers), not your audio",
    })),
    getPermission: jest.fn(async () => true),
    requestPermission: jest.fn(async () => true),
    ...overrides,
  };
  return { deps, sources, saved };
}

function queryId(
  comp: renderer.ReactTestRenderer,
  id: string,
): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id);
  return found.length > 0 ? found[0] : null;
}

function textOf(node: ReactTestInstance): string {
  return node
    .findAll((n) => typeof n.type === "string")
    .flatMap((n) => n.children)
    .filter((c): c is string => typeof c === "string")
    .join("");
}

async function render(
  deps: VoiceTrainingDeps,
  handlers = { onDone: jest.fn(), onCancel: jest.fn() },
) {
  let comp!: renderer.ReactTestRenderer;
  await act(async () => {
    comp = renderer.create(<VoiceTrainingFlow {...handlers} deps={deps} />);
  });
  return { comp, handlers };
}

/** Record ~`seconds` of audio for the CURRENT phrase and stop. */
async function recordPhrase(
  comp: renderer.ReactTestRenderer,
  sources: FakeSource[],
  seconds = 3,
  sampleRate = 16000,
) {
  await act(async () => queryId(comp, "vt-record")!.props.onPress());
  const source = sources[sources.length - 1];
  await act(async () => {
    source.emit(seconds, sampleRate);
  });
  await act(async () => queryId(comp, "vt-stop")!.props.onPress());
}

describe("VoiceTrainingFlow — phrase progression", () => {
  it("ships exactly four short prompted phrases", () => {
    expect(PHRASES).toHaveLength(4);
    for (const p of PHRASES) {
      expect(typeof p).toBe("string");
      expect(p.length).toBeGreaterThan(20);
    }
  });

  it("shows phrase 1 of 4, records, and advances phrase by phrase", async () => {
    const { deps, sources } = makeDeps();
    const { comp } = await render(deps);

    expect(textOf(queryId(comp, "vt-progress")!)).toContain("1 of 4");
    expect(textOf(queryId(comp, "vt-phrase")!)).toContain(PHRASES[0]);
    expect(queryId(comp, "vt-stop")).toBeNull();

    await act(async () => queryId(comp, "vt-record")!.props.onPress());
    expect(queryId(comp, "vt-stop")).toBeTruthy();
    expect(queryId(comp, "vt-record")).toBeNull();
    expect(sources[0].started).toBe(true);

    await act(async () => {
      sources[0].emit(3);
    });
    await act(async () => queryId(comp, "vt-stop")!.props.onPress());
    // Mic released between phrases; on to phrase 2.
    expect(sources[0].started).toBe(false);
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("2 of 4");
    expect(textOf(queryId(comp, "vt-phrase")!)).toContain(PHRASES[1]);

    act(() => comp.unmount());
  });

  it("keeps the phrase and says so when a take is too short to use", async () => {
    const { deps, sources } = makeDeps();
    const { comp } = await render(deps);

    // Stop with almost nothing captured (< MIN_TAKE_MS).
    await recordPhrase(comp, sources, MIN_TAKE_MS / 1000 / 10);
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("1 of 4");
    expect(queryId(comp, "vt-take-note")).toBeTruthy();
    expect(textOf(queryId(comp, "vt-take-note")!)).toMatch(/didn.t hear/i);

    // A proper take clears the note and advances.
    await recordPhrase(comp, sources, 3);
    expect(queryId(comp, "vt-take-note")).toBeNull();
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("2 of 4");

    act(() => comp.unmount());
  });

  it("surfaces a mic start failure honestly and stays recordable", async () => {
    const { deps, sources } = makeDeps({
      makeSource: () => {
        const s = new FakeSource();
        s.failStart = true;
        return s;
      },
    });
    const { comp } = await render(deps);
    void sources;

    await act(async () => queryId(comp, "vt-record")!.props.onPress());
    expect(queryId(comp, "vt-take-note")).toBeTruthy();
    expect(textOf(queryId(comp, "vt-take-note")!)).toMatch(/microphone/i);
    expect(queryId(comp, "vt-record")).toBeTruthy(); // can try again

    act(() => comp.unmount());
  });
});

describe("VoiceTrainingFlow — upload & outcomes", () => {
  it("after the 4th phrase uploads ONE wav of all takes and reports the count", async () => {
    const { deps, sources, saved } = makeDeps();
    const { comp, handlers } = await render(deps);

    for (let i = 0; i < 4; i++) {
      await recordPhrase(comp, sources, 3);
    }

    // One upload of one wav containing all four 3s takes.
    expect(deps.enroll).toHaveBeenCalledTimes(1);
    expect(deps.enroll).toHaveBeenCalledWith(
      "file:///cache/guided-enrollment.wav",
      "guided-enrollment.wav",
    );
    expect(saved).toHaveLength(1);
    const wav = saved[0];
    const v = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getUint32(40, true)).toBe(4 * 3 * 16000 * 2);

    // Success is stated with the server's real count, then handed back.
    const success = queryId(comp, "vt-success")!;
    expect(textOf(success)).toContain("3 sample");
    await act(async () => queryId(comp, "vt-success-done")!.props.onPress());
    expect(handlers.onDone).toHaveBeenCalledWith(3);

    act(() => comp.unmount());
  });

  it("shows the server's honest 422 detail and can retry the upload", async () => {
    const enroll = jest
      .fn()
      .mockRejectedValueOnce(
        Object.assign(new Error("not enough speech in the clip to enroll"), {
          status: 422,
        }),
      )
      .mockResolvedValueOnce({
        enrolled: true,
        enroll_count: 1,
        dim: 192,
        updated_at: "t",
        stored: "s",
      });
    const { deps, sources } = makeDeps({ enroll });
    const { comp } = await render(deps);

    for (let i = 0; i < 4; i++) {
      await recordPhrase(comp, sources, 3);
    }
    expect(queryId(comp, "vt-error")).toBeTruthy();
    expect(textOf(queryId(comp, "vt-error")!)).toContain("not enough speech");

    await act(async () => queryId(comp, "vt-retry-upload")!.props.onPress());
    expect(enroll).toHaveBeenCalledTimes(2);
    expect(queryId(comp, "vt-success")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("reports a network failure honestly with a retry and a start-over", async () => {
    const enroll = jest.fn().mockRejectedValue(new Error("Network request failed"));
    const { deps, sources } = makeDeps({ enroll });
    const { comp } = await render(deps);

    for (let i = 0; i < 4; i++) {
      await recordPhrase(comp, sources, 3);
    }
    expect(textOf(queryId(comp, "vt-error")!)).toMatch(/couldn.t upload/i);
    expect(queryId(comp, "vt-retry-upload")).toBeTruthy();

    // Start over returns to phrase 1 with the takes discarded.
    await act(async () => queryId(comp, "vt-start-over")!.props.onPress());
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("1 of 4");

    act(() => comp.unmount());
  });
});

describe("VoiceTrainingFlow — permission & cancel", () => {
  it("gates on mic permission and proceeds after a grant", async () => {
    const getPermission = jest.fn(async () => false);
    const requestPermission = jest.fn(async () => true);
    const { deps } = makeDeps({ getPermission, requestPermission });
    const { comp } = await render(deps);

    expect(queryId(comp, "vt-permission-gate")).toBeTruthy();
    expect(queryId(comp, "vt-record")).toBeNull();

    await act(async () => queryId(comp, "vt-grant-mic")!.props.onPress());
    expect(requestPermission).toHaveBeenCalled();
    expect(queryId(comp, "vt-permission-gate")).toBeNull();
    expect(queryId(comp, "vt-record")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("stays gated honestly when the grant is denied", async () => {
    const { deps } = makeDeps({
      getPermission: jest.fn(async () => false),
      requestPermission: jest.fn(async () => false),
    });
    const { comp } = await render(deps);

    await act(async () => queryId(comp, "vt-grant-mic")!.props.onPress());
    expect(queryId(comp, "vt-permission-gate")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("cancel stops any live capture and hands control back", async () => {
    const { deps, sources } = makeDeps();
    const { comp, handlers } = await render(deps);

    await act(async () => queryId(comp, "vt-record")!.props.onPress());
    expect(sources[0].started).toBe(true);
    await act(async () => queryId(comp, "vt-cancel")!.props.onPress());
    expect(sources[0].started).toBe(false);
    expect(handlers.onCancel).toHaveBeenCalledTimes(1);

    act(() => comp.unmount());
  });
});
