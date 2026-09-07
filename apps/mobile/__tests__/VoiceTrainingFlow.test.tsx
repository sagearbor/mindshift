import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";

// First-render transform of react-native + expo modules can exceed the global
// 30s allowance on cold CI/sandbox workers (the suite's first component test
// pays it all); the work itself is fast. Same de-flake rationale as jest-setup.
jest.setTimeout(120000);
import VoiceTrainingFlow, {
  MIN_TAKE_MS,
  PHRASES,
  PROMPTS,
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
  it("ships four ordinary prompts and TWO raised ones, last", () => {
    expect(PROMPTS).toHaveLength(6);
    for (const p of PROMPTS) {
      expect(typeof p.text).toBe("string");
      expect(p.text.length).toBeGreaterThan(20);
    }
    // The raised takes are last on purpose: someone who stops before them
    // still ends up with a complete ordinary print, exactly as before.
    expect(PROMPTS.slice(0, 4).every((p) => p.register === "normal")).toBe(true);
    const raised = PROMPTS.filter((p) => p.register === "raised");
    // TWO, not one. The server needs 3 s of ACTUAL speech per upload
    // (speaker_id.MIN_ENROLL_SECONDS) and the raised takes are their own
    // upload — one short shouted line gave 1.3 s on the owner's phone and was
    // rejected outright. Shouted speech is also FASTER than ordinary speech,
    // so the lines have to be long as well as plural.
    expect(raised).toHaveLength(2);
    for (const p of raised) {
      expect(p.text.length).toBeGreaterThan(60);
      // Each has to SAY it is different, or people read them normally and the
      // second prototype is just a duplicate of the first.
      expect(p.instruction).toMatch(/loud/i);
    }
  });

  it("shows phrase 1 of 6, records, and advances phrase by phrase", async () => {
    const { deps, sources } = makeDeps();
    const { comp } = await render(deps);

    expect(textOf(queryId(comp, "vt-progress")!)).toContain("1 of 6");
    // An ordinary prompt carries no shouting instruction.
    expect(queryId(comp, "vt-instruction")).toBeNull();
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
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("2 of 6");
    expect(textOf(queryId(comp, "vt-phrase")!)).toContain(PHRASES[1]);

    act(() => comp.unmount());
  });

  it("keeps the phrase and says so when a take is too short to use", async () => {
    const { deps, sources } = makeDeps();
    const { comp } = await render(deps);

    // Stop with almost nothing captured (< MIN_TAKE_MS).
    await recordPhrase(comp, sources, MIN_TAKE_MS / 1000 / 10);
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("1 of 6");
    expect(queryId(comp, "vt-take-note")).toBeTruthy();
    expect(textOf(queryId(comp, "vt-take-note")!)).toMatch(/didn.t hear/i);

    // A proper take clears the note and advances.
    await recordPhrase(comp, sources, 3);
    expect(queryId(comp, "vt-take-note")).toBeNull();
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("2 of 6");

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
  it("uploads the ordinary takes as ONE wav and the raised take SEPARATELY", async () => {
    const { deps, sources, saved } = makeDeps();
    const { comp, handlers } = await render(deps);

    for (let i = 0; i < 6; i++) {
      await recordPhrase(comp, sources, 3);
    }

    // TWO uploads, not one. The raised clip must never be concatenated with
    // the ordinary ones — the server keeps it as a separate prototype, and
    // averaging the two would give a print that matches neither voice.
    expect(deps.enroll).toHaveBeenCalledTimes(2);
    expect(deps.enroll).toHaveBeenNthCalledWith(
      1,
      "file:///cache/guided-enrollment.wav",
      "guided-enrollment.wav",
      undefined,
      undefined,
    );
    expect(deps.enroll).toHaveBeenNthCalledWith(
      2,
      // The fake saveWav returns one fixed path; what matters is the NAME and
      // the register the raised clip is uploaded under.
      "file:///cache/guided-enrollment.wav",
      "guided-enrollment-raised.wav",
      undefined,
      "raised",
    );
    expect(saved).toHaveLength(2);
    const v = new DataView(saved[0].buffer, saved[0].byteOffset, saved[0].byteLength);
    expect(v.getUint32(24, true)).toBe(16000);
    expect(v.getUint32(40, true)).toBe(4 * 3 * 16000 * 2);
    const raised = new DataView(saved[1].buffer, saved[1].byteOffset, saved[1].byteLength);
    expect(raised.getUint32(40, true)).toBe(2 * 3 * 16000 * 2);

    // Success is stated with the server's real count, then handed back.
    const success = queryId(comp, "vt-success")!;
    expect(textOf(success)).toContain("3 sample");
    // The owner's exact bug this whole fix addresses: guided enrollment only
    // ever writes the voiceprint, never touches a stored recording — the
    // success screen must point at Growth's catch-up option for recordings
    // made before today (there's no "This is me" tap to guess at here).
    expect(textOf(success)).toContain("Catch up my past recordings");
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
      .mockResolvedValue({
        enrolled: true,
        enroll_count: 1,
        dim: 192,
        updated_at: "t",
        stored: "s",
      });
    const { deps, sources } = makeDeps({ enroll });
    const { comp } = await render(deps);

    for (let i = 0; i < 6; i++) {
      await recordPhrase(comp, sources, 3);
    }
    expect(queryId(comp, "vt-error")).toBeTruthy();
    expect(textOf(queryId(comp, "vt-error")!)).toContain("not enough speech");

    // Retry re-sends the ordinary group (which failed) AND then the raised
    // one, which had not been attempted yet: 1 failure + 2 = 3.
    await act(async () => queryId(comp, "vt-retry-upload")!.props.onPress());
    expect(enroll).toHaveBeenCalledTimes(3);
    expect(queryId(comp, "vt-success")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("reports a network failure honestly with a retry and a start-over", async () => {
    const enroll = jest.fn().mockRejectedValue(new Error("Network request failed"));
    const { deps, sources } = makeDeps({ enroll });
    const { comp } = await render(deps);

    for (let i = 0; i < 6; i++) {
      await recordPhrase(comp, sources, 3);
    }
    expect(textOf(queryId(comp, "vt-error")!)).toMatch(/couldn.t upload/i);
    expect(queryId(comp, "vt-retry-upload")).toBeTruthy();

    // Start over returns to phrase 1 with the takes discarded.
    await act(async () => queryId(comp, "vt-start-over")!.props.onPress());
    expect(textOf(queryId(comp, "vt-progress")!)).toContain("1 of 6");

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
  it("a retry never re-uploads a group that already landed", async () => {
    // The ordinary clip succeeds, the raised one fails. Retrying must send ONLY
    // the raised clip — re-sending the ordinary one would store it twice.
    const enroll = jest
      .fn()
      .mockResolvedValueOnce({ enrolled: true, enroll_count: 1, dim: 192, updated_at: "t", stored: "s" })
      .mockRejectedValueOnce(Object.assign(new Error("network"), { status: 0 }))
      .mockResolvedValue({ enrolled: true, enroll_count: 2, dim: 192, updated_at: "t", stored: "s" });
    const { deps, sources } = makeDeps({ enroll });
    const { comp } = await render(deps);

    for (let i = 0; i < 6; i++) {
      await recordPhrase(comp, sources, 3);
    }
    expect(queryId(comp, "vt-error")).toBeTruthy();
    expect(enroll).toHaveBeenCalledTimes(2);

    await act(async () => queryId(comp, "vt-retry-upload")!.props.onPress());
    expect(enroll).toHaveBeenCalledTimes(3);
    expect(enroll.mock.calls[2][3]).toBe("raised");
    expect(queryId(comp, "vt-success")).toBeTruthy();

    act(() => comp.unmount());
  });

  it("every register's prompts hold enough speech to clear the server's floor", () => {
    // The bug this pins (2026-09-07): the raised take is its OWN upload, and
    // the server needs MIN_ENROLL_SECONDS = 3 s of ACTUAL speech per upload.
    // One short shouted line measured 1.3 s on the owner's phone and was
    // rejected — the enrollment simply failed at the last step, after four
    // phrases had already been read.
    //
    // Estimated at a DELIBERATELY pessimistic 4.5 words/second: shouted speech
    // is faster than ordinary speech, and the floor counts voiced frames, not
    // clip length. Ordinary speech runs nearer 2.5-3 w/s, so this is roughly a
    // 1.7x safety margin on the normal group too.
    const FAST_WORDS_PER_SEC = 4.5;
    const MIN_ENROLL_SECONDS = 3;
    for (const register of ["normal", "raised"] as const) {
      const words = PROMPTS.filter((p) => p.register === register)
        .reduce((n, p) => n + p.text.trim().split(/\s+/).length, 0);
      const seconds = words / FAST_WORDS_PER_SEC;
      expect({ register, ok: seconds >= MIN_ENROLL_SECONDS * 1.5 }).toEqual({ register, ok: true });
    }
  });
});
