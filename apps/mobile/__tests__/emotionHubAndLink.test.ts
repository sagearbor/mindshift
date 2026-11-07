import { DEFAULT_USER_SETTINGS, MODEL_FILES } from "../src/emotion/config";
import { EmotionDevHub, type EmotionRuntime, type RuntimeBuild } from "../src/emotion/devHub";
import { parseSettings } from "../src/emotion/devPrefs";
import { OnnxEmotionEngine } from "../src/emotion/engine";
import { N_FRAMES, N_MELS } from "../src/emotion/logMel";
import { androidExternalEmotionDir, modelSearchDirs, resolveBoth, resolveEmotionModel } from "../src/emotion/modelFiles";
import { ServerWatchLink } from "../src/emotion/watchLink";
import type { OnnxSession } from "../src/live/ort";
import { describeStatus } from "../src/components/EmotionDevPanel";

describe("model sideload resolution", () => {
  const dirs = modelSearchDirs({ documentDirUri: "file:///doc/", androidExternalDirUri: androidExternalEmotionDir("com.x.app") });
  const files: Record<string, number> = {};
  const fsLike = { stat: (u: string) => ({ exists: u in files, size: files[u] ?? 0 }) };

  it("search order: external app folder first, then documents/emotion-models", () => {
    expect(dirs).toEqual(["file:///storage/emulated/0/Android/data/com.x.app/files/emotion", "file:///doc/emotion-models"]);
  });
  it("reports 'missing' with the folders it looked in", () => {
    const r = resolveEmotionModel("tiny", fsLike, dirs);
    expect(r.status).toBe("missing");
    expect((r as { reason: string }).reason).toContain(MODEL_FILES.tiny);
  });
  it("uses a file in place (strips file://) and rejects a truncated copy", () => {
    files[`${dirs[1]}/${MODEL_FILES.tiny}`] = 9_900_000;
    files[`${dirs[1]}/${MODEL_FILES.small}`] = 1234;
    const r = resolveBoth(fsLike, dirs);
    expect(r.tiny).toMatchObject({ status: "ready", path: "/doc/emotion-models/" + MODEL_FILES.tiny });
    expect(r.small.status).toBe("missing");
    expect((r.small as { reason: string }).reason).toMatch(/truncated/);
  });
  it("an unreadable folder is skipped, not fatal", () => {
    const bad = { stat: (u: string) => { if (u.includes("Android/data")) throw new Error("EACCES"); return { exists: false, size: 0 }; } };
    expect(resolveEmotionModel("tiny", bad, dirs).status).toBe("missing");
  });
});

describe("persisted settings", () => {
  it("default is OFF and unreadable input falls back to it", () => {
    expect(DEFAULT_USER_SETTINGS.enabled).toBe(false);
    expect(parseSettings(null)).toEqual(DEFAULT_USER_SETTINGS);
    expect(parseSettings("not json")).toEqual(DEFAULT_USER_SETTINGS);
    expect(parseSettings(JSON.stringify({ enabled: true, maxNudgesPerHour: 999 })).maxNudgesPerHour).toBe(DEFAULT_USER_SETTINGS.maxNudgesPerHour);
    expect(parseSettings(JSON.stringify({ enabled: true, maxNudgesPerHour: 7 }))).toMatchObject({ enabled: true, maxNudgesPerHour: 7 });
  });
});

describe("OnnxEmotionEngine", () => {
  const mel = { data: new Float32Array(N_MELS * N_FRAMES), validFrames: 135 };
  const session = (out: Float32Array): OnnxSession & { seen?: unknown } => {
    const s: OnnxSession & { seen?: unknown } = {
      inputNames: ["log_mel", "valid_frames"],
      outputNames: ["probs"],
      async run(feeds) {
        s.seen = feeds;
        return { probs: { type: "float32", data: out, dims: [1, 6] } };
      },
      async release() {},
    };
    return s;
  };
  it("feeds log_mel [1,80,600] float32 + valid_frames int64 [1] and returns the probabilities", async () => {
    const s = session(Float32Array.from([0.1, 0.1, 0.1, 0.1, 0.5, 0.1]));
    const p = await new OnnxEmotionEngine("tiny", s).classify(mel);
    expect(p[4]).toBeCloseTo(0.5);
    const feeds = s.seen as Record<string, { type: string; dims: number[]; data: BigInt64Array }>;
    expect(feeds.log_mel.dims).toEqual([1, 80, 600]);
    expect(feeds.valid_frames).toMatchObject({ type: "int64", dims: [1] });
    expect(feeds.valid_frames.data[0]).toBe(135n);
  });
  it("refuses a model output that is not a probability vector (no fake success)", async () => {
    await expect(new OnnxEmotionEngine("tiny", session(Float32Array.from([2, 2, 2, 2, 2, 2]))).classify(mel)).rejects.toThrow(/probability/);
  });
});

describe("ServerWatchLink", () => {
  const mk = (reply: (url: string, init?: { method?: string; body?: string }) => { ok: boolean; status: number; body: unknown }) => {
    const calls: Array<{ url: string; init?: { method?: string; body?: string } }> = [];
    const link = new ServerWatchLink("https://api.test", async () => ({ Authorization: "Bearer t" }), async (url, init) => {
      calls.push({ url, init });
      const r = reply(url, init);
      return { ok: r.ok, status: r.status, json: async () => r.body };
    });
    return { link, calls };
  };
  const state = { v: 1 as const, id: "a", t: 1, label: "anger", prob: 0.7, nudge: true, paused: false, suggestion: null };

  it("POSTs the state and reports delivered only when the server says a watch got it", async () => {
    const a = mk(() => ({ ok: true, status: 200, body: { delivered: true } }));
    expect(await a.link.sendState(state)).toBe(true);
    expect(a.calls[0].url).toBe("https://api.test/emotion/event");
    expect(JSON.parse(a.calls[0].init!.body!)).toMatchObject({ label: "anger", nudge: true });
    expect(await mk(() => ({ ok: true, status: 200, body: { delivered: false } })).link.sendState(state)).toBe(false);
    expect(await mk(() => ({ ok: false, status: 503, body: {} })).link.sendState(state)).toBe(false);
  });
  it("never throws on a network failure", async () => {
    const link = new ServerWatchLink("https://api.test", async () => ({}), async () => { throw new Error("offline"); });
    expect(await link.sendState(state)).toBe(false);
    expect(await link.pollControls(0)).toEqual([]);
  });
  it("parses controls, drops malformed ones, sorts by seq", async () => {
    const { link, calls } = mk(() => ({
      ok: true,
      status: 200,
      body: { items: [
        { seq: 3, action: "resume", t: 3 },
        { seq: 2, action: "flag", flag: "bad", nudge_id: "n1", t: 2 },
        { seq: 4, action: "explode" },
        { action: "pause" },
        { seq: 5, action: "flag", flag: "nonsense", t: 5 },
      ] },
    }));
    const out = await link.pollControls(1);
    expect(calls[0].url).toBe("https://api.test/emotion/ctl?after=1");
    expect(out.map((c) => c.seq)).toEqual([2, 3, 5]);
    expect(out[0]).toMatchObject({ action: "flag", flag: "bad", nudgeId: "n1" });
    expect(out[2].flag).toBeUndefined();
  });
});

describe("EmotionDevHub", () => {
  const fakeRuntime = (): EmotionRuntime & { pushed: number; stopped: number; disposed: number; maxPerHour: number } => {
    const rt = {
      pushed: 0, stopped: 0, disposed: 0, maxPerHour: -1,
      controller: {
        pushPcm: () => { rt.pushed++; },
        stop: () => { rt.stopped++; },
        setMaxNudgesPerHour: (n: number) => { rt.maxPerHour = n; },
        snapshot: () => ({ paused: false, windows: 0, skippedSilent: 0, errors: 0, lastLabel: null, lastProb: null, lastSmallLabel: null, nudges: 0, lastNudgeId: null, lastError: null }),
        setPaused: () => {},
        flag: () => "id",
      },
      store: { usage: () => ({ clips: 0, clipBytes: 0, logBytes: 0 }), deleteAll: jest.fn(), exportLogText: () => "" },
      models: { tiny: true, small: false, smallReason: "small not found" },
      dispose: async () => { rt.disposed++; },
    };
    return rt as never;
  };
  const tick = () => new Promise((r) => setImmediate(r));

  it("costs nothing while OFF: no build, audio dropped", async () => {
    const build = jest.fn();
    const hub = new EmotionDevHub(build);
    hub.pushPcm(new Int16Array(100));
    expect(build).not.toHaveBeenCalled();
    expect(hub.status().state).toBe("off");
  });

  it("builds lazily on the first frame once enabled, then forwards audio", async () => {
    const rt = fakeRuntime();
    const build = jest.fn(async (): Promise<RuntimeBuild> => ({ ok: true, runtime: rt }));
    const hub = new EmotionDevHub(build);
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: true, maxNudgesPerHour: 6 });
    expect(hub.status().state).toBe("waiting-for-audio");
    hub.pushPcm(new Int16Array(10));
    hub.pushPcm(new Int16Array(10)); // while building: no second build
    await tick();
    expect(build).toHaveBeenCalledTimes(1);
    hub.pushPcm(new Int16Array(10));
    expect(rt.pushed).toBe(1);
    expect(rt.maxPerHour).toBe(6);
    expect(hub.status().state).toBe("running");
  });

  it("'models missing' is a status with a reason, not an error, and stops retrying until asked", async () => {
    const build = jest.fn(async (): Promise<RuntimeBuild> => ({ ok: false, code: "models-missing", message: "Models missing — x" }));
    const hub = new EmotionDevHub(build);
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: true });
    hub.pushPcm(new Int16Array(10));
    await tick();
    expect(hub.status()).toEqual({ state: "models-missing", message: "Models missing — x" });
    hub.pushPcm(new Int16Array(10));
    await tick();
    expect(build).toHaveBeenCalledTimes(1);
    hub.retry();
    hub.pushPcm(new Int16Array(10));
    await tick();
    expect(build).toHaveBeenCalledTimes(2);
  });

  it("disabling tears the runtime down (clips flushed, sessions released)", async () => {
    const rt = fakeRuntime();
    const hub = new EmotionDevHub(async () => ({ ok: true, runtime: rt }));
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: true });
    hub.pushPcm(new Int16Array(10));
    await tick();
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: false });
    expect(rt.stopped).toBe(1);
    expect(rt.disposed).toBe(1);
    hub.pushPcm(new Int16Array(10));
    expect(rt.pushed).toBe(0);
  });

  it("a build that finishes after the mode was switched off is disposed, not used", async () => {
    const rt = fakeRuntime();
    let release!: (r: RuntimeBuild) => void;
    const hub = new EmotionDevHub(() => new Promise<RuntimeBuild>((r) => (release = r)));
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: true });
    hub.pushPcm(new Int16Array(10));
    await hub.configure({ ...DEFAULT_USER_SETTINGS, enabled: false });
    release({ ok: true, runtime: rt });
    await tick();
    expect(rt.disposed).toBe(1);
  });

  it("delete-all and usage work without a runtime (mode off / models missing)", () => {
    const store = { usage: () => ({ clips: 2, clipBytes: 5, logBytes: 1 }), deleteAll: jest.fn(), exportLogText: () => "log" };
    const hub = new EmotionDevHub(async () => ({ ok: false, code: "error", message: "x" }), () => store as never);
    expect(hub.usage()).toEqual({ clips: 2, clipBytes: 5, logBytes: 1 });
    hub.deleteAll();
    expect(store.deleteAll).toHaveBeenCalled();
    expect(hub.exportLogText()).toBe("log");
  });

  it("status text is honest about SMALL missing and about pause", () => {
    const snap = { paused: true, windows: 3, skippedSilent: 0, errors: 0, lastLabel: "anger" as const, lastProb: 0.6, lastSmallLabel: null, nudges: 1, lastNudgeId: null, lastError: null };
    const t = describeStatus({ state: "running", models: { tiny: true, small: false, smallReason: "small not found" }, snapshot: snap });
    expect(t).toMatch(/PAUSED/);
    expect(t).toMatch(/SMALL off \(small not found\)/);
    expect(describeStatus({ state: "models-missing", message: "Models missing — nope" })).toContain("Models missing");
  });
});
