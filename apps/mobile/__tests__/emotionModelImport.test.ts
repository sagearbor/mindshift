import { checkModelSize, checkStorage, describePrerequisites, validatePickedModel } from "../src/emotion/modelImport";

const MB = 1024 * 1024;

describe("validatePickedModel", () => {
  it("names the copy with the expected file name whatever the picked name was", () => {
    expect(validatePickedModel("tiny", { name: "emotion-int8.onnx", size: 10 * MB })).toEqual({ ok: true, destName: "emotion-tiny-int8.onnx" });
    expect(validatePickedModel("small", { name: "emotion-int8.onnx", size: 95 * MB })).toEqual({ ok: true, destName: "emotion-small-int8.onnx" });
  });
  it("rejects non-onnx files", () => {
    expect(validatePickedModel("tiny", { name: "notes.txt", size: 10 * MB }).ok).toBe(false);
  });
  it("rejects truncated files, and SMALL passed as TINY", () => {
    expect(validatePickedModel("tiny", { name: "a.onnx", size: 1000 }).ok).toBe(false);
    expect(validatePickedModel("small", { name: "a.onnx", size: 10 * MB }).ok).toBe(false);
    expect(validatePickedModel("tiny", { name: "a.onnx", size: 95 * MB }).ok).toBe(false);
  });
  it("rejects a name that points at the other model", () => {
    expect(validatePickedModel("tiny", { name: "emotion-small-int8.onnx", size: 10 * MB }).ok).toBe(false);
    expect(validatePickedModel("small", { name: "emotion-tiny-int8.onnx", size: 95 * MB }).ok).toBe(false);
  });
  it("defers the size check when the picker reports no size", () => {
    expect(validatePickedModel("tiny", { name: "x.onnx", size: null }).ok).toBe(true);
    expect(checkModelSize("tiny", 100)).toMatch(/Too small/);
  });
});

describe("checkStorage", () => {
  it("fails when free space is below size + headroom", () => {
    expect(checkStorage(95 * MB, 100 * MB)).toMatch(/Not enough free storage/);
    expect(checkStorage(95 * MB, 500 * MB)).toBeNull();
  });
  it("does not block when free space is unknown", () => {
    expect(checkStorage(95 * MB, null)).toBeNull();
  });
});

describe("describePrerequisites", () => {
  it("names every missing prerequisite", () => {
    const lines = describePrerequisites({ enabled: false, tiny: { status: "missing", reason: "nf" }, small: { status: "missing", reason: "nf" }, liveSessionActive: false }).join("\n");
    expect(lines).toMatch(/Developer mode: OFF/);
    expect(lines).toMatch(/TINY model: MISSING.*required/);
    expect(lines).toMatch(/SMALL model: MISSING.*optional/);
    expect(lines).toMatch(/Live Coach session: NOT running/);
  });
  it("shows all ready", () => {
    const ok = { status: "ready", path: "/p", bytes: 10 * MB, dir: "/d" } as const;
    const lines = describePrerequisites({ enabled: true, tiny: ok, small: ok, liveSessionActive: true }).join("\n");
    expect(lines).not.toMatch(/MISSING|OFF|NOT/);
  });
});
