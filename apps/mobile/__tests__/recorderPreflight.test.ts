import {
  runPreflight,
  estimateSessionBytes,
  MIN_STORAGE_SECONDS,
} from "../src/recorder/preflight";

// iOS WAV rate: 16 kHz mono 16-bit ≈ 32 KB/s. Convenient round numbers.
const RATE = 32000;
const PLANNED = 90 * 60; // 90-minute planned session

function preflight(overrides: {
  freeBytes?: number | null;
  batteryLevel?: number | null;
}) {
  return runPreflight({
    freeBytes: overrides.freeBytes === undefined ? 10e9 : overrides.freeBytes,
    batteryLevel:
      overrides.batteryLevel === undefined ? 0.9 : overrides.batteryLevel,
    bytesPerSecond: RATE,
    plannedSeconds: PLANNED,
  });
}

describe("estimateSessionBytes", () => {
  it("multiplies rate by duration", () => {
    expect(estimateSessionBytes(RATE, 60)).toBe(RATE * 60);
  });
});

describe("runPreflight — storage gate", () => {
  it("passes with ample free space", () => {
    const r = preflight({ freeBytes: 10e9 });
    expect(r.storage.blocking).toBe(false);
    expect(r.storage.message).toBeNull();
    expect(r.canStart).toBe(true);
  });

  it("blocks when less than the minimum floor fits", () => {
    // Just under MIN_STORAGE_SECONDS worth of audio.
    const r = preflight({ freeBytes: RATE * MIN_STORAGE_SECONDS - 1 });
    expect(r.storage.blocking).toBe(true);
    expect(r.storage.message).toBeTruthy();
    expect(r.canStart).toBe(false);
  });

  it("warns without blocking when the planned session does not fit", () => {
    // Room for 30 minutes, planned 90.
    const r = preflight({ freeBytes: RATE * 30 * 60 });
    expect(r.storage.blocking).toBe(false);
    expect(r.canStart).toBe(true);
    // The honest message names roughly how many minutes actually fit.
    expect(r.storage.message).toContain("30");
  });

  it("warns honestly when free space is unknown", () => {
    const r = preflight({ freeBytes: null });
    expect(r.storage.blocking).toBe(false);
    expect(r.canStart).toBe(true);
    expect(r.storage.message).toMatch(/storage/i);
  });
});

describe("runPreflight — battery gate", () => {
  it("warns below 30% with the actual percentage", () => {
    const r = preflight({ batteryLevel: 0.2 });
    expect(r.battery.warn).toBe(true);
    expect(r.battery.message).toContain("20");
  });

  it("does not warn at or above 30%", () => {
    expect(preflight({ batteryLevel: 0.3 }).battery.warn).toBe(false);
    expect(preflight({ batteryLevel: 0.95 }).battery.warn).toBe(false);
  });

  it("stays silent when the level is unknown — never invents a reading", () => {
    const r = preflight({ batteryLevel: null });
    expect(r.battery.warn).toBe(false);
    expect(r.battery.message).toBeNull();
  });

  it("never blocks the start (battery is a warning, not a gate)", () => {
    expect(preflight({ batteryLevel: 0.05 }).canStart).toBe(true);
  });
});
