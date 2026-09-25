/**
 * Android/iOS parity guard. The two binaries share one JS bundle, so the only
 * way they drift is a native change that reaches one store and not the other,
 * or an OTA that outruns both. This test makes that a build failure:
 *
 *  - native inputs (app.json minus counters, eas.json, plugins/, targets/,
 *    native dependency versions) must match native-fingerprint.json, OR
 *    expo.version must have been bumped since the record was written;
 *  - runtimeVersion policy must stay `appVersion` so an OTA can never reach a
 *    binary built from a different expo.version;
 *  - the production EAS profile must auto-increment both platforms' counters.
 *
 * When it fails legitimately: bump expo.version in app.json, run
 * `node scripts/nativeFingerprint.js --update "<why>"`, and build both
 * platforms with scripts/release_both.sh.
 */
import fs from "fs";
import path from "path";

// eslint-disable-next-line @typescript-eslint/no-var-requires
const fp = require("../scripts/nativeFingerprint.js");

const ROOT = path.resolve(__dirname, "..");
const appJson = JSON.parse(fs.readFileSync(path.join(ROOT, "app.json"), "utf8"));
const easJson = JSON.parse(fs.readFileSync(path.join(ROOT, "eas.json"), "utf8"));

describe("native parity guard", () => {
  it("has a recorded fingerprint", () => {
    expect(fp.readRecord()).not.toBeNull();
  });

  it("native inputs unchanged since the record, or expo.version bumped since", () => {
    const rec = fp.readRecord();
    const { hash } = fp.computeFingerprint();
    const { version } = fp.currentVersions();
    if (hash === rec.hash) return;
    expect({
      why: "Native inputs changed (app.json / eas.json / plugins / targets / a native dependency). Bump expo.version, then `node scripts/nativeFingerprint.js --update \"<why>\"`, then build BOTH platforms (scripts/release_both.sh).",
      recordedVersion: rec.version,
      currentVersion: version,
      versionBumped: version !== rec.version,
    }).toMatchObject({ versionBumped: true });
    // Version was bumped but the record is stale: still fail, with the fix.
    expect({
      why: "expo.version was bumped but native-fingerprint.json is stale — run `node scripts/nativeFingerprint.js --update \"<why>\"` and commit it alongside the version bump.",
      hashMatches: false,
    }).toMatchObject({ hashMatches: true });
  });

  it("keeps the OTA runtime tied to expo.version", () => {
    expect(appJson.expo.runtimeVersion).toEqual({ policy: "appVersion" });
  });

  it("production builds auto-increment both platform counters", () => {
    expect(easJson.build.production.autoIncrement).toBe(true);
    expect(easJson.build.production.android.buildType).toBe("app-bundle");
    expect(easJson.build.production.ios).toBeDefined();
  });

  it("the OTA env is identical for every profile so no platform can point at a different server", () => {
    const envs = Object.values(easJson.build)
      .map((p: any) => p.env)
      .filter(Boolean);
    const urls = new Set(envs.map((e: any) => e.EXPO_PUBLIC_API_URL));
    expect(urls.size).toBe(1);
  });
});
