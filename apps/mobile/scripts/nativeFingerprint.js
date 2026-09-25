/**
 * Native fingerprint — the parity guard between the Android and iOS binaries.
 *
 * Both stores ship the same JavaScript over the air, so the only way the two
 * platforms drift is when something NATIVE changes (a native module, a plugin,
 * an app.json permission, the watch target) and only one store gets a new
 * build — or neither does, and an OTA lands JS that needs native code the
 * installed binary lacks (runtimeVersion policy is `appVersion`, so the guard
 * is: native inputs changed ⇒ expo.version must bump ⇒ both platforms build).
 *
 * The fingerprint hashes exactly the inputs that can change native behavior:
 *   - app.json with the version/build counters removed
 *   - eas.json
 *   - plugins/**, targets/** (config plugins and the watchOS target)
 *   - the name@version of every dependency that ships android/ or ios/ code
 * JS-only dependency bumps do not move it, on purpose.
 *
 * Used by __tests__/nativeParity.test.ts and scripts/release_both.sh.
 */
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const ROOT = path.resolve(__dirname, "..");
const RECORD = path.join(ROOT, "native-fingerprint.json");

function walk(dir, out) {
  if (!fs.existsSync(dir)) return out;
  for (const name of fs.readdirSync(dir).sort()) {
    const p = path.join(dir, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) walk(p, out);
    else if (!/\.(png|jpg|jpeg|md)$/i.test(name)) out.push(p);
  }
  return out;
}

function stripVersions(appJson) {
  const clone = JSON.parse(JSON.stringify(appJson));
  const e = clone.expo ?? clone;
  delete e.version;
  if (e.android) delete e.android.versionCode;
  if (e.ios) delete e.ios.buildNumber;
  return clone;
}

function nativeDependencies() {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, "package.json"), "utf8"));
  const deps = { ...(pkg.dependencies ?? {}) };
  const out = [];
  for (const name of Object.keys(deps).sort()) {
    // Workspace hoisting puts most packages in the repo-root node_modules, so
    // resolve the way Node does instead of assuming apps/mobile/node_modules.
    let dir = path.join(ROOT, "node_modules", name);
    try {
      dir = path.dirname(require.resolve(`${name}/package.json`, { paths: [ROOT] }));
    } catch {
      /* unresolvable exports map or not installed: keep the local guess */
    }
    const hasNative =
      fs.existsSync(path.join(dir, "android")) ||
      fs.existsSync(path.join(dir, "ios")) ||
      fs.existsSync(path.join(dir, "app.plugin.js")) ||
      fs.existsSync(path.join(dir, "expo-module.config.json"));
    if (!hasNative) continue;
    let installed = deps[name];
    try {
      installed = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8")).version;
    } catch {
      /* not installed: fall back to the declared range */
    }
    out.push(`${name}@${installed}`);
  }
  return out;
}

function computeFingerprint() {
  const h = crypto.createHash("sha256");
  const appJson = JSON.parse(fs.readFileSync(path.join(ROOT, "app.json"), "utf8"));
  h.update("app.json\n" + JSON.stringify(stripVersions(appJson)) + "\n");
  h.update("eas.json\n" + fs.readFileSync(path.join(ROOT, "eas.json"), "utf8") + "\n");
  for (const dir of ["plugins", "targets"]) {
    for (const f of walk(path.join(ROOT, dir), [])) {
      h.update(path.relative(ROOT, f) + "\n" + fs.readFileSync(f) + "\n");
    }
  }
  const deps = nativeDependencies();
  h.update("native-deps\n" + deps.join("\n") + "\n");
  return { hash: h.digest("hex"), nativeDependencies: deps };
}

function currentVersions() {
  const e = JSON.parse(fs.readFileSync(path.join(ROOT, "app.json"), "utf8")).expo;
  return {
    version: e.version,
    androidVersionCode: e.android?.versionCode ?? null,
    iosBuildNumber: e.ios?.buildNumber ?? null,
  };
}

function readRecord() {
  return fs.existsSync(RECORD) ? JSON.parse(fs.readFileSync(RECORD, "utf8")) : null;
}

function writeRecord(note) {
  const { hash, nativeDependencies: deps } = computeFingerprint();
  const rec = {
    _comment:
      "Written by `node scripts/nativeFingerprint.js --update` after a native change AND an expo.version bump. Both platforms must be built from this version. See scripts/nativeFingerprint.js.",
    hash,
    version: currentVersions().version,
    recordedAt: new Date().toISOString(),
    note: note ?? null,
    nativeDependencies: deps,
  };
  fs.writeFileSync(RECORD, JSON.stringify(rec, null, 2) + "\n");
  return rec;
}

module.exports = { computeFingerprint, currentVersions, readRecord, writeRecord, RECORD };

if (require.main === module) {
  const args = process.argv.slice(2);
  if (args[0] === "--update") {
    const rec = writeRecord(args.slice(1).join(" ") || null);
    console.log(`native-fingerprint.json written: ${rec.hash.slice(0, 12)} @ ${rec.version}`);
  } else {
    const rec = readRecord();
    const { hash } = computeFingerprint();
    const v = currentVersions();
    const same = rec && rec.hash === hash;
    console.log(`fingerprint ${hash.slice(0, 12)} (recorded ${rec ? rec.hash.slice(0, 12) : "none"} @ ${rec?.version ?? "-"}); app ${v.version} android ${v.androidVersionCode} ios ${v.iosBuildNumber}`);
    if (!same) {
      console.error(
        rec && rec.version === v.version
          ? "NATIVE INPUTS CHANGED without an expo.version bump — bump expo.version, build BOTH platforms, then run --update."
          : "fingerprint differs from the record — if you already bumped expo.version and will build both platforms, run --update.",
      );
      process.exit(1);
    }
  }
}
