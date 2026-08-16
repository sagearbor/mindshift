import fs from "fs";
import path from "path";

// Guards the openapi-typescript codegen output (scripts/gen_api_types.sh /
// `npm run gen:api` at repo root) against silent staleness or accidental
// deletion. Not a type-correctness check (that's what `tsc` catches when
// something eventually imports these types) — just a tripwire that the
// committed file exists and still describes both domains it must cover:
// the pre-existing MindShift API and the ported watch/Gauge domain.
const GENERATED_PATH = path.join(
  __dirname,
  "..",
  "src",
  "api",
  "generated",
  "openapi.d.ts",
);

describe("generated OpenAPI types", () => {
  const contents = fs.existsSync(GENERATED_PATH)
    ? fs.readFileSync(GENERATED_PATH, "utf8")
    : null;

  it("exists (run `npm run gen:api` if this fails)", () => {
    expect(contents).not.toBeNull();
  });

  it("mentions the watch domain (/live-sessions)", () => {
    expect(contents).toContain('"/live-sessions"');
  });

  it("mentions the pre-existing domain (/recordings)", () => {
    expect(contents).toContain('"/recordings"');
  });
});
