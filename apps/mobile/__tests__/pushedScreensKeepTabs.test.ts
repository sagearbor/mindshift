/**
 * Rule: every screen reachable from the app keeps the bottom tab bar.
 * App.tsx renders a screen only through renderScreen(), inside either
 * AppChrome (primary) or PushedScreenChrome (pushed) — both render
 * ChromeTabBar. This guard fails if a third render path appears (a screen
 * rendered outside both wrappers) or a wrapper stops carrying the tab bar.
 */
import * as fs from "fs";
import * as path from "path";

const read = (rel: string) => fs.readFileSync(path.join(__dirname, "..", rel), "utf8");

describe("every screen keeps the bottom tabs", () => {
  const app = read("App.tsx");

  it("App.tsx renders renderScreen() only inside AppChrome or PushedScreenChrome", () => {
    expect((app.match(/\{renderScreen\(\)\}/g) ?? []).length).toBe(2);
    const primary = app.indexOf("<AppChrome");
    const end = app.indexOf("</PushedScreenChrome>");
    expect(primary).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(primary);
    for (const m of app.matchAll(/\{renderScreen\(\)\}/g)) {
      expect(m.index!).toBeGreaterThan(primary);
      expect(m.index!).toBeLessThan(end);
    }
  });

  it("both chromes render ChromeTabBar", () => {
    expect(read("src/components/AppChrome.tsx")).toContain("<ChromeTabBar");
    expect(read("src/components/PushedScreenChrome.tsx")).toContain("<ChromeTabBar");
  });
});
