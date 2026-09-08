import { parseJournalLink } from "../src/nav/journalLink";

describe("parseJournalLink", () => {
  it("parses start and stop on the app scheme", () => {
    expect(parseJournalLink("mindshift://journal/start")).toBe("start");
    expect(parseJournalLink("mindshift://journal/stop")).toBe("stop");
    expect(parseJournalLink("mindshift://journal/start/")).toBe("start");
    expect(parseJournalLink("  mindshift://journal/stop  ")).toBe("stop");
  });

  it("accepts the web origin's path and a bare path", () => {
    expect(parseJournalLink("https://arborfam-hub.web.app/journal/start")).toBe("start");
    expect(parseJournalLink("/journal/stop")).toBe("stop");
  });

  it("ignores a query/hash", () => {
    expect(parseJournalLink("mindshift://journal/start?utm=assistant")).toBe("start");
    expect(parseJournalLink("mindshift://journal/stop#x")).toBe("stop");
  });

  it("rejects everything else", () => {
    expect(parseJournalLink(null)).toBeNull();
    expect(parseJournalLink(undefined)).toBeNull();
    expect(parseJournalLink("")).toBeNull();
    expect(parseJournalLink("mindshift://call/ABC123")).toBeNull();
    expect(parseJournalLink("mindshift://journal")).toBeNull();
    expect(parseJournalLink("mindshift://journal/pause")).toBeNull();
    expect(parseJournalLink("otherapp://journal/start")).toBeNull();
    expect(parseJournalLink("journal/start")).toBeNull();
  });
});
