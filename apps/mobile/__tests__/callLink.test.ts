import {
  callDeepLink,
  callWebUrl,
  inviteMessage,
  isJoinCode,
  parseCallLink,
} from "../src/nav/callLink";
import { shareInvite } from "../src/live/call/invite";

describe("callLink", () => {
  it("parses both invite shapes and the bare web path", () => {
    expect(parseCallLink("mindshift://call/K7M2-PQ")).toEqual({ code: "K7M2-PQ", role: "participant" });
    expect(parseCallLink("https://arborfam-hub.web.app/call/K7M2PQ")).toEqual({ code: "K7M2PQ", role: "participant" });
    expect(parseCallLink("https://arborfam-hub.web.app/call/K7M2PQ/?utm=1#x")).toEqual({ code: "K7M2PQ", role: "participant" });
    expect(parseCallLink("/call/abc123")).toEqual({ code: "abc123", role: "participant" });
    expect(parseCallLink("http://localhost:8081/call/abc123")).toEqual({ code: "abc123", role: "participant" });
    // The role rides a query param on either shape.
    expect(parseCallLink("https://arborfam-hub.web.app/call/K7M2PQ?role=therapist")).toEqual({ code: "K7M2PQ", role: "therapist" });
    expect(parseCallLink("mindshift://call/K7M2PQ?role=therapist")).toEqual({ code: "K7M2PQ", role: "therapist" });
    expect(parseCallLink("https://arborfam-hub.web.app/call/K7M2PQ?role=bogus")).toEqual({ code: "K7M2PQ", role: "participant" });
  });

  it("rejects everything else", () => {
    expect(parseCallLink(null)).toBeNull();
    expect(parseCallLink("")).toBeNull();
    expect(parseCallLink("https://arborfam-hub.web.app/")).toBeNull();
    expect(parseCallLink("https://arborfam-hub.web.app/calls/abc")).toBeNull();
    expect(parseCallLink("mindshift://home")).toBeNull();
    expect(parseCallLink("https://arborfam-hub.web.app/call/a b")).toBeNull();
    expect(parseCallLink("https://arborfam-hub.web.app/call/../x")).toBeNull();
    expect(parseCallLink("mailto:call/abc")).toBeNull();
  });

  it("builds the links and the invite text", () => {
    expect(callWebUrl("K7")).toBe("https://arborfam-hub.web.app/call/K7");
    expect(callWebUrl("K7", "therapist")).toBe("https://arborfam-hub.web.app/call/K7?role=therapist");
    expect(callDeepLink("K7")).toBe("mindshift://call/K7");
    expect(callDeepLink("K7", "therapist")).toBe("mindshift://call/K7?role=therapist");
    expect(isJoinCode("K7")).toBe(false);
    expect(isJoinCode("K7M")).toBe(true);
    const inv = inviteMessage("K7M", null);
    expect(inv.url).toBe("https://arborfam-hub.web.app/call/K7M");
    expect(inv.message).toContain("https://arborfam-hub.web.app/call/K7M");
    expect(inv.message).toContain("mindshift://call/K7M");
    // The server's own join_url wins when it is a real URL.
    expect(inviteMessage("K7M", "https://example.test/j/K7M").url).toBe("https://example.test/j/K7M");
    expect(inviteMessage("K7M", "not a url").url).toBe("https://arborfam-hub.web.app/call/K7M");
  });
});

describe("shareInvite", () => {
  it("uses the native share sheet and reports 'shown' when it fails", async () => {
    const share = jest.fn().mockResolvedValue({ action: "sharedAction" });
    expect(await shareInvite("K7M", null, "participant", { share, platform: "android" })).toEqual({
      outcome: "shared",
      url: "https://arborfam-hub.web.app/call/K7M",
    });
    expect(share).toHaveBeenCalledWith(expect.objectContaining({ url: "https://arborfam-hub.web.app/call/K7M" }));
    const failing = jest.fn().mockRejectedValue(new Error("no sheet"));
    expect((await shareInvite("K7M", null, "participant", { share: failing, platform: "android" })).outcome).toBe("shown");
  });

  it("on the web prefers the Web Share API, then the clipboard, then just shows the link", async () => {
    const webShare = jest.fn().mockResolvedValue(undefined);
    expect((await shareInvite("K7M", null, "participant", { platform: "web", webShare, copy: null })).outcome).toBe("shared");
    const copy = jest.fn().mockResolvedValue(undefined);
    expect((await shareInvite("K7M", null, "participant", { platform: "web", webShare: null, copy })).outcome).toBe("copied");
    expect(copy).toHaveBeenCalledWith("https://arborfam-hub.web.app/call/K7M");
    expect((await shareInvite("K7M", null, "participant", { platform: "web", webShare: null, copy: null })).outcome).toBe("shown");
  });
});
