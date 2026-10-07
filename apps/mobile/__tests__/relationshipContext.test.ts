/**
 * F. No client-side relationship default ("Husband / Wife" is gone); an
 * optional relationship reaches the on-device prompt only when chosen.
 */
jest.mock("../src/api/client", () => ({
  __esModule: true,
  postRespond: jest.fn().mockResolvedValue({ suggestions: [], tone_score: {} }),
}));

import { postRespond } from "../src/api/client";
import { useSessionStore } from "../src/store/sessionStore";
import { buildPrompt, type SuggestInput } from "../src/live/localLlm";
import { isRelationship, RELATIONSHIPS } from "../src/live/sessionContext";

const base: SuggestInput = {
  text: "you never listen",
  speaker: "Speaker B",
  isSelf: false,
  empathy: 50,
  context: [],
  mode: "earpiece",
};

describe("relationship: generic unless the user says otherwise", () => {
  it("the text-review store starts with no relationship and sends a generic role", async () => {
    const fresh = jest.requireActual("../src/store/sessionStore") as typeof import("../src/store/sessionStore");
    expect(fresh.useSessionStore.getState().role).toBe("");
    expect(JSON.stringify(fresh.useSessionStore.getState())).not.toMatch(/husband|wife/i);
    useSessionStore.setState({ turns: [{ speaker: "A", text: "hi" }] });
    await useSessionStore.getState().fetchSuggestions();
    expect((postRespond as jest.Mock).mock.calls[0][0].role).toBe("general conversation");
  });

  it("offers child / partner / parent / coworker / friend / other", () => {
    expect([...RELATIONSHIPS]).toEqual(["child", "partner", "parent", "coworker", "friend", "other"]);
    expect(isRelationship("partner")).toBe(true);
    expect(isRelationship("Husband")).toBe(false);
  });

  it("the on-device prompt says nothing about a relationship unless one was chosen", () => {
    expect(buildPrompt(base).user).not.toMatch(/talking with/);
    expect(buildPrompt({ ...base, relationship: null }).user).not.toMatch(/talking with/);
    expect(buildPrompt({ ...base, relationship: "coworker" }).user).toMatch(/^The coached person is talking with their coworker\./);
    expect(buildPrompt({ ...base, relationship: "other" }).user).not.toMatch(/talking with/);
  });
});
