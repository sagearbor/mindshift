import { parseTranscript } from "../src/store/sessionStore";

describe("parseTranscript (async review paste)", () => {
  it("splits labeled turns on 'Name:' prefixes", () => {
    const turns = parseTranscript("Me: I do all the cooking\nHer: I've been swamped");
    expect(turns).toEqual([
      { speaker: "Me", text: "I do all the cooking" },
      { speaker: "Her", text: "I've been swamped" },
    ]);
  });

  it("skips blank lines", () => {
    const turns = parseTranscript("Me: hi\n\n\nHer: hello\n");
    expect(turns).toHaveLength(2);
  });

  it("keeps an unlabeled line as text with an empty speaker", () => {
    const turns = parseTranscript("just thinking out loud here");
    expect(turns).toEqual([{ speaker: "", text: "just thinking out loud here" }]);
  });

  it("does not mistake a mid-sentence colon for a speaker label", () => {
    const turns = parseTranscript("here's the thing: you never listen");
    expect(turns).toEqual([
      { speaker: "", text: "here's the thing: you never listen" },
    ]);
  });

  it("allows multi-word speaker labels", () => {
    const turns = parseTranscript("My wife: please help with dishes");
    expect(turns).toEqual([
      { speaker: "My wife", text: "please help with dishes" },
    ]);
  });

  it("returns nothing for empty input", () => {
    expect(parseTranscript("   \n  \n")).toEqual([]);
  });
});
