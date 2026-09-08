/**
 * src/live/naturalTurn.ts — the NaturalTurn port (Cooney & Reece 2025).
 * Classification cases mirror upstream determine_utterance_type's decision
 * order; the batch cases mirror _label_turns' containment + merge.
 */
import {
  classifyUtterance,
  labelTurns,
  liveTurnKind,
  mergePrimaries,
  naturalTurns,
  wordsOf,
  MAX_PAUSE_SECONDS,
} from "../src/live/naturalTurn";

describe("wordsOf", () => {
  it("lowercases, strips punctuation, keeps apostrophes", () => {
    expect(wordsOf("Yeah, exactly!")).toEqual(["yeah", "exactly"]);
    expect(wordsOf("I'm... okay?")).toEqual(["i'm", "okay"]);
    expect(wordsOf("...")).toEqual([]);
  });
});

describe("classifyUtterance (upstream decision order)", () => {
  it("all-cue utterances are backchannels regardless of length", () => {
    expect(classifyUtterance("yeah", 0.4)).toBe("backchannel");
    expect(classifyUtterance("oh wow really okay yeah", 2.0)).toBe("backchannel"); // 5 words, prop 1
  });
  it("more than three words (not all cues) is a secondary turn", () => {
    expect(classifyUtterance("and then what happened", 1.2)).toBe("secondary");
  });
  it("a short turn STARTING with a not-cue is secondary, never a backchannel", () => {
    expect(classifyUtterance("so yeah", 0.6)).toBe("secondary");
    expect(classifyUtterance("i know right", 0.8)).toBe("secondary");
  });
  it("cue proportion >= 0.5 is a backchannel; below is other", () => {
    expect(classifyUtterance("yeah exactly totally", 0.9)).toBe("backchannel"); // 2/3 cues
    expect(classifyUtterance("totally agree man", 0.9)).toBe("other"); // 0/3 cues
  });
  it("no words -> null", () => {
    expect(classifyUtterance("...", 0.2)).toBeNull();
  });

  it("'ok' (ASR spelling of okay) is a backchannel cue", () => {
    expect(classifyUtterance("ok", 0.4)).toBe("backchannel");
    expect(classifyUtterance("oh ok", 0.6)).toBe("backchannel"); // both cues
  });
});

describe("liveTurnKind", () => {
  it("backchannel or primary only (single-mic live has no containment)", () => {
    expect(liveTurnKind("mhm", 0.3)).toBe("backchannel");
    expect(liveTurnKind("so yeah", 0.6)).toBe("primary"); // secondary counts as real speech live
    expect(liveTurnKind("you never listen to me", 1.8)).toBe("primary");
    expect(liveTurnKind("", 0.1)).toBe("primary"); // no words: never suppress on empties
  });
});

describe("labelTurns (containment)", () => {
  const A = "spk-a";
  const B = "spk-b";
  it("an utterance fully inside another's span is non-primary and attached", () => {
    const labeled = labelTurns([
      { speaker: A, start: 0, end: 10, text: "we went to the lake and it was beautiful" },
      { speaker: B, start: 4, end: 4.5, text: "oh wow" },
      { speaker: B, start: 11, end: 13, text: "tell me more about it" },
    ]);
    expect(labeled.map((u) => u.kind)).toEqual(["primary", "backchannel", "primary"]);
    expect(labeled[1].interjects).toBe(0);
  });
  it("the forward scan stops at the first non-contained utterance (upstream break)", () => {
    const labeled = labelTurns([
      { speaker: A, start: 0, end: 5, text: "here is my whole story about that" },
      { speaker: B, start: 4, end: 7, text: "hang on that is not what i heard" }, // overlaps but NOT contained
      { speaker: B, start: 4.2, end: 4.8, text: "yeah" }, // starts inside A but scan already broke
    ]);
    // sorted order: A(0-5), B(4-7), B(4.2-4.8): the 4.2-4.8 utterance IS
    // contained in A's span but sits after the break -> stays primary.
    expect(labeled[1].kind).toBe("primary");
    expect(labeled[2].kind).toBe("backchannel"); // contained in B(4-7)'s span, claimed by it
    expect(labeled[2].interjects).toBe(1);
  });
});

describe("mergePrimaries", () => {
  const A = "spk-a";
  const B = "spk-b";
  it("joins a speaker's consecutive primaries across pauses <= MAX_PAUSE_SECONDS", () => {
    const merged = naturalTurns([
      { speaker: A, start: 0, end: 2, text: "we should talk about" },
      { speaker: A, start: 3.2, end: 5, text: "the thing from yesterday" }, // 1.2 s pause: merges
      { speaker: A, start: 7.0, end: 8, text: "anyway" }, // 2.0 s pause: new turn
    ]);
    expect(merged).toHaveLength(2);
    expect(merged[0].text).toBe("we should talk about the thing from yesterday");
    expect(merged[0].parts).toBe(2);
    expect(MAX_PAUSE_SECONDS).toBe(1.5);
  });
  it("a backchannel between two primaries never breaks the merge", () => {
    const merged = naturalTurns([
      { speaker: A, start: 0, end: 3, text: "so i told him the plan was off" },
      { speaker: B, start: 1.0, end: 1.4, text: "mhm" },
      { speaker: A, start: 3.8, end: 6, text: "and he took it well" },
    ]);
    expect(merged).toHaveLength(1);
    expect(merged[0].attached.map((u) => u.kind)).toEqual(["backchannel"]);
    expect(merged[0].text).toBe("so i told him the plan was off and he took it well");
  });
  it("different speakers never merge", () => {
    const merged = mergePrimaries(
      labelTurns([
        { speaker: A, start: 0, end: 2, text: "how was school today" },
        { speaker: B, start: 2.3, end: 4, text: "it was fine i guess" },
      ]),
    );
    expect(merged).toHaveLength(2);
  });
});
