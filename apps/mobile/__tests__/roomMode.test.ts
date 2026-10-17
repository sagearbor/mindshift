import {
  applyRoomFrame,
  IDLE_ROOM_STATE,
  MAX_ROOM_ANSWERS,
  MAX_ROOM_CARDS,
  modeRunsCoachingLoop,
  parseRoomFrame,
  ROOM_CONSENT_LINE,
  ROOM_WAKE_PHRASE,
  speechRouteRule,
  type RoomViewState,
} from "../src/live/roomMode";

const card = (id: string, t = 1) => ({
  type: "room_card",
  id,
  title: "Acme · Q3",
  fact: "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.",
  source_item_id: "item-acme",
  source_title: "Acme account",
  t,
});

function fold(frames: unknown[], start: RoomViewState = IDLE_ROOM_STATE): RoomViewState {
  return frames.reduce<RoomViewState>((s, f) => {
    const parsed = parseRoomFrame(f);
    return parsed ? applyRoomFrame(s, parsed) : s;
  }, start);
}

describe("parseRoomFrame", () => {
  it("parses a card with its source", () => {
    expect(parseRoomFrame(card("c1", 8))).toEqual({
      kind: "card",
      card: {
        id: "c1",
        title: "Acme · Q3",
        fact: "Q3 2026: Acme bought 120 seats of the Pro plan for $48,000.",
        sourceItemId: "item-acme",
        sourceTitle: "Acme account",
        t: 8,
      },
    });
  });

  it("parses an answer and an admitted unknown", () => {
    expect(
      parseRoomFrame({ type: "room_answer", question: "q", text: "Acme bought 120 seats.", known: true, source_item_ids: ["a", 3], t: 33, speak: true }),
    ).toEqual({ kind: "answer", answer: { question: "q", text: "Acme bought 120 seats.", known: true, sourceItemIds: ["a"], t: 33 } });
    const unknown = parseRoomFrame({ type: "room_answer", question: "churn?", text: "I don't have that.", known: false, t: 63 });
    expect(unknown?.kind === "answer" && unknown.answer.known).toBe(false);
  });

  it("rejects non-room and malformed frames", () => {
    expect(parseRoomFrame({ type: "suggestion", suggestions: ["x"] })).toBeNull();
    expect(parseRoomFrame({ type: "room_card", id: "c", fact: "  " })).toBeNull();
    expect(parseRoomFrame({ type: "room_answer", text: "" })).toBeNull();
    expect(parseRoomFrame(null)).toBeNull();
    expect(parseRoomFrame("room_card")).toBeNull();
  });
});

describe("applyRoomFrame", () => {
  it("dedupes cards by id and keeps the newest few, newest first", () => {
    const ids = ["a", "b", "a", "c", "d", "e", "f"];
    const s = fold(ids.map((id, i) => card(id, i)));
    expect(s.cards.map((c) => c.id)).toEqual(["f", "e", "d", "c"]);
    expect(s.cards).toHaveLength(MAX_ROOM_CARDS);
  });

  it("listening -> answer clears listening; errors are shown until the next answer", () => {
    let s = fold([{ type: "room_listening", t: 1 }]);
    expect(s.listening).toBe(true);
    s = fold([{ type: "room_answer_error", question: "q", reason: "RuntimeError", t: 2 }], s);
    expect(s).toMatchObject({ listening: false, answerError: "RuntimeError" });
    s = fold([{ type: "room_answer", question: "q", text: "A.", known: true, t: 3 }], s);
    expect(s).toMatchObject({ answerError: null, listening: false });
    const many = fold(
      Array.from({ length: 5 }, (_, i) => ({ type: "room_answer", question: `q${i}`, text: `A${i}.`, t: i })),
    );
    expect(many.answers.map((a) => a.question)).toEqual(["q4", "q3", "q2"]);
    expect(many.answers).toHaveLength(MAX_ROOM_ANSWERS);
  });
});

describe("mode policy", () => {
  it("room may use the loudspeaker; earpiece never; others unchanged", () => {
    expect(speechRouteRule("room")).toBe("loudspeaker-ok");
    expect(speechRouteRule("earpiece")).toBe("private-only");
    expect(speechRouteRule("speaker")).toBe("any");
    expect(speechRouteRule("call")).toBe("any");
  });

  it("room never runs the personal coaching loop", () => {
    expect(modeRunsCoachingLoop("room")).toBe(false);
    expect(modeRunsCoachingLoop("journal")).toBe(false);
    expect(modeRunsCoachingLoop("earpiece")).toBe(true);
    expect(modeRunsCoachingLoop("speaker")).toBe(true);
  });

  it("the consent line names the wake phrase", () => {
    expect(ROOM_CONSENT_LINE).toContain(ROOM_WAKE_PHRASE);
    expect(ROOM_CONSENT_LINE.toLowerCase()).toContain("listening");
  });
});
