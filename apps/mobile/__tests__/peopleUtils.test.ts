/** People labeling — the pure helpers behind the People screen, the "Who is
 *  this?" sheet, and the speaker badges. */
import {
  enrollRefusalReason,
  enrollRefusalTitle,
  isEnrolledPersonLabel,
  labelPersonId,
  personDisplayName,
  personSeconds,
  personSummary,
  slugifyPersonId,
  sortPeople,
} from "../src/utils/people";
import type { VoicePerson } from "../src/api/client";
import { isYou, labelProvenanceNote } from "../src/utils/speakerLabels";

function person(over: Partial<VoicePerson>): VoicePerson {
  return {
    available: true,
    storage_enabled: true,
    enrolled: true,
    enroll_count: 1,
    person_id: "mom",
    display_name: "Mom",
    is_self: false,
    samples: [],
    ...over,
  };
}

describe("slugifyPersonId", () => {
  it("lowercases, strips accents/punctuation, and never returns the reserved self id", () => {
    expect(slugifyPersonId("Mom")).toBe("mom");
    expect(slugifyPersonId("Aunt Béa!")).toBe("aunt-bea");
    expect(slugifyPersonId("  ")).toBe("person");
    expect(slugifyPersonId("self")).toBe("person");
    expect(slugifyPersonId("日本")).toBe("person");
  });

  it("suffixes a taken id so two 'Mom's never collide", () => {
    expect(slugifyPersonId("Mom", ["mom"])).toBe("mom-2");
    expect(slugifyPersonId("Mom", ["mom", "mom-2"])).toBe("mom-3");
  });

  it("caps the stem so the result fits the server's 40-char pattern", () => {
    const id = slugifyPersonId("a".repeat(80));
    expect(id.length).toBeLessThanOrEqual(40);
    expect(id).toMatch(/^[a-z0-9][a-z0-9_-]{0,39}$/);
  });
});

describe("personDisplayName / sortPeople / summary", () => {
  it("renders the owner as You and partners by name, id as a last resort", () => {
    expect(personDisplayName(person({ person_id: "self", is_self: true, display_name: null }))).toBe("You");
    expect(personDisplayName(person({}))).toBe("Mom");
    expect(personDisplayName(person({ display_name: "  " }))).toBe("mom");
    expect(personDisplayName(null)).toBe("");
  });

  it("pins You first, then A→Z", () => {
    const sorted = sortPeople([
      person({ person_id: "zed", display_name: "Zed" }),
      person({ person_id: "self", is_self: true, display_name: null }),
      person({ person_id: "amy", display_name: "Amy" }),
    ]);
    expect(sorted.map((p) => p.person_id)).toEqual(["self", "amy", "zed"]);
  });

  it("sums learned seconds only when the server measured them", () => {
    const p = person({
      enroll_count: 3,
      samples: [
        { id: "a", recording_id: "r", speaker: "S", at: null, seconds: 12.4 },
        { id: "b", recording_id: null, speaker: null, at: null, note: "guided enrollment" },
        { id: "c", recording_id: "r2", speaker: "S", at: null, seconds: 7.9 },
      ],
    });
    expect(personSeconds(p)).toBe(20);
    expect(personSummary(p)).toBe("3 samples · 20 s of speech");
    expect(personSeconds(person({ samples: [] }))).toBeNull();
    expect(personSummary(person({ enroll_count: 1 }))).toBe("1 sample");
  });
});

describe("isEnrolledPersonLabel / labelPersonId", () => {
  const people = [{ person_id: "mom" }];

  it("is true for the enrolled rung and for a manual-person label of a known person", () => {
    expect(isEnrolledPersonLabel({ display_label: "You", label_source: "enrolled" })).toBe(true);
    expect(
      isEnrolledPersonLabel({ display_label: "Mom", label_source: "manual-person", person_id: "mom" }, people),
    ).toBe(true);
    expect(
      isEnrolledPersonLabel({ display_label: "You", label_source: "manual-person", person_id: "self" }, []),
    ).toBe(true);
  });

  it("is false for a plain manual name, an unknown person, or no label", () => {
    expect(isEnrolledPersonLabel({ display_label: "Mom", label_source: "manual" }, people)).toBe(false);
    expect(
      isEnrolledPersonLabel({ display_label: "Dad", label_source: "manual-person", person_id: "dad" }, people),
    ).toBe(false);
    expect(isEnrolledPersonLabel(null, people)).toBe(false);
  });

  it("reads the person id off a label", () => {
    expect(labelPersonId({ display_label: "Mom", label_source: "manual-person", person_id: "mom" })).toBe("mom");
    expect(labelPersonId({ display_label: "You", label_source: "enrolled" })).toBe("self");
    expect(labelPersonId({ display_label: "Alex", label_source: "enrolled" })).toBeNull();
    expect(labelPersonId({ display_label: "Mom", label_source: "manual" })).toBeNull();
  });

  it("speakerLabels helpers know the new rung", () => {
    expect(isYou({ display_label: "You", label_source: "manual-person", person_id: "self" })).toBe(true);
    expect(isYou({ display_label: "Mom", label_source: "manual-person", person_id: "mom" })).toBe(false);
    expect(labelProvenanceNote("manual-person")).toBe("named by you · a person you know");
  });
});

describe("enrollRefusalReason", () => {
  function err(status: number, detail?: string) {
    const e = new Error(detail ?? `API error: ${status}`) as Error & { status?: number; detail?: string };
    e.status = status;
    e.detail = detail;
    return e;
  }

  it("keys the three honest 422 reasons off the server's bracketed tag, verbatim", () => {
    const little = enrollRefusalReason(err(422, "[too-little-speech] only 1.0s of that speaker's voice…"));
    expect(little.kind).toBe("too-little-speech");
    expect(little.message).toBe("only 1.0s of that speaker's voice…");
    expect(enrollRefusalTitle(little.kind)).toBe("Not enough of their voice here");

    const alike = enrollRefusalReason(err(422, "[sounds-like-someone-else] that voice sounds like Mom"));
    expect(alike.kind).toBe("sounds-like-someone-else");
    expect(alike.message).toBe("that voice sounds like Mom");

    const none = enrollRefusalReason(err(422, "[no-audio] this live session kept no audio"));
    expect(none.kind).toBe("no-audio");
    expect(enrollRefusalTitle(none.kind)).toBe("No audio to learn from");
  });

  it("maps 503 to unavailable and everything else to a generic retry", () => {
    expect(enrollRefusalReason(err(503, "voice enrollment not available on this server")).kind).toBe("unavailable");
    expect(enrollRefusalReason(err(422, "speaker 'Speaker Z' is not in this recording")).kind).toBe("other");
    expect(enrollRefusalReason(err(422, "speaker 'Speaker Z' is not in this recording")).message).toContain("Speaker Z");
    const generic = enrollRefusalReason(new Error("network"));
    expect(generic.kind).toBe("other");
    expect(generic.message).toMatch(/try again/i);
    expect(enrollRefusalReason(undefined).kind).toBe("other");
  });
});
