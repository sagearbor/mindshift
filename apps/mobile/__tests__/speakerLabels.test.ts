import { speakerLabel, labelProvenanceNote } from "../src/utils/speakerLabels";
import type { SpeakerLabel } from "../src/api/client";

const labels: Record<string, SpeakerLabel> = {
  "Speaker A": { display_label: "Joe", label_source: "name" },
  "Speaker B": { display_label: "Higher voice", label_source: "voice" },
  "Speaker C": { display_label: "Speaker C", label_source: "generic" },
  "Speaker D": { display_label: "  ", label_source: "name" }, // blank → fallback
  "Speaker E": { display_label: "You", label_source: "enrolled" },
};

describe("speakerLabel", () => {
  it("returns the inferred name for a name-sourced speaker", () => {
    expect(speakerLabel("Speaker A", labels)).toBe("Joe");
  });

  it('renders "You" for an enrolled-voiceprint speaker (top rung)', () => {
    expect(speakerLabel("Speaker E", labels)).toBe("You");
  });

  it("returns the relative voice label for a voice-sourced speaker", () => {
    expect(speakerLabel("Speaker B", labels)).toBe("Higher voice");
  });

  it("returns the generic id when the source is generic", () => {
    expect(speakerLabel("Speaker C", labels)).toBe("Speaker C");
  });

  it("falls back to the raw id when the display_label is blank", () => {
    expect(speakerLabel("Speaker D", labels)).toBe("Speaker D");
  });

  it("falls back to the raw id for a speaker missing from the map", () => {
    expect(speakerLabel("Speaker Z", labels)).toBe("Speaker Z");
  });

  it("falls back to the raw id when the whole map is absent (old recording)", () => {
    expect(speakerLabel("Speaker A")).toBe("Speaker A");
    expect(speakerLabel("Speaker A", undefined)).toBe("Speaker A");
    expect(speakerLabel("Speaker A", null)).toBe("Speaker A");
  });

  it("never returns an empty string", () => {
    expect(speakerLabel("", labels)).toBe("");
    expect(speakerLabel("Bob", {})).toBe("Bob");
  });
});

describe("labelProvenanceNote", () => {
  it('names a manual override "named by you" (top rung)', () => {
    expect(labelProvenanceNote("manual")).toBe("named by you");
  });

  it("describes the inferred rungs plainly", () => {
    expect(labelProvenanceNote("enrolled")).toBe("your enrolled voice");
    expect(labelProvenanceNote("name")).toBe("detected from the words");
    expect(labelProvenanceNote("voice")).toBe("detected voice");
  });

  it("returns null for the raw-id rung and unknown/absent sources", () => {
    expect(labelProvenanceNote("generic")).toBeNull();
    expect(labelProvenanceNote("something-new")).toBeNull();
    expect(labelProvenanceNote(undefined)).toBeNull();
    expect(labelProvenanceNote(null)).toBeNull();
  });
});
