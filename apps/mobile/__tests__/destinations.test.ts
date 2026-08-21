import {
  DESTINATIONS,
  PRIMARY_ELIGIBLE_DESTINATIONS,
  getDestination,
  isPrimaryEligible,
  type DestId,
} from "../src/nav/destinations";

const EXPECTED_IDS: DestId[] = [
  "coach",
  "analyze",
  "recordings",
  "growth",
  "watchSetup",
  "voiceProfile",
  "therapistDashboard",
  "settings",
  "tutorial",
];

describe("destinations registry", () => {
  it("has exactly the nine real destinations from the current app", () => {
    expect(DESTINATIONS.map((d) => d.id).sort()).toEqual(
      [...EXPECTED_IDS].sort(),
    );
  });

  it("every destination has a non-empty title and iconId", () => {
    for (const d of DESTINATIONS) {
      expect(d.title.length).toBeGreaterThan(0);
      expect(d.iconId.length).toBeGreaterThan(0);
    }
  });

  it("every destination's screen has a non-empty name", () => {
    for (const d of DESTINATIONS) {
      expect(d.screen.name.length).toBeGreaterThan(0);
    }
  });

  it("has no duplicate ids", () => {
    const ids = DESTINATIONS.map((d) => d.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  // 2026-08-19 primary-eligible-expand: therapistDashboard joined the
  // primary-eligible set (the owner asked whether something like the
  // Therapist Dashboard should be addable to the bottom bar / home boxes
  // too — see its registry entry's comment for why it was safe to flip).
  it("marks exactly the five daily-use destinations as primary-eligible", () => {
    const eligible = DESTINATIONS.filter((d) => d.primaryEligible).map(
      (d) => d.id,
    );
    expect(eligible.sort()).toEqual(
      ["coach", "analyze", "recordings", "growth", "therapistDashboard"].sort(),
    );
  });

  it("marks settings/tutorial/watchSetup/voiceProfile as catalog-only", () => {
    const catalogOnly = [
      "settings",
      "tutorial",
      "watchSetup",
      "voiceProfile",
    ] as const;
    for (const id of catalogOnly) {
      expect(getDestination(id)?.primaryEligible).toBe(false);
    }
  });

  it("recordings navigates back to home by default", () => {
    const recordings = getDestination("recordings");
    expect(recordings?.screen).toEqual({ name: "recordings", returnTo: "home" });
  });

  describe("getDestination", () => {
    it("finds a known destination by id", () => {
      expect(getDestination("coach")?.title).toBe("Live Coach");
    });

    it("returns undefined for an unknown id", () => {
      expect(getDestination("not-a-real-id")).toBeUndefined();
    });
  });

  describe("PRIMARY_ELIGIBLE_DESTINATIONS", () => {
    it("only contains primary-eligible destinations", () => {
      expect(PRIMARY_ELIGIBLE_DESTINATIONS.length).toBe(5);
      for (const d of PRIMARY_ELIGIBLE_DESTINATIONS) {
        expect(d.primaryEligible).toBe(true);
      }
    });
  });

  describe("isPrimaryEligible", () => {
    it("is true for a known, primary-eligible id", () => {
      expect(isPrimaryEligible("growth")).toBe(true);
    });

    it("is true for therapistDashboard (2026-08-19 primary-eligible-expand)", () => {
      expect(isPrimaryEligible("therapistDashboard")).toBe(true);
    });

    it("is false for tutorial (deliberately left catalog-only — see its registry comment)", () => {
      expect(isPrimaryEligible("tutorial")).toBe(false);
    });

    it("is false for a known but catalog-only id", () => {
      expect(isPrimaryEligible("settings")).toBe(false);
    });

    it("is false for an unknown id", () => {
      expect(isPrimaryEligible("not-a-real-id")).toBe(false);
    });
  });
});
