/** The therapist dashboard's two-sided additions: pending "wants to share
 *  with you" requests (accept / decline), the patient list ("You" first,
 *  linked patients even with no sessions yet), and pull-to-refresh. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import TherapistDashboard, { patientRows } from "../src/screens/TherapistDashboard";
import { useDashboardStore, type SavedSession } from "../src/store/dashboardStore";
import { listDashboardSessions } from "../src/api/client";
import { acceptPatient, declinePatient, listPatients, markPatientSeen } from "../src/api/therapist";

/**
 * Unmount every tree this file creates.
 *
 * react-test-renderer keeps a mounted tree scheduled, and a state update that
 * lands AFTER Jest tears the environment down throws "You are trying to
 * `import` a file after the Jest environment has been torn down" — which is
 * not a test failure but a WORKER CRASH, taking every unrelated suite sharing
 * that worker with it. That is why running the whole suite at once used to
 * fail a handful of random files that each passed on their own.
 */
const __trees: renderer.ReactTestRenderer[] = [];
function track<T extends renderer.ReactTestRenderer>(t: T): T {
  __trees.push(t);
  return t;
}
afterEach(async () => {
  await act(async () => {});
  act(() => {
    for (const t of __trees.splice(0)) {
      try {
        t.unmount();
      } catch {
        // A tree a test already unmounted, or one whose teardown throws, must
        // not fail the test that otherwise passed.
      }
    }
  });
});


jest.mock("../src/api/client", () => ({
  listDashboardSessions: jest.fn(),
}));
jest.mock("../src/api/therapist", () => ({
  listPatients: jest.fn(),
  acceptPatient: jest.fn(),
  declinePatient: jest.fn(),
  markPatientSeen: jest.fn(),
}));
const mockListSessions = listDashboardSessions as jest.Mock;
const mockListPatients = listPatients as jest.Mock;
const mockAccept = acceptPatient as jest.Mock;
const mockDecline = declinePatient as jest.Mock;
const mockSeen = markPatientSeen as jest.Mock;

function queryId(comp: renderer.ReactTestRenderer, id: string): ReactTestInstance | null {
  const found = comp.root.findAll((n) => n.props?.testID === id && typeof n.type === "string");
  return found.length > 0 ? found[0] : null;
}
function press(comp: renderer.ReactTestRenderer, id: string) {
  const found = comp.root.findAll((n) => n.props?.testID === id && typeof n.props?.onPress === "function");
  if (found.length === 0) throw new Error(`no pressable ${id}`);
  found[0].props.onPress();
}

const sageSession: SavedSession = {
  id: "e1",
  recordingId: "e1",
  date: "2026-08-24T18:05:00+00:00",
  role: "sage@example.com",
  shared: true,
  source: "live",
  mode: "speaker",
  avgPleasantness: 62,
  turns: [{ speaker: "You", text: "hi", toneScores: { pleasantness: 62 } }],
};
const ownSession: SavedSession = { ...sageSession, id: "own", recordingId: "own", role: "You", shared: false };

const flush = () => act(async () => { await Promise.resolve(); });

beforeEach(() => {
  mockListSessions.mockReset().mockResolvedValue([ownSession, sageSession]);
  mockListPatients.mockReset().mockResolvedValue([]);
  mockAccept.mockReset();
  mockDecline.mockReset();
  mockSeen.mockReset().mockResolvedValue("2026-08-25T00:00:00Z");
  act(() => {
    useDashboardStore.setState({ sessions: [], selectedSessionId: null, roleFilter: null, loading: false });
  });
});

describe("patientRows", () => {
  it("'You' first, then session patients and accepted linked patients (even with no sessions), pending excluded", () => {
    const rows = patientRows([ownSession, sageSession], [
      { patient_uid: "u1", patient_email: "sage@example.com", status: "accepted", auto_share: true, created_at: null, accepted_at: null },
      { patient_uid: "u2", patient_email: "alex@example.com", status: "accepted", auto_share: true, created_at: null, accepted_at: null },
      { patient_uid: "u3", patient_email: "pending@example.com", status: "pending", auto_share: true, created_at: null, accepted_at: null },
    ]);
    expect(rows).toEqual([
      { label: "You", sessions: 1, linked: false, patientUid: null, unread: false },
      { label: "alex@example.com", sessions: 0, linked: true, patientUid: "u2", unread: false },
      { label: "sage@example.com", sessions: 1, linked: true, patientUid: "u1", unread: true },
    ]);
  });

  /**
   * The unread mark (POST /therapist/patients/{uid}/seen). It is server-side
   * so every device this therapist uses agrees about what is new; it is
   * about someone ELSE's sessions, so "You" is never unread.
   */
  it("a linked patient is unread until this therapist has opened them", () => {
    const link = (last_seen_at: string | null) => ({
      patient_uid: "u1", patient_email: "sage@example.com", status: "accepted" as const,
      auto_share: true, created_at: null, accepted_at: null, last_seen_at,
    });
    const row = (last_seen_at: string | null) =>
      patientRows([sageSession], [link(last_seen_at)]).find((r) => r.label === "sage@example.com")!;

    // Never opened: unread the moment they share anything.
    expect(row(null).unread).toBe(true);
    // Read AFTER their newest session (2026-08-24T18:05Z): caught up.
    expect(row("2026-08-25T00:00:00Z").unread).toBe(false);
    // Read BEFORE it: something new since.
    expect(row("2026-08-01T00:00:00Z").unread).toBe(true);
    // An unparseable mark is not taken as "read" — it fails toward showing.
    expect(row("not a date").unread).toBe(true);
    // A patient with nothing shared yet has nothing to be unread about.
    expect(patientRows([], [link(null)])[0].unread).toBe(false);
    // "You" is your own list, never an unread patient.
    expect(patientRows([ownSession], [])[0]).toMatchObject({ label: "You", unread: false });
  });

  it("a patient who shared by hand (no link) has no uid to mark seen", () => {
    const row = patientRows([sageSession], [])[0];
    expect(row).toMatchObject({ label: "sage@example.com", linked: false, patientUid: null, unread: false });
  });
});

describe("TherapistDashboard — patients", () => {
  it("shows a pending request; Accept moves the patient into the list, Decline removes it", async () => {
    mockListPatients.mockResolvedValue([
      { patient_uid: "u1", patient_email: "sage@example.com", status: "pending", auto_share: true, created_at: "2026-08-24T00:00:00Z", accepted_at: null },
    ]);
    mockAccept.mockResolvedValue({ patient_uid: "u1", patient_email: "sage@example.com", status: "accepted", auto_share: true, created_at: "2026-08-24T00:00:00Z", accepted_at: "now" });
    let comp: renderer.ReactTestRenderer;
    act(() => {
      comp = track(renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />));
    });
    await flush();
    await flush();
    expect(queryId(comp!, "pending-patients")).toBeTruthy();
    expect(JSON.stringify(comp!.toJSON())).toContain("Wants to share sessions with you");
    // The patient's already-shared session is listed regardless (the existing grant).
    expect(queryId(comp!, "session-e1")).toBeTruthy();

    await act(async () => {
      press(comp!, "accept-u1");
    });
    await flush();
    expect(mockAccept).toHaveBeenCalledWith("u1");
    expect(queryId(comp!, "pending-patients")).toBeNull();
    const chipNode = comp!.root.findAll(
      (n) => typeof n.type === "string" && n.props?.testID === "filter-sage@example.com",
    )[0];
    const chip = chipNode
      .findAll((n) => typeof n.type === "string")
      .flatMap((n) => n.children)
      .filter((c): c is string => typeof c === "string")
      .join("");
    expect(chip).toContain("✓");
    expect(chip).toContain("· 1");
  });

  it("Decline removes the request; a failure is stated", async () => {
    mockListPatients.mockResolvedValue([
      { patient_uid: "u1", patient_email: "sage@example.com", status: "pending", auto_share: true, created_at: null, accepted_at: null },
    ]);
    mockDecline.mockRejectedValueOnce(new Error("503")).mockResolvedValueOnce(undefined);
    let comp: renderer.ReactTestRenderer;
    act(() => {
      comp = track(renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />));
    });
    await flush();
    await flush();
    await act(async () => {
      press(comp!, "decline-u1");
    });
    await flush();
    expect(queryId(comp!, "patient-error")).toBeTruthy();
    await act(async () => {
      press(comp!, "decline-u1");
    });
    await flush();
    expect(queryId(comp!, "pending-patients")).toBeNull();
  });

  it("a linked patient with no sessions yet is listed and explains the empty filter; pull-to-refresh re-reads both", async () => {
    mockListSessions.mockResolvedValue([ownSession]);
    mockListPatients.mockResolvedValue([
      { patient_uid: "u2", patient_email: "alex@example.com", status: "accepted", auto_share: true, created_at: null, accepted_at: "x" },
    ]);
    let comp: renderer.ReactTestRenderer;
    act(() => {
      comp = track(renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />));
    });
    await flush();
    await flush();
    expect(comp!.root.findByProps({ testID: "filter-alex@example.com" })).toBeTruthy();
    await act(async () => {
      press(comp!, "filter-alex@example.com");
    });
    expect(JSON.stringify(comp!.toJSON())).toContain("No sessions from this patient yet");

    await act(async () => {
      await queryId(comp!, "dashboard-refresh")!.props.onRefresh();
    });
    expect(mockListSessions).toHaveBeenCalledTimes(2);
    expect(mockListPatients).toHaveBeenCalledTimes(2);
  });

  it("an older server (patients call fails) still shows the sessions", async () => {
    mockListPatients.mockRejectedValue(new Error("404"));
    let comp: renderer.ReactTestRenderer;
    act(() => {
      comp = track(renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />));
    });
    await flush();
    await flush();
    expect(queryId(comp!, "pending-patients")).toBeNull();
    expect(queryId(comp!, "session-e1")).toBeTruthy();
    expect(comp!.root.findByProps({ testID: "filter-You" })).toBeTruthy();
  });

  /** Opening a patient marks them read for THIS therapist, on the server. */
  describe("the unread mark", () => {
    const linked = (last_seen_at: string | null) => [{
      patient_uid: "u1", patient_email: "sage@example.com", status: "accepted",
      auto_share: true, created_at: null, accepted_at: "x", last_seen_at,
    }];
    const mount = async () => {
      let comp!: renderer.ReactTestRenderer;
      act(() => {
        comp = track(renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />));
      });
      await flush();
      await flush();
      return comp;
    };

    /** The chip's label, joined (RN splits it into fragments). */
    const chip = (comp: renderer.ReactTestRenderer, id: string) =>
      comp.root
        .findByProps({ testID: id })
        .findAll((n) => typeof n.type === "string")
        .flatMap((n) => n.children)
        .filter((c): c is string => typeof c === "string")
        .join("");

    it("shows a dot, and clears it by marking the patient seen", async () => {
      mockListPatients.mockResolvedValue(linked(null));
      const comp = await mount();
      expect(chip(comp, "filter-sage@example.com")).toBe("● sage@example.com ✓ · 1");
      await act(async () => {
        press(comp, "filter-sage@example.com");
      });
      expect(mockSeen).toHaveBeenCalledWith("u1");
      await flush();
      // The dot is gone once the server's mark comes back — and stays gone
      // when the filter is cleared again.
      await act(async () => {
        press(comp, "filter-sage@example.com");
      });
      expect(chip(comp, "filter-sage@example.com")).toBe("sage@example.com ✓ · 1");
      // Deselecting is not a second "read": only opening marks.
      expect(mockSeen).toHaveBeenCalledTimes(1);
    });

    it("doesn't mark a patient who has no link to mark", async () => {
      mockListPatients.mockResolvedValue([]);
      const comp = await mount();
      await act(async () => {
        press(comp, "filter-sage@example.com");
      });
      expect(mockSeen).not.toHaveBeenCalled();
    });

    it("a failed mark leaves the badge, never an error in the therapist's face", async () => {
      mockListPatients.mockResolvedValue(linked(null));
      mockSeen.mockRejectedValue(new Error("503"));
      const comp = await mount();
      await act(async () => {
        press(comp, "filter-sage@example.com");
        await Promise.resolve();
      });
      expect(queryId(comp, "patient-error")).toBeNull();
      // Filtering still worked — the read mark is the only thing that didn't.
      expect(queryId(comp, "session-e1")).toBeTruthy();
    });
  });
});
