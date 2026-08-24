/** The therapist dashboard's two-sided additions: pending "wants to share
 *  with you" requests (accept / decline), the patient list ("You" first,
 *  linked patients even with no sessions yet), and pull-to-refresh. */
import React from "react";
import renderer, { act, ReactTestInstance } from "react-test-renderer";
import TherapistDashboard, { patientRows } from "../src/screens/TherapistDashboard";
import { useDashboardStore, type SavedSession } from "../src/store/dashboardStore";
import { listDashboardSessions } from "../src/api/client";
import { acceptPatient, declinePatient, listPatients } from "../src/api/therapist";

jest.mock("../src/api/client", () => ({
  listDashboardSessions: jest.fn(),
}));
jest.mock("../src/api/therapist", () => ({
  listPatients: jest.fn(),
  acceptPatient: jest.fn(),
  declinePatient: jest.fn(),
}));
const mockListSessions = listDashboardSessions as jest.Mock;
const mockListPatients = listPatients as jest.Mock;
const mockAccept = acceptPatient as jest.Mock;
const mockDecline = declinePatient as jest.Mock;

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
      { label: "You", sessions: 1, linked: false },
      { label: "alex@example.com", sessions: 0, linked: true },
      { label: "sage@example.com", sessions: 1, linked: true },
    ]);
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
      comp = renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />);
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
      comp = renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />);
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
      comp = renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />);
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
      comp = renderer.create(<TherapistDashboard onSelectSession={jest.fn()} />);
    });
    await flush();
    await flush();
    expect(queryId(comp!, "pending-patients")).toBeNull();
    expect(queryId(comp!, "session-e1")).toBeTruthy();
    expect(comp!.root.findByProps({ testID: "filter-You" })).toBeTruthy();
  });
});
