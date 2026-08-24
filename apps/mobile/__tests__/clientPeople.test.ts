/** People labeling — the client calls behind the People screen and the
 *  "Who is this?" sheet. Mirrors clientGrowth.test.ts's global-fetch style. */
import {
  deleteVoicePerson,
  enrollPersonFromRecording,
  listVoicePeople,
  patchSpeakerLabels,
  renameVoicePerson,
} from "../src/api/client";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(null);
});

describe("listVoicePeople", () => {
  it("GETs /voice/people and normalizes the shape", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ available: true, storage_enabled: true, people: [{ person_id: "self" }] }),
    });
    const res = await listVoicePeople();
    expect(mockFetch.mock.calls[0][0]).toMatch(/\/voice\/people$/);
    expect(res.people).toHaveLength(1);
    expect(res.available).toBe(true);
  });

  it("treats a 404 (older server) as unavailable rather than throwing", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404, json: async () => ({}) });
    expect(await listVoicePeople()).toEqual({ available: false, storage_enabled: false, people: [] });
  });
});

describe("patchSpeakerLabels with people", () => {
  it("sends the people map only when given", async () => {
    mockFetch.mockResolvedValue({ ok: true, status: 200, json: async () => ({ id: "r1", manual_speaker_labels: {}, speaker_labels: {} }) });
    await patchSpeakerLabels("r1", { "Speaker B": "Mom" });
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({ labels: { "Speaker B": "Mom" } });
    await patchSpeakerLabels("r1", { "Speaker B": "Mom" }, { "Speaker B": "mom" });
    expect(JSON.parse(mockFetch.mock.calls[1][1].body)).toEqual({
      labels: { "Speaker B": "Mom" }, people: { "Speaker B": "mom" },
    });
  });
});

describe("enrollPersonFromRecording", () => {
  it("POSTs the recording + speaker (+ name for a new person) and returns the result", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ enrolled: true, person_id: "mom", seconds: 12.4, enroll_count: 1, speaker_labels: {} }),
    });
    const res = await enrollPersonFromRecording("mom", "r1", "Speaker B", "Mom");
    expect(mockFetch.mock.calls[0][0]).toMatch(/\/voice\/people\/mom\/enroll-from-recording$/);
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({
      recording_id: "r1", speaker_label: "Speaker B", display_name: "Mom",
    });
    expect(res.seconds).toBe(12.4);
  });

  it("surfaces the server's 422 detail with the status on the error", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({ detail: "[too-little-speech] only 1.0s of that speaker's voice…" }),
    });
    await expect(enrollPersonFromRecording("mom", "r1", "Speaker C")).rejects.toMatchObject({
      status: 422,
      detail: "[too-little-speech] only 1.0s of that speaker's voice…",
    });
  });
});

describe("renameVoicePerson / deleteVoicePerson", () => {
  it("PATCHes the display name and DELETEs the person", async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ person_id: "mom", display_name: "Mum" }) });
    const renamed = await renameVoicePerson("mom", "Mum");
    expect(mockFetch.mock.calls[0][1].method).toBe("PATCH");
    expect(JSON.parse(mockFetch.mock.calls[0][1].body)).toEqual({ display_name: "Mum" });
    expect(renamed.display_name).toBe("Mum");

    mockFetch.mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ deleted: true, person_id: "mom" }) });
    expect(await deleteVoicePerson("mom")).toEqual({ deleted: true, person_id: "mom" });
    expect(mockFetch.mock.calls[1][1].method).toBe("DELETE");
    expect(mockFetch.mock.calls[1][0]).toMatch(/\/voice\/people\/mom$/);
  });
});
