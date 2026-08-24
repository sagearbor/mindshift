import {
  acceptPatient,
  declinePatient,
  getSessionNote,
  getTherapistLink,
  listPatients,
  putSessionNote,
  setAutoShare,
  setTherapistLink,
  unlinkTherapist,
} from "../src/api/therapist";

jest.mock("../src/auth/authToken", () => ({
  getFreshToken: jest.fn().mockResolvedValue("id-token"),
  getCachedToken: jest.fn(() => "id-token"),
}));

const fetchMock = global.fetch as jest.Mock;

function ok(body: unknown, status = 200) {
  return { ok: status < 400, status, json: async () => body };
}

beforeEach(() => {
  fetchMock.mockReset();
});

describe("therapist api", () => {
  it("GET /therapist/link with bearer auth; a missing linked flag reads false", async () => {
    fetchMock.mockResolvedValueOnce(ok({ linked: true, therapist_email: "mom@example.com", status: "pending", auto_share: true }));
    const link = await getTherapistLink();
    expect(link.linked).toBe(true);
    expect(link.therapist_email).toBe("mom@example.com");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/therapist\/link$/);
    expect(init.method).toBe("GET");
    expect(init.headers.Authorization).toBe("Bearer id-token");
    expect(init.headers["Content-Type"]).toBeUndefined();

    fetchMock.mockResolvedValueOnce(ok({}));
    expect((await getTherapistLink()).linked).toBe(false);
  });

  it("PUT /therapist/link surfaces the server's detail verbatim on failure", async () => {
    fetchMock.mockResolvedValueOnce(ok({ linked: true, therapist_email: "mom@example.com" }));
    await setTherapistLink("Mom@Example.com");
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body)).toEqual({ email: "Mom@Example.com" });

    fetchMock.mockResolvedValueOnce(ok({ detail: "no MindShift account with that email" }, 404));
    await expect(setTherapistLink("nobody@example.com")).rejects.toMatchObject({
      status: 404,
      detail: "no MindShift account with that email",
      message: "no MindShift account with that email",
    });

    fetchMock.mockResolvedValueOnce({ ok: false, status: 503, json: async () => { throw new Error("no body"); } });
    await expect(setTherapistLink("x@y.z")).rejects.toMatchObject({ status: 503, message: "API error: 503" });
  });

  it("PATCH auto_share / DELETE link", async () => {
    fetchMock.mockResolvedValueOnce(ok({ linked: true, auto_share: false }));
    expect((await setAutoShare(false)).auto_share).toBe(false);
    expect(fetchMock.mock.calls[0][1].method).toBe("PATCH");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ auto_share: false });

    fetchMock.mockResolvedValueOnce({ ok: true, status: 204, json: async () => ({}) });
    await expect(unlinkTherapist()).resolves.toBeUndefined();
    expect(fetchMock.mock.calls[1][1].method).toBe("DELETE");
  });

  it("therapist side: list / accept / decline", async () => {
    fetchMock.mockResolvedValueOnce(ok({ patients: [{ patient_uid: "u1", patient_email: "sage@example.com", status: "pending" }] }));
    const rows = await listPatients();
    expect(rows).toHaveLength(1);
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/therapist\/patients$/);

    fetchMock.mockResolvedValueOnce(ok({ patients: "nope" }));
    expect(await listPatients()).toEqual([]);

    fetchMock.mockResolvedValueOnce(ok({ patient_uid: "u 1", status: "accepted" }));
    expect((await acceptPatient("u 1")).status).toBe("accepted");
    expect(fetchMock.mock.calls[2][0]).toMatch(/\/therapist\/patients\/u%201\/accept$/);
    expect(fetchMock.mock.calls[2][1].method).toBe("POST");

    fetchMock.mockResolvedValueOnce({ ok: true, status: 204, json: async () => ({}) });
    await declinePatient("u1");
    expect(fetchMock.mock.calls[3][0]).toMatch(/\/therapist\/patients\/u1\/decline$/);

    fetchMock.mockResolvedValueOnce(ok({ detail: "no such patient link" }, 404));
    await expect(acceptPatient("zz")).rejects.toMatchObject({ status: 404 });
  });

  it("notes: GET / PUT keyed by episode", async () => {
    fetchMock.mockResolvedValueOnce(ok({ episode_id: "e1", text: "", updated_at: null }));
    expect((await getSessionNote("e1")).text).toBe("");
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/therapist\/notes\/e1$/);

    fetchMock.mockResolvedValueOnce(ok({ episode_id: "e1", text: "Defensive.", updated_at: "now" }));
    const saved = await putSessionNote("e1", "Defensive.");
    expect(saved.updated_at).toBe("now");
    expect(fetchMock.mock.calls[1][1].method).toBe("PUT");
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ text: "Defensive." });
  });
});
