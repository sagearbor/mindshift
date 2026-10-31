/** src/api/library.ts — the coach knowledge library REST client
 *  (server/routers/library.py). Global-fetch style, like clientAccount. */
import {
  createLibraryNote,
  createLibraryProject,
  deleteLibraryProject,
  updateLibraryProject,
  deleteLibraryItem,
  getLibraryItem,
  listLibrary,
  updateLibraryItem,
  uploadLibraryFile,
} from "../src/api/library";
import { setCachedToken, setTokenProvider } from "../src/auth/authToken";

const mockFetch = global.fetch as jest.Mock;

beforeEach(() => {
  mockFetch.mockReset();
  setCachedToken(null);
  setTokenProvider(async () => "tok");
});

afterAll(() => setTokenProvider(null));

function ok(body: unknown, status = 200) {
  return { ok: true, status, json: async () => body };
}
function fail(status: number, detail: unknown) {
  return { ok: false, status, json: async () => ({ detail }) };
}

const ITEM = {
  id: "11111111-1111-4111-8111-111111111111",
  title: "Pricing",
  kind: "note",
  chars: 20,
  status: "ready",
  error: null,
  created_at: "2026-10-07T00:00:00Z",
  updated_at: "2026-10-07T00:00:00Z",
  indexed: false,
};

describe("library API client", () => {
  it("lists items with retrieval availability and limits, authenticated", async () => {
    mockFetch.mockResolvedValueOnce(
      ok({ items: [ITEM], retrieval_available: false, limits: { max_file_bytes: 10, max_text_chars: 5, max_items: 50 } }),
    );
    const res = await listLibrary();
    expect(mockFetch.mock.calls[0][0]).toMatch(/\/library\/items$/);
    expect(mockFetch.mock.calls[0][1].headers.Authorization).toBe("Bearer tok");
    expect(res.items[0].title).toBe("Pricing");
    expect(res.retrieval_available).toBe(false);
  });

  it("creates a note as JSON", async () => {
    mockFetch.mockResolvedValueOnce(ok(ITEM, 201));
    await createLibraryNote("Pricing", "Starter is 49.");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/library\/items$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ kind: "note", title: "Pricing", text: "Starter is 49." });
    expect(init.headers["Content-Type"]).toBe("application/json");
  });

  it("uploads a picked file as multipart without forcing a Content-Type", async () => {
    mockFetch.mockResolvedValueOnce(ok({ ...ITEM, kind: "document" }, 201));
    await uploadLibraryFile({ uri: "file:///doc.pdf", name: "doc.pdf", mimeType: "application/pdf" }, "Brief");
    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/library\/items$/);
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.headers["Content-Type"]).toBeUndefined();
    expect(init.headers.Authorization).toBe("Bearer tok");
  });

  it("always sends an explicit title (picked name minus extension) when none is given", async () => {
    mockFetch.mockResolvedValueOnce(ok({ ...ITEM, kind: "document" }, 201));
    await uploadLibraryFile({ uri: "file:///cache/9f1c-uuid.pdf", name: "Q3 Brief.final.pdf" });
    const form = mockFetch.mock.calls[0][1].body as FormData;
    expect(form.get("title")).toBe("Q3 Brief.final");
  });

  it("an explicit title wins over the picked name; an extensionless name is kept whole", async () => {
    mockFetch.mockResolvedValueOnce(ok({ ...ITEM, kind: "document" }, 201));
    await uploadLibraryFile({ uri: "file:///a", name: "x.pdf" }, "  Mine ");
    expect((mockFetch.mock.calls[0][1].body as FormData).get("title")).toBe("Mine");
    mockFetch.mockResolvedValueOnce(ok({ ...ITEM, kind: "document" }, 201));
    await uploadLibraryFile({ uri: "file:///a", name: "README" });
    expect((mockFetch.mock.calls[1][1].body as FormData).get("title")).toBe("README");
  });

  it("sends project_id on upload and note create when given", async () => {
    mockFetch.mockResolvedValueOnce(ok({ ...ITEM, kind: "document" }, 201));
    await uploadLibraryFile({ uri: "file:///a", name: "x.pdf" }, undefined, "proj-1");
    expect((mockFetch.mock.calls[0][1].body as FormData).get("project_id")).toBe("proj-1");
    mockFetch.mockResolvedValueOnce(ok(ITEM, 201));
    await createLibraryNote("T", "body", "proj-2");
    expect(JSON.parse(mockFetch.mock.calls[1][1].body)).toMatchObject({ kind: "note", project_id: "proj-2" });
  });

  it("project calls hit /library/projects with the right verbs and bodies", async () => {
    const P = { id: "p1", name: "A", parent_id: null, created_at: "t", updated_at: "t" };
    mockFetch.mockResolvedValue(ok(P));
    await createLibraryProject("A", null);
    await updateLibraryProject("p1", { name: "B" });
    await updateLibraryProject("p1", { parent_id: null });
    await deleteLibraryProject("p1");
    await deleteLibraryProject("p1", "move_to_parent");
    await updateLibraryItem("i1", { project_id: "p1" });
    const c = mockFetch.mock.calls;
    expect(c[0][0]).toMatch(/\/library\/projects$/);
    expect(c[0][1].method).toBe("POST");
    expect(JSON.parse(c[0][1].body)).toEqual({ name: "A", parent_id: null });
    expect(c[1][0]).toMatch(/\/library\/projects\/p1$/);
    expect(c[1][1].method).toBe("PATCH");
    expect(JSON.parse(c[1][1].body)).toEqual({ name: "B" });
    expect(JSON.parse(c[2][1].body)).toEqual({ parent_id: null });
    expect(c[3][0]).toMatch(/\/library\/projects\/p1$/);
    expect(c[3][1].method).toBe("DELETE");
    expect(c[4][0]).toMatch(/\/library\/projects\/p1\?contents=move_to_parent$/);
    expect(JSON.parse(c[5][1].body)).toEqual({ project_id: "p1" });
  });

  it("renames, edits note text, gets detail and deletes by id", async () => {
    mockFetch
      .mockResolvedValueOnce(ok(ITEM))
      .mockResolvedValueOnce(ok(ITEM))
      .mockResolvedValueOnce(ok({ ...ITEM, preview: "Starter", preview_truncated: false, chunk_count: 0 }))
      .mockResolvedValueOnce(ok({ deleted: true, id: ITEM.id }));
    await updateLibraryItem(ITEM.id, { title: "New" });
    await updateLibraryItem(ITEM.id, { text: "Body" });
    const detail = await getLibraryItem(ITEM.id);
    await deleteLibraryItem(ITEM.id);
    const calls = mockFetch.mock.calls;
    expect(calls[0][0]).toMatch(new RegExp(`/library/items/${ITEM.id}$`));
    expect(calls[0][1].method).toBe("PATCH");
    expect(JSON.parse(calls[0][1].body)).toEqual({ title: "New" });
    expect(JSON.parse(calls[1][1].body)).toEqual({ text: "Body" });
    expect(detail.preview).toBe("Starter");
    expect(calls[3][1].method).toBe("DELETE");
  });

  it("asks for the full text only when requested (?full=1)", async () => {
    mockFetch
      .mockResolvedValueOnce(ok({ ...ITEM, preview: "S", preview_truncated: true, chunk_count: 0, text: "Starter long" }))
      .mockResolvedValueOnce(ok({ ...ITEM, preview: "S", preview_truncated: true, chunk_count: 0 }));
    const full = await getLibraryItem(ITEM.id, { full: true });
    await getLibraryItem(ITEM.id);
    const calls = mockFetch.mock.calls;
    expect(calls[0][0]).toMatch(new RegExp(`/library/items/${ITEM.id}\\?full=1$`));
    expect(calls[1][0]).toMatch(new RegExp(`/library/items/${ITEM.id}$`));
    expect(full.text).toBe("Starter long");
  });

  it("surfaces the server's reason and status (413 too big)", async () => {
    mockFetch.mockResolvedValueOnce(fail(413, "file is larger than the 10,485,760-byte (10 MB) limit"));
    await expect(
      uploadLibraryFile({ uri: "file:///big.pdf", name: "big.pdf", mimeType: "application/pdf" }),
    ).rejects.toMatchObject({ status: 413, message: expect.stringContaining("10 MB") });
  });

  it("falls back to a generic message when the body has no detail", async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500, json: async () => { throw new Error("no json"); } });
    await expect(listLibrary()).rejects.toMatchObject({ status: 500, message: expect.stringContaining("500") });
  });
});
