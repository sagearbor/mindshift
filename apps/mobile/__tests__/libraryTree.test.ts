import type { LibraryItem, LibraryProject } from "../src/api/library";
import {
  buildTree,
  findNode,
  moveTargets,
  projectCheckState,
  projectItemIds,
  toggleProject,
} from "../src/live/libraryTree";
import { MAX_LIBRARY_SELECTION } from "../src/live/librarySelection";

const T = "2026-10-31T00:00:00Z";
const proj = (id: string, name: string, parent_id: string | null = null): LibraryProject => ({
  id, name, parent_id, created_at: T, updated_at: T,
});
const item = (id: string, project_id?: string | null, status: LibraryItem["status"] = "ready"): LibraryItem => ({
  id, title: id, kind: "note", chars: 1, status, created_at: T, updated_at: T, indexed: false, project_id,
});

const projects = [proj("a", "A"), proj("b", "B", "a"), proj("c", "C", "b"), proj("z", "Z")];
const items = [item("i0"), item("i1", "a"), item("i2", "b"), item("i3", "c"), item("i4", "c", "processing")];

describe("libraryTree", () => {
  it("builds nesting; items without a project (or an unknown one) are top level", () => {
    const t = buildTree(projects, [...items, item("orphan", "gone")]);
    expect(t.roots.map((r) => r.project.id)).toEqual(["a", "z"]);
    expect(t.rootItems.map((i) => i.id)).toEqual(["i0", "orphan"]);
    expect(findNode(t, "c")!.depth).toBe(2);
  });

  it("resolves a project to every READY file under it and its sub-projects", () => {
    const t = buildTree(projects, items);
    expect(projectItemIds(findNode(t, "a")!).sort()).toEqual(["i1", "i2", "i3"]);
    expect(projectItemIds(findNode(t, "z")!)).toEqual([]);
  });

  it("ticks and unticks a project, tracking partial state", () => {
    const t = buildTree(projects, items);
    const a = findNode(t, "a")!;
    expect(projectCheckState(a, [])).toBe("none");
    expect(projectCheckState(a, ["i1"])).toBe("some");
    const on = toggleProject(a, ["i0"]);
    expect(on).toEqual({ ok: true, ids: ["i0", "i1", "i2", "i3"] });
    if (!on.ok) throw new Error();
    expect(projectCheckState(a, on.ids)).toBe("all");
    expect(toggleProject(a, on.ids)).toEqual({ ok: true, ids: ["i0"] });
  });

  it("refuses with a clear message when a project exceeds MAX_LIBRARY_SELECTION", () => {
    const many = Array.from({ length: MAX_LIBRARY_SELECTION + 1 }, (_, n) => item(`m${n}`, "a"));
    const t = buildTree([proj("a", "Big")], many);
    const r = toggleProject(findNode(t, "a")!, []);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.message).toMatch(new RegExp(`"Big".*${MAX_LIBRARY_SELECTION + 1}.*at most ${MAX_LIBRARY_SELECTION}`));
    // Union overflow is a different message and leaves the selection alone.
    const t2 = buildTree([proj("a", "P")], [item("p1", "a"), item("p2", "a")]);
    const full = Array.from({ length: MAX_LIBRARY_SELECTION - 1 }, (_, n) => `x${n}`);
    const r2 = toggleProject(findNode(t2, "a")!, full);
    expect(r2.ok).toBe(false);
    if (!r2.ok) expect(r2.message).toMatch(/would exceed/);
  });

  it("an empty project cannot be ticked", () => {
    const t = buildTree(projects, items);
    expect(toggleProject(findNode(t, "z")!, [])).toMatchObject({ ok: false });
  });

  it("move targets: top level + all projects, excluding the moving project's subtree", () => {
    const t = buildTree(projects, items);
    expect(moveTargets(t).map((m) => m.id)).toEqual([null, "a", "b", "c", "z"]);
    expect(moveTargets(t, "b").map((m) => m.id)).toEqual([null, "a", "z"]);
    expect(moveTargets(t, "b").find((m) => m.id === "a")!.depth).toBe(1);
  });
});
