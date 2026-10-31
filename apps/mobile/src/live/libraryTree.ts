/**
 * Pure helpers over the library's flat `projects` + `items` lists: build the
 * folder tree, resolve a project to the item ids beneath it (all sub-projects
 * included), and list valid "Move to…" targets. No project_id = top level.
 */
import type { LibraryItem, LibraryProject } from "../api/library";
import { MAX_LIBRARY_SELECTION } from "./librarySelection";

export interface ProjectNode {
  project: LibraryProject;
  depth: number; // top level = 0
  children: ProjectNode[];
  items: LibraryItem[];
}

export interface LibraryTree {
  roots: ProjectNode[];
  /** Items at top level (no project, or a project that no longer exists). */
  rootItems: LibraryItem[];
}

export function buildTree(projects: LibraryProject[], items: LibraryItem[]): LibraryTree {
  const byId = new Map(projects.map((p) => [p.id, p]));
  const nodes = new Map<string, ProjectNode>();
  for (const p of projects) nodes.set(p.id, { project: p, depth: 0, children: [], items: [] });
  const roots: ProjectNode[] = [];
  for (const p of projects) {
    const node = nodes.get(p.id)!;
    const parent = p.parent_id ? nodes.get(p.parent_id) : undefined;
    if (parent) parent.children.push(node);
    else roots.push(node); // top level, or orphaned (parent gone) -> top level
  }
  const setDepth = (n: ProjectNode, d: number, seen: Set<string>) => {
    if (seen.has(n.project.id)) return; // defensive: never loop on bad data
    seen.add(n.project.id);
    n.depth = d;
    n.children.sort((a, b) => a.project.name.localeCompare(b.project.name));
    for (const c of n.children) setDepth(c, d + 1, seen);
  };
  roots.sort((a, b) => a.project.name.localeCompare(b.project.name));
  const seen = new Set<string>();
  for (const r of roots) setDepth(r, 0, seen);
  const rootItems: LibraryItem[] = [];
  for (const it of items) {
    const node = it.project_id ? nodes.get(it.project_id) : undefined;
    if (node && byId.has(node.project.id)) node.items.push(it);
    else rootItems.push(it);
  }
  return { roots, rootItems };
}

export function findNode(tree: LibraryTree, projectId: string): ProjectNode | null {
  const stack = [...tree.roots];
  while (stack.length) {
    const n = stack.pop()!;
    if (n.project.id === projectId) return n;
    stack.push(...n.children);
  }
  return null;
}

/** Every item under the node (itself and all sub-projects), tree order. */
export function nodeItems(node: ProjectNode): LibraryItem[] {
  const out = [...node.items];
  for (const c of node.children) out.push(...nodeItems(c));
  return out;
}

/** Ids a project selects: the *ready* files under it and all sub-projects. */
export function projectItemIds(node: ProjectNode): string[] {
  return nodeItems(node)
    .filter((i) => i.status === "ready")
    .map((i) => i.id);
}

export type ProjectCheck = "none" | "some" | "all";

export function projectCheckState(node: ProjectNode, selectedIds: string[]): ProjectCheck {
  const ids = projectItemIds(node);
  if (ids.length === 0) return "none";
  const n = ids.filter((id) => selectedIds.includes(id)).length;
  return n === 0 ? "none" : n === ids.length ? "all" : "some";
}

export type ToggleResult =
  | { ok: true; ids: string[] }
  | { ok: false; message: string };

/** Tick/untick a project. Ticking unions its ids into the selection; if the
 *  result would exceed MAX_LIBRARY_SELECTION nothing changes and a clear
 *  message comes back. Unticking (when all are on) removes them. */
export function toggleProject(node: ProjectNode, selectedIds: string[]): ToggleResult {
  const ids = projectItemIds(node);
  if (ids.length === 0) {
    return { ok: false, message: `"${node.project.name}" has no ready files to select.` };
  }
  if (projectCheckState(node, selectedIds) === "all") {
    return { ok: true, ids: selectedIds.filter((id) => !ids.includes(id)) };
  }
  const merged = [...selectedIds, ...ids.filter((id) => !selectedIds.includes(id))];
  if (merged.length > MAX_LIBRARY_SELECTION) {
    const own = ids.length;
    return {
      ok: false,
      message:
        own > MAX_LIBRARY_SELECTION
          ? `"${node.project.name}" has ${own} files; a session can use at most ${MAX_LIBRARY_SELECTION}. Tick individual files or a smaller sub-project.`
          : `Adding "${node.project.name}" (${own} files) would exceed ${MAX_LIBRARY_SELECTION} per session. Untick something first.`,
    };
  }
  return { ok: true, ids: merged };
}

export interface MoveTarget {
  id: string | null; // null = top level
  label: string;
  depth: number;
}

/** "Move to…" choices: Top level first, then every project depth-first.
 *  When moving a project, itself and its sub-projects are excluded (a project
 *  can't move into its own subtree). */
export function moveTargets(tree: LibraryTree, movingProjectId?: string): MoveTarget[] {
  const out: MoveTarget[] = [{ id: null, label: "Top level", depth: 0 }];
  const walk = (n: ProjectNode) => {
    if (n.project.id === movingProjectId) return;
    out.push({ id: n.project.id, label: n.project.name, depth: n.depth + 1 });
    for (const c of n.children) walk(c);
  };
  for (const r of tree.roots) walk(r);
  return out;
}
