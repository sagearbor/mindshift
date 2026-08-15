import type { RecorderFs } from "./types";

/**
 * In-memory RecorderFs. The test double for the whole persistence stack —
 * "crash simulation" is simply constructing a fresh store over the same
 * MemoryFs, exactly as a relaunch reads the same disk.
 */
export class MemoryFs implements RecorderFs {
  private files = new Map<string, Uint8Array>();
  private dirs = new Set<string>();
  private free: number | null;

  constructor(opts: { freeBytes?: number | null } = {}) {
    this.free = opts.freeBytes === undefined ? 50e9 : opts.freeBytes;
    this.dirs.add(this.documentDirUri());
  }

  documentDirUri(): string {
    return "file:///doc";
  }

  ensureDir(dirUri: string): void {
    // Create all ancestors too, like { intermediates: true }.
    const parts = dirUri.split("/");
    for (let i = 3; i <= parts.length; i++) {
      this.dirs.add(parts.slice(0, i).join("/"));
    }
  }

  exists(uri: string): boolean {
    return this.files.has(uri) || this.dirs.has(uri);
  }

  listDirNames(dirUri: string): string[] {
    const prefix = `${dirUri}/`;
    const names: string[] = [];
    for (const d of this.dirs) {
      if (d.startsWith(prefix) && !d.slice(prefix.length).includes("/")) {
        names.push(d.slice(prefix.length));
      }
    }
    return names.sort();
  }

  listFileNames(dirUri: string): string[] {
    const prefix = `${dirUri}/`;
    const names: string[] = [];
    for (const f of this.files.keys()) {
      if (f.startsWith(prefix) && !f.slice(prefix.length).includes("/")) {
        names.push(f.slice(prefix.length));
      }
    }
    return names.sort();
  }

  readText(fileUri: string): string {
    return new TextDecoder().decode(this.readBytes(fileUri));
  }

  writeText(fileUri: string, text: string): void {
    this.writeBytes(fileUri, new TextEncoder().encode(text));
  }

  readBytes(fileUri: string): Uint8Array {
    const bytes = this.files.get(fileUri);
    if (!bytes) throw new Error(`MemoryFs: no such file ${fileUri}`);
    return bytes;
  }

  writeBytes(fileUri: string, bytes: Uint8Array): void {
    this.files.set(fileUri, bytes);
  }

  appendBytes(fileUri: string, bytes: Uint8Array): void {
    const existing = this.files.get(fileUri);
    if (!existing) {
      // Copy so a caller-held view can't mutate the "disk" afterwards.
      this.files.set(fileUri, bytes.slice());
      return;
    }
    const joined = new Uint8Array(existing.byteLength + bytes.byteLength);
    joined.set(existing, 0);
    joined.set(bytes, existing.byteLength);
    this.files.set(fileUri, joined);
  }

  writeBytesAt(fileUri: string, offset: number, bytes: Uint8Array): void {
    const existing = this.files.get(fileUri);
    if (!existing) throw new Error(`MemoryFs: no such file ${fileUri}`);
    if (offset + bytes.byteLength > existing.byteLength) {
      throw new Error(`MemoryFs: writeBytesAt past end of ${fileUri}`);
    }
    existing.set(bytes, offset);
  }

  move(srcUri: string, destUri: string): void {
    const bytes = this.readBytes(srcUri);
    this.files.delete(srcUri);
    this.files.set(destUri, bytes);
  }

  deleteRecursive(uri: string): void {
    this.files.delete(uri);
    this.dirs.delete(uri);
    const prefix = `${uri}/`;
    for (const f of [...this.files.keys()]) {
      if (f.startsWith(prefix)) this.files.delete(f);
    }
    for (const d of [...this.dirs]) {
      if (d.startsWith(prefix)) this.dirs.delete(d);
    }
  }

  sizeOf(fileUri: string): number | null {
    const bytes = this.files.get(fileUri);
    return bytes ? bytes.byteLength : null;
  }

  freeBytes(): number | null {
    return this.free;
  }
}
