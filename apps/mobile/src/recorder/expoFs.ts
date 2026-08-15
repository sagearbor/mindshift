import { Directory, File, FileMode, Paths } from "expo-file-system";
import type { RecorderFs } from "./types";

/** Strip a trailing slash so all path joins are `${dir}/${name}`. */
function trimSlash(uri: string): string {
  return uri.endsWith("/") ? uri.slice(0, -1) : uri;
}

/**
 * RecorderFs over expo-file-system's SDK-57 synchronous File/Directory API.
 * Deliberately a thin, mechanical mapping — all recorder logic lives above
 * this seam (and is tested against MemoryFs); anything platform-flaky here
 * degrades to an honest null/throw the callers already handle.
 */
export class ExpoRecorderFs implements RecorderFs {
  documentDirUri(): string {
    return trimSlash(Paths.document.uri);
  }

  ensureDir(dirUri: string): void {
    const dir = new Directory(dirUri);
    if (!dir.exists) dir.create({ intermediates: true });
  }

  exists(uri: string): boolean {
    return new File(uri).exists || new Directory(uri).exists;
  }

  listDirNames(dirUri: string): string[] {
    return new Directory(dirUri)
      .list()
      .filter((entry): entry is Directory => entry instanceof Directory)
      .map((d) => trimSlash(d.uri).split("/").pop() ?? "")
      .filter(Boolean);
  }

  listFileNames(dirUri: string): string[] {
    return new Directory(dirUri)
      .list()
      .filter((entry) => !(entry instanceof Directory))
      .map((f) => trimSlash(f.uri).split("/").pop() ?? "")
      .filter(Boolean);
  }

  readText(fileUri: string): string {
    return new File(fileUri).textSync();
  }

  writeText(fileUri: string, text: string): void {
    new File(fileUri).write(text);
  }

  readBytes(fileUri: string): Uint8Array {
    return new File(fileUri).bytesSync();
  }

  writeBytes(fileUri: string, bytes: Uint8Array): void {
    new File(fileUri).write(bytes);
  }

  appendBytes(fileUri: string, bytes: Uint8Array): void {
    const file = new File(fileUri);
    if (!file.exists) {
      file.write(bytes);
      return;
    }
    file.write(bytes, { append: true });
  }

  writeBytesAt(fileUri: string, offset: number, bytes: Uint8Array): void {
    // Random-access patch via a seekable handle (SDK-57 FileHandle). The
    // handle is closed in finally so a native throw can't leak a descriptor.
    const handle = new File(fileUri).open(FileMode.ReadWrite);
    try {
      handle.offset = offset;
      handle.writeBytes(bytes);
    } finally {
      handle.close();
    }
  }

  move(srcUri: string, destUri: string): void {
    new File(srcUri).moveSync(new File(destUri));
  }

  deleteRecursive(uri: string): void {
    const dir = new Directory(uri);
    if (dir.exists) {
      dir.delete(); // deletes contents too
      return;
    }
    const file = new File(uri);
    if (file.exists) file.delete();
  }

  sizeOf(fileUri: string): number | null {
    try {
      const size = new File(fileUri).size;
      return typeof size === "number" && size > 0 ? size : null;
    } catch {
      return null;
    }
  }

  freeBytes(): number | null {
    try {
      const free = Paths.availableDiskSpace;
      return typeof free === "number" && free > 0 ? free : null;
    } catch {
      return null;
    }
  }
}
