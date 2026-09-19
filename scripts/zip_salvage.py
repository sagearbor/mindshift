"""Pull every COMPLETE member out of a truncated zip, without the central directory.

Why this exists: ibug's CONFER server neither honours Range requests nor keeps
a connection open past roughly a gigabyte, so multi-GB fold archives arrive
truncated and `unzip` refuses them outright ("cannot find zipfile directory").
But a zip is a sequence of self-describing local records; everything that
fully arrived is recoverable by walking those records. `zip -FF` does the same
job in principle and took over ten minutes on 1 GB; this does it in seconds.

Members written with a streaming data-descriptor (sizes unknown until after the
data) cannot be bounded without the directory and are skipped, reported as such.

    python scripts/zip_salvage.py <truncated.zip> <out_dir>
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

LOCAL_SIG = b"PK\x03\x04"


def salvage(zip_path: Path, out_dir: Path) -> tuple[int, int, int]:
    data = zip_path.read_bytes()
    n, pos = len(data), 0
    ok = partial = streamed = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        i = data.find(LOCAL_SIG, pos)
        if i < 0 or i + 30 > n:
            break
        (_, flags, method, _, _, _crc, csize, usize, nlen, xlen) = struct.unpack("<HHHHHIIIHH", data[i + 4:i + 30])
        name_start = i + 30
        name = data[name_start:name_start + nlen].decode("utf-8", "replace")
        body = name_start + nlen + xlen
        if flags & 0x08 and csize == 0:
            streamed += 1
            pos = body                      # cannot bound it; skip to the next signature
            continue
        end = body + csize
        if end > n:
            partial += 1                    # the member the connection died inside
            break
        raw = data[body:end]
        try:
            payload = raw if method == 0 else zlib.decompress(raw, -15) if method == 8 else None
        except zlib.error:
            payload = None
        if payload is None or len(payload) != usize:
            partial += 1
            pos = end
            continue
        dest = out_dir / Path(name).name
        if name.endswith("/") or not dest.name:
            pos = end
            continue
        dest.write_bytes(payload)
        ok += 1
        pos = end
    return ok, partial, streamed


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ok, partial, streamed = salvage(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"recovered {ok} complete member(s); {partial} cut off; {streamed} streamed (unbounded, skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
