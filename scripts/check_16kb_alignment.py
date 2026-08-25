#!/usr/bin/env python3
"""Fail if any arm64 .so in an AAB/APK is not 16 KB page aligned.

Google Play blocks a release with "Your app does not support 16 KB memory page
sizes" when an arm64 native library has a PT_LOAD segment aligned to less than
16384. The console does not tell you WHICH library, so this does: it parses the
ELF program headers directly (no NDK tooling required) and names the offenders.

    python3 scripts/check_16kb_alignment.py path/to/app.aab

History: vc 36 was rejected for exactly one library out of 25 —
libonnxruntimejsi.so, the JSI shim onnxruntime-react-native compiles from
source. The fix lives in apps/mobile/plugins/withOrtGradle9.js (4).
"""
from __future__ import annotations

import struct
import sys
import zipfile

PAGE = 16384
PT_LOAD = 1


def max_load_align(data: bytes) -> int | None:
    """Largest PT_LOAD p_align in a 64-bit little-endian ELF, or None."""
    if len(data) < 0x40 or data[:4] != b"\x7fELF" or data[4] != 2:
        return None
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38)[0]
    best = 0
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if off + 0x38 > len(data):
            break
        if struct.unpack_from("<I", data, off)[0] == PT_LOAD:
            best = max(best, struct.unpack_from("<Q", data, off + 0x30)[0])
    return best


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    archive = argv[1]

    good: list[str] = []
    bad: list[tuple[str, int | None]] = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            # Only arm64 matters: 16 KB pages are an arm64-only Android feature.
            if not name.endswith(".so") or "arm64-v8a" not in name:
                continue
            align = max_load_align(z.read(name))
            short = name.rsplit("/", 1)[-1]
            (good if align and align >= PAGE else bad).append(
                short if align and align >= PAGE else (short, align)
            )

    total = len(good) + len(bad)
    if total == 0:
        print(f"no arm64 .so found in {archive} — is this an AAB/APK?")
        return 2

    print(f"arm64 libraries: {total}   16 KB-ready: {len(good)}   misaligned: {len(bad)}")
    if not bad:
        print("OK — every arm64 library is 16 KB page aligned.")
        return 0

    print("\nThese block the Play release:")
    for short, align in sorted(bad):
        print(f"  {short:44} p_align={align}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
