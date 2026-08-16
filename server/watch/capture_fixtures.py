# Ported from gauge@2157433 server/capture_fixtures.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B7): FIXTURE_ROOT now points at server/tests/watch/fixtures/
# (this repo's watch test tree) instead of gauge's server/tests/fixtures/ --
# no other change. `write_fixture`'s default re-export command line still
# references `scripts/export_capture_fixtures.py`; that script is NOT part
# of this task's ported file list (see the B7 report's export-test
# disposition note) and doesn't exist in this repo yet, so the default
# command is provenance text only until a future task ports the script.
"""Pure, network-free helpers for the capture regression-fixture layout
(gauge Task 17). No CLI, no I/O beyond the local filesystem — the actual
export (fetching captures from a running server) lives in
``scripts/export_capture_fixtures.py``, which is a thin wrapper over
``write_fixture`` below.

The fixture LAYOUT is the stable contract (the web dashboard's export UI was
built against these exact filenames — see gauge's Task 13 brief), not the
label schema, which is deliberately an opaque dict that will churn during
tuning::

    server/tests/watch/fixtures/captures/<capture_id>/
        audio.pcm      # raw PCM16 mono 16 kHz, exactly the bytes
                       #   GET /captures/{id}/audio returned      [GITIGNORED]
        meta.json      # the Capture doc with `labels` removed     [COMMITTED]
        labels.json    # the ground-truth payload verbatim         [COMMITTED]
        README.md      # provenance: source URL, exported_at       [COMMITTED]

``audio.pcm`` is gitignored (multi-MB, re-fetchable) so a fresh clone only
has the committed metadata/labels until someone runs the export script on
the founder's machine — ``CaptureFixture.audio_available`` lets a future
engine test skip honestly (``pytest.skip(...)``) rather than fail or fake
bytes when the clip hasn't been exported locally.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "tests" / "watch" / "fixtures" / "captures"

# Bump whenever write_fixture's output shape changes (new README fields,
# meta/labels transformation, ...) so a fixture's README honestly records
# which exporter version produced it.
EXPORTER_VERSION = "1"


class CaptureFixture(BaseModel):
    capture_id: str
    meta: dict
    labels: dict
    audio_path: Path
    audio_available: bool  # False when audio.pcm is absent (gitignored, not yet exported)

    def audio(self) -> bytes:
        if not self.audio_available:
            raise FileNotFoundError(
                f"capture fixture audio not exported: {self.audio_path}"
            )
        return self.audio_path.read_bytes()


def _readme_text(
    capture_id: str, source_url: str, exported_at: str, reexport_cmd: str | None = None
) -> str:
    cmd = reexport_cmd or (
        "python3 scripts/export_capture_fixtures.py --account default "
        f"--capture {capture_id}"
    )
    return (
        f"# Capture fixture: {capture_id}\n\n"
        f"Exported from: {source_url}\n"
        f"Exported at: {exported_at}\n"
        f"Exporter version: {EXPORTER_VERSION}\n\n"
        "Re-export with:\n"
        f"    {cmd}\n"
    )


def write_fixture(
    root: Path,
    capture: dict,
    audio: bytes | None,
    source_url: str,
    exported_at: str,
    reexport_cmd: str | None = None,
) -> Path:
    """Create/overwrite ``<root>/<capture["id"]>/`` with meta.json,
    labels.json, README.md and (when ``audio`` is not None) audio.pcm.
    Returns the fixture directory. Overwrites in place -- safe to re-run for
    an updated capture (audio replaced, labels re-exported).

    ``reexport_cmd``, when given, is the exact re-export command line
    written into the fixture's README (e.g. reflecting the real
    ``--account``/``--token`` the export script was actually invoked with,
    never a raw token value -- see the script's ``_reexport_flag``). Falls
    back to a generic ``--account default`` example when omitted, e.g. for
    direct/test callers of this pure helper."""
    capture_id = capture["id"]
    d = root / capture_id
    d.mkdir(parents=True, exist_ok=True)

    meta = {k: v for k, v in capture.items() if k != "labels"}
    labels = capture.get("labels", {})

    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    (d / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (d / "README.md").write_text(_readme_text(capture_id, source_url, exported_at, reexport_cmd))

    if audio is not None:
        (d / "audio.pcm").write_bytes(audio)

    return d


def load_fixture(root: Path, capture_id: str) -> CaptureFixture:
    d = root / capture_id
    meta = json.loads((d / "meta.json").read_text())
    labels = json.loads((d / "labels.json").read_text())
    audio_path = d / "audio.pcm"
    return CaptureFixture(
        capture_id=capture_id,
        meta=meta,
        labels=labels,
        audio_path=audio_path,
        audio_available=audio_path.exists(),
    )


def list_fixtures(root: Path) -> list[str]:
    """Capture ids with a meta.json, sorted -- audio may or may not be
    present. Empty list (not an error) when ``root`` doesn't exist."""
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists()
    )
