"""The Play Console answer packs must stay parseable, complete, and provable.

docs/play/README.md states the rule these packs exist to enforce: "Never write
a data-safety answer from the app's description. Write it from the code." The
mechanism behind that rule is the ``proof:`` key — every claim carries a
``file:line`` so the next person can re-check it in ten seconds.

That mechanism rots silently. A file gets renamed, a module moves, and the
proof still *reads* fine in the YAML while pointing at nothing. Then someone
re-derives the answer from the description after all, because the proof was no
help. These tests are the cheap guard against that:

1. Both packs parse as YAML.
2. Both carry every top-level section the browser-agent prompt walks through,
   so a pack can never half-exist.
3. Every path named in any ``proof``-ish key resolves to a real file in this
   repo.

Deliberately NOT checked: that the line numbers are still the right lines. A
line number drifts on every edit above it and pinning them would make this test
a chore rather than a guard; the path is the part whose breakage is silent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAY_DIR = REPO_ROOT / "docs" / "play"

PACKS = {
    "phone": PLAY_DIR / "play-answers-mindshift.yaml",
    "wear": PLAY_DIR / "play-answers-mindshift-wear.yaml",
}

# The sections the browser-agent prompt in docs/play/README.md walks in order,
# plus the schema/provenance bookends. A pack missing any of these is a pack a
# browser agent will improvise its way through.
REQUIRED_SECTIONS = (
    "_schema",
    "app_identity",
    "app_access",
    "ads",
    "content_rating",
    "target_audience",
    "data_safety",
    "government_apps",
    "other_declarations",
    "store_listing",
    "graphics_checklist",
    "notes_for_the_agent",
    "catch_all",
    "_provenance",
)

# Any key whose value is a proof reference. `proof` itself plus the
# `proof_<something>` variants the packs use to prove one specific sibling.
_PROOF_KEY = re.compile(r"^proof(_[a-z0-9_]+)?$")

# "server/watch/auth.py:149-153" / "apps/mobile/app.json:31-40" / a bare path.
# The trailing ":<lines>" is optional; anything after a "#" is a comment the
# YAML loader already stripped for inline comments but not for list items that
# were written as plain strings.
_PROOF_PATH = re.compile(r"^(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)(?::[\d,\-]+)?$")


def _load(pack: Path) -> dict:
    with pack.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_proof_strings(node, where="root"):
    """Yield (location, string) for every string under a proof-ish key."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{where}.{key}"
            if isinstance(key, str) and _PROOF_KEY.match(key):
                yield from _flatten(value, child)
            else:
                yield from _iter_proof_strings(value, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _iter_proof_strings(item, f"{where}[{i}]")


def _flatten(value, where):
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _flatten(item, f"{where}[{i}]")
    elif isinstance(value, dict):
        # e.g. a `- note: "..."` entry inside a proof list — prose, not a path.
        for key, item in value.items():
            yield from _flatten(item, f"{where}.{key}")


def _candidate_paths(raw: str):
    """Split a proof string into the repo paths it names, ignoring prose.

    Proof values are usually one path, sometimes a comma-separated few, and
    occasionally a sentence (a `note:` smuggled into a proof list). Only tokens
    that look like `some/dir/file.ext[:lines]` are treated as paths.
    """
    for token in re.split(r"[,\s]+", raw.strip()):
        token = token.strip().strip("`\"'")
        if not token or "/" not in token:
            continue
        match = _PROOF_PATH.match(token)
        if match:
            yield match.group("path")


@pytest.mark.parametrize("name", sorted(PACKS))
def test_pack_parses(name):
    pack = PACKS[name]
    assert pack.exists(), f"{pack} is missing"
    data = _load(pack)
    assert isinstance(data, dict), f"{pack} did not parse as a mapping"


@pytest.mark.parametrize("name", sorted(PACKS))
def test_pack_has_every_required_section(name):
    data = _load(PACKS[name])
    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    assert not missing, f"{PACKS[name].name} is missing section(s): {missing}"


@pytest.mark.parametrize("name", sorted(PACKS))
def test_pack_declares_its_commit_and_date(name):
    schema = _load(PACKS[name])["_schema"]
    for key in ("verified_against_commit", "last_updated"):
        assert schema.get(key), f"{PACKS[name].name}: _schema.{key} is empty"


@pytest.mark.parametrize("name", sorted(PACKS))
def test_every_proof_path_exists(name):
    data = _load(PACKS[name])
    broken = []
    checked = 0
    for where, raw in _iter_proof_strings(data):
        for path in _candidate_paths(raw):
            checked += 1
            if not (REPO_ROOT / path).exists():
                broken.append(f"{where}: {path}")
    assert checked > 50, f"{PACKS[name].name}: only {checked} proof paths found — the walker is broken"
    assert not broken, (
        f"{PACKS[name].name}: {len(broken)} proof path(s) do not exist in the repo:\n  "
        + "\n  ".join(broken)
    )


@pytest.mark.parametrize("name", sorted(PACKS))
def test_owner_must_supply_is_spelled_exactly(name):
    """The browser agent greps for this literal. A typo silently disarms it."""
    text = PACKS[name].read_text(encoding="utf-8")
    assert "OWNER_MUST_SUPPLY" in text
    for near_miss in ("OWNER MUST SUPPLY", "OWNER_MUST_PROVIDE", "owner_must_supply"):
        assert near_miss not in text, f"{PACKS[name].name} contains {near_miss!r}"


def test_the_two_packs_are_for_different_apps():
    phone = _load(PACKS["phone"])["app_identity"]["package_name"]
    wear = _load(PACKS["wear"])["app_identity"]["package_name"]
    assert phone == "com.sagearbor.mindshift.app"
    assert wear == "com.sagearbor.gauge.wear"


def test_wear_pack_declares_health_data():
    """The watch reads a body sensor and sends the reading to the server.

    This is the one answer the wear pack must not inherit from the phone pack,
    and the one most likely to be "harmonised" away by a well-meaning edit.
    """
    data = _load(PACKS["wear"])
    declared = [t["play_path"] for t in data["data_safety"]["data_types"]]
    assert any("Health" in p for p in declared), (
        "the wear pack must declare Health info — see "
        "apps/watch/wearApp/src/main/kotlin/app/gauge/wear/sensors/HrSource.kt"
    )


def test_readme_links_both_packs():
    readme = (PLAY_DIR / "README.md").read_text(encoding="utf-8")
    for pack in PACKS.values():
        assert pack.name in readme, f"docs/play/README.md does not mention {pack.name}"
