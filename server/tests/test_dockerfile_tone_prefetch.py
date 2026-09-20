"""Guards that the Dockerfile actually BAKES the tone-audio backends this
deployment runs, so a live turn on a cold Cloud Run instance never blocks on
a Hugging Face download (server/tone_id.py, ``docs/decisions/
2026-09-20-heat-judge-plan.md`` "Step 2").

Plain-text assertions on the Dockerfile source, deliberately not a Docker
build (that would need Docker + network + the multi-hundred-MB weights
themselves — far too heavy for a unit test). The point isn't to prove the
download SUCCEEDS here; it's to make it impossible for the Dockerfile and
``tone_id.BACKEND_INFO`` to silently drift apart:

* every backend this codebase currently ships as production-selectable
  (:data:`PRODUCTION_BACKENDS` below) must be named in the Dockerfile's
  ``INSTALL_VOICE`` layer, so a future ``git blame`` on either file makes the
  other's omission obvious;
* a backend added to ``tone_id.TONE_BACKENDS`` that this test doesn't yet
  know how to classify (baked vs. deliberately-not-baked) fails LOUDLY
  rather than silently shipping unbaked — see ``test_new_backend_must_be_classified``.
"""
from __future__ import annotations

from pathlib import Path

import tone_id

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "Dockerfile"

# Backends production actually runs today (MINDSHIFT_TONE_BACKEND defaults to
# odyssey_dim; iemocap is the round-1 model still selectable via that env
# var). `superb_er` is intentionally excluded: tone_id.py's module docstring
# records its per-speaker delta as no better than a plain volume signal, so
# it is not something an operator is expected to flip MINDSHIFT_TONE_BACKEND
# to in this deployment. If that ever changes, add it here AND to the
# Dockerfile's bake line in the same change.
PRODUCTION_BACKENDS = ("odyssey_dim", "iemocap")
NOT_BAKED_BACKENDS = ("superb_er",)


def _dockerfile_text() -> str:
    assert _DOCKERFILE.is_file(), f"expected a Dockerfile at {_DOCKERFILE}"
    return _DOCKERFILE.read_text(encoding="utf-8")


def test_production_backends_are_covered_by_prod_and_excluded_lists():
    """Sanity check on the two lists above, before trusting them below."""
    known = set(tone_id.TONE_BACKENDS)
    classified = set(PRODUCTION_BACKENDS) | set(NOT_BAKED_BACKENDS)
    assert classified == known, (
        f"tone_id.TONE_BACKENDS={known!r} vs classified={classified!r} — "
        "a backend is missing from PRODUCTION_BACKENDS/NOT_BAKED_BACKENDS above"
    )
    assert not (set(PRODUCTION_BACKENDS) & set(NOT_BAKED_BACKENDS))


def test_dockerfile_bakes_every_production_backend():
    """Every backend production can select must be pre-fetched at build time
    inside the INSTALL_VOICE layer — a plain substring check on the
    Dockerfile source is enough to catch a backend that got added to
    PRODUCTION_BACKENDS (or to tone_id.TONE_BACKENDS + this file) without a
    matching bake line."""
    text = _dockerfile_text()
    assert "INSTALL_VOICE" in text
    for name in PRODUCTION_BACKENDS:
        assert name in text, (
            f"backend {name!r} is production-selectable (MINDSHIFT_TONE_BACKEND) "
            "but is not named anywhere in the Dockerfile — it will cold-download "
            f"{tone_id.BACKEND_INFO[name]['source']} on a live turn instead of "
            "shipping baked into the image"
        )


def test_dockerfile_sets_tone_cache_explicitly():
    """MINDSHIFT_TONE_CACHE must be set explicitly in the image so the bake
    step and the runtime loader are provably reading/writing the SAME
    directory, independent of any future change to how server/ is laid out
    in the container."""
    text = _dockerfile_text()
    assert "MINDSHIFT_TONE_CACHE" in text, (
        "Dockerfile must set ENV MINDSHIFT_TONE_CACHE explicitly (see "
        "tone_id.cache_dir()) so the baked snapshot and the runtime load "
        "path are guaranteed to agree"
    )


def test_tone_prefetch_layer_precedes_full_server_copy():
    """The tone weights bake must happen BEFORE `COPY server/ ./server/` so
    the (large, slow) download layer stays cached across changes to any
    OTHER file under server/ — only invalidating when tone_id.py's own pins
    change. A regression here (moving the bake after the full copy) would
    silently make every unrelated server-code change re-pay the download."""
    text = _dockerfile_text()
    tone_copy_idx = text.find("COPY server/tone_id.py")
    full_copy_idx = text.find("COPY server/ ./server/")
    assert tone_copy_idx != -1, "expected a targeted `COPY server/tone_id.py` before the bake RUN step"
    assert full_copy_idx != -1, "expected the full `COPY server/ ./server/` line"
    assert tone_copy_idx < full_copy_idx, (
        "COPY server/tone_id.py must appear BEFORE COPY server/ ./server/ "
        "so the tone weights layer isn't invalidated by unrelated server changes"
    )


def test_new_backend_must_be_classified():
    """If a new backend is ever added to tone_id.TONE_BACKENDS, this test
    fails until someone deliberately puts it into PRODUCTION_BACKENDS (and
    bakes it — see test_dockerfile_bakes_every_production_backend) or
    NOT_BAKED_BACKENDS (and says why, in a comment, the way superb_er does
    above). A backend can never silently ship un-baked."""
    unclassified = set(tone_id.TONE_BACKENDS) - set(PRODUCTION_BACKENDS) - set(NOT_BAKED_BACKENDS)
    assert not unclassified, (
        f"tone_id.TONE_BACKENDS gained {unclassified!r} with no bake decision recorded — "
        "add it to PRODUCTION_BACKENDS (and the Dockerfile) or NOT_BAKED_BACKENDS "
        "(with a reason) in server/tests/test_dockerfile_tone_prefetch.py"
    )
