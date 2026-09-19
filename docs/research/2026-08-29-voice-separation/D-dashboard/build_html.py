"""Assemble the self-contained voice-separability dashboard.

Reads every ``data_<fixture>.json`` in this directory (written by compute.py),
inlines ``dashboard.css`` + ``dashboard.js`` and writes one HTML file
(default: ``<repo>/tmp/voice-dashboard-20260829.html``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORDER = ["maggiano3", "poker6", "family_real", "scene_family3", "scene_meeting4",
         "openai", "gptaudio", "scene_couple"]

TITLES = {
    "poker6": "Poker night — 6 men, ~5 s each (real)",
    "family_real": "Owner + son, alternating 5 s turns (real)",
    "maggiano3": "Restaurant — owner + wife + son (real, private)",
    "scene_family3": "TTS scene — family of 3",
    "scene_meeting4": "TTS scene — meeting of 4",
    "openai": "TTS — 2 OpenAI voices",
    "gptaudio": "TTS — 2 gpt-audio voices",
    "scene_couple": "TTS scene — couple escalation",
}


def main(out: Path) -> None:
    fixtures = []
    for name in ORDER:
        p = HERE / f"data_{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            d["title"] = TITLES.get(name, name)
            fixtures.append(d)
    css = (HERE / "dashboard.css").read_text()
    js = (HERE / "dashboard.js").read_text()
    payload = json.dumps(fixtures, separators=(",", ":")).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice separability dashboard — 2026-08-29</title>
<style>
{css}
</style>
</head>
<body>
<nav id="switcher" class="switcher" aria-label="Recording"></nav>
<header class="page-head">
  <h1>Why these voices should be separable</h1>
  <details class="howto">
    <summary>How to read this</summary>
    <p class="lede">Pick a recording above. The main chart draws the <b>raw measurements</b> over time — pitch, energy,
    spectral centroid / tilt, formants, and the speaker-ID model’s voiceprint on its first principal components — each
    rescaled to 0–1 so they can be overlaid; tick the boxes to add or remove lines, hover or tap for real units. Nothing
    in that chart is coloured by a predicted speaker: the pale bands and the thin top strip are the owner’s own
    listen-through rubric, and can be switched off. Below it, collapsed: what the production diarizer
    (<code>diarize_local.diarize_turns</code>, pinned ECAPA) says today when handed the true turn boundaries; the same
    features split per ground-truth speaker (median ± IQR); the voiceprint scatter and pooled-cosine heatmap; and a
    ranked “what separates these voices” summary.</p>
    <p class="lede small">Frames: 100 ms hop, energy-VAD silence skipped. Voiceprints: 1.5 s windows, 0.25 s hop.
    Separability ratio = spread of speaker medians ÷ typical within-speaker spread (IQR/1.35); “alone” = frame accuracy
    of a nearest-median classifier on that single feature.</p>
  </details>
</header>
<main id="app"></main>
<footer class="page-foot">Generated {json.dumps(str(out.name))[1:-1]} · data in
<code>docs/research/2026-08-29-voice-separation/D-dashboard/</code></footer>
<script id="viz-data" type="application/json">{payload}</script>
<script>
{js}
</script>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(fixtures)} fixtures)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp" / "voice-dashboard-20260829.html")
