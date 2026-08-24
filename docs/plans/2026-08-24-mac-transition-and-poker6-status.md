# Mac transition + poker6 diarization status (2026-08-24)

> **Read this if you're a fresh Claude Code session on a newly-cloned checkout.**
> The owner moved from an old Mac (which was having severe, unrelated performance
> problems — see "Old Mac issue" below) to a new one. This doc + the auto-memory
> under `~/.claude/projects/.../memory/` (if that was copied over too) are the
> continuity bridge. If the memory directory is empty/fresh, this file plus
> `docs/research/poker6-sliding-window/README.md` are the full picture.

## First: machine setup this repo needs (none of this comes from `git clone`)

These are all gitignored or machine-local — set them up before doing real work:

- **`.env`** (repo root) — copy from the old Mac via a secure channel (AirDrop,
  not chat/Slack/git). Contains real API keys. There's also a stray `.env2`
  that existed on the old Mac — check if it's actually used before copying it
  too (it wasn't referenced by any script as of this writing).
- **Python venvs** (`tmp/venv-voice`, `tmp/venv-whisper` — not tracked, `tmp/`
  is gitignored): recreate fresh, don't try to copy them.
  `python3 -m venv tmp/venv-voice && tmp/venv-voice/bin/pip install -r <whatever server/ needs>`
  — check `server/` for a requirements file or `pyproject.toml`.
- **`node_modules/`** (root + `apps/mobile/`): `npm install` in both.
- **ffmpeg**: `brew install ffmpeg`.
- **`server/.ecapa_cache/`** (gitignored, ~85MB, the ECAPA-TDNN speaker-embedding
  model): optional to copy — will auto-redownload from HuggingFace on first
  use if absent, just takes a minute.
- **`gcloud` CLI auth**: `gcloud auth login` + `gcloud auth application-default
  login` — needed for `scripts/deploy_cloudrun.sh`.
- **`gh` CLI auth**: `gh auth login` — needed for PR create/merge (this whole
  project's workflow is dispatch-agent → PR → review → squash-merge).
- **EAS/Expo login**: `eas login` — needed for `scripts/ota_publish.sh`
  (**never run raw `eas update`** — see AGENTS.md's Deploying section for why).

## Old Mac issue (context, probably not relevant to you unless it recurs)

On 2026-08-23 the old Mac's load average spiked to 200-500+ (on only 4 cores).
Root cause: 62 orphaned `bun server.ts` processes from the Claude Code
`telegram` plugin (`~/.claude/plugins/cache/claude-plugins-official/telegram/`),
some running for 9+ days, each stuck in a CPU-burning loop with no actual
network activity. The plugin was an old open-source Telegram remote-control
integration the owner had set up that broke when its backing service stopped
being funded — it kept respawning a new stuck instance roughly daily without
ever cleaning up the old one. Fixed by `kill -9`-ing all 62 and setting
`"telegram@claude-plugins-official": false` in `~/.claude/settings.json`
(that settings file is global, not part of this repo — if the owner copied
their whole `~/.claude/` directory to the new Mac, this fix carried over; if
not, and remote-control is timing out again, check for this same plugin
respawning first.

## Branch hygiene note

As of this doc, `main` is the **only** branch on `origin` — 108 stale branches
(69 already squash-merged into `main` months ago and just never deleted, plus
5 more from this week's own work) were confirmed safe and deleted on
2026-08-24. If you're looking for old feature history, it's all already in
`main`'s commit log (squash commits), not on a separate branch — don't go
looking for `feat/whatever` branches, they're gone on purpose, not lost.

## Poker6 6-speaker diarization — current status

**The question:** a real 6-speaker recording (`server/tests/fixtures/audio/
test_recording_poker6_real.wav`) only gets 4 of 6 speakers correctly
identified by the shipped production pipeline (`server/diarize_local.py`).

**What's been tried (full detail in
[`docs/research/poker6-sliding-window/README.md`](../research/poker6-sliding-window/README.md)):**

1. Raising `MAX_SPEAKERS_LOCAL` 4→6 — **shipped** (safe, helps other
   recordings, doesn't fix this specific file — the architecture can only
   relabel whole turns a transcript already segmented, never discover a new
   speaker boundary mid-utterance).
2. A from-scratch sliding-window, voice-only speaker-change detector
   (`server/diarize_sliding_window.py`, merged to `main` but **not wired into
   the live pipeline** — pure research code, self-contained, inert). Final
   result on a clean/idle machine: correctly finds all 6 speakers, passes the
   production accuracy gate automatically, **71% per-turn accuracy** (5/7),
   **52 seconds** for a 30-second clip (cost is solved; a same-code
   measurement under heavy concurrent machine load misleadingly read as ~30
   minutes — that was pure contention noise, not the algorithm).

**Two known remaining failure modes** (see the README for exact timestamps):
one speaker's continuous speech gets incorrectly split into two different
predicted speakers; the last new speaker to appear gets folded into an
earlier speaker's cluster instead of recognized as new.

**Recommended next step, not yet started:** evaluate **pyannote.audio**
(current best-in-class open-source diarizer, with a dedicated overlapping-
speech mode) before further hand-tuning the bespoke sliding-window approach.
It was investigated once already and found **blocked purely on missing
HuggingFace credentials** (the model weights are gated — free to use, but
requires accepting a license on huggingface.co and generating an access
token), not on capability. If the owner has since done that and can provide
an `HF_TOKEN`, that's the highest-leverage next step: run pyannote against
the poker6 fixture plus the 3 safety-check fixtures
(`test_recording_family_real.wav`, `test_recording_openai.wav`,
`test_recording_gptaudio.wav`) and compare honestly against the 71%
sliding-window baseline.

**Also still open:** whichever approach eventually wins needs to be ported
into the real Cloud Run production pipeline and measured there — nothing
diarization-related beyond the shipped `MAX_SPEAKERS_LOCAL` change has
actually been deployed to production yet. Local/Mac timing numbers don't
predict Cloud Run cost or latency.
