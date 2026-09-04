# NaturalTurn + conversation-quality plan (2026-09-04)

Owner-approved direction from the CANDOR meetings (2026-09-04). Source
research: the CANDOR corpus paper (Reece et al., Sci Adv 2023,
doi:10.1126/sciadv.adf3197) and its follow-up **NaturalTurn** (Cooney &
Reece, Sci Reports 2025, doi:10.1038/s41598-025-24381-1;
github.com/betterup/natural-turn-transcription, MIT; OSF osf.io/xy35p).

## License constraints (read first)

- **NaturalTurn CODE: MIT** — free to port into the product.
- **CANDOR-derived DATA (transcripts--naturalturn.csv, surveys.csv on OSF):
  CC BY-NC** — research/eval/publication ONLY. Never bake into the shipped
  product or train shipped models on it without a written agreement.
  Owner is pursuing collaboration/co-authorship for broader access.

## The measured problem (why this plan exists)

- Raw ASR/turn-taking treats every listener interjection ("yeah", "uh-huh",
  ~1,000/hour in real conversation) as a full turn → median turn collapses
  to 0.74 s; turn stats, coaching triggers, and airtime are all computed on
  confetti. NaturalTurn's merge+classify recovers 6.58 s median turns and
  the psychology reappears (turn duration ↔ enjoyment r=.14 vs r=.002).
- Brief overlap is NORMAL (35–48% of transitions; no harm to partner
  enjoyment). LONG GAPS are the anti-signal (b = −0.73 on enjoyment).
  Sustained steamrolling is a different phenomenon than brief overlap and
  must be measured separately before any nudge keys on it.

## Workstream 1 — NaturalTurn-style segmentation (phone + server)

Port the algorithm (words + timestamps in, psychological turns out; input
is ASR-agnostic — the paper validated on Deepgram, which we use):

1. Merge consecutive same-speaker utterances separated by ≤ `max_pause`
   (paper default 1.5 s; make it a constant, mirror on both runtimes).
2. Any vocalization by B entirely inside A's turn → non-primary.
3. Classify non-primary: **backchannel** (≤3 words AND matches the cue
   vocabulary — mine the list from their MIT repo, e.g. yeah/mhm/uh-huh/
   wow/right/okay), **secondary turn** (longer, or clause-openers
   "and"/"but"), else **other**.

Where:
- Phone: `apps/mobile/src/live/fastLoop.ts` + `segmenter.ts` — tag turns
  with `kind: "primary" | "backchannel" | "secondary"` in `LocalTurn` and
  `turn_local` (additive wire field; server ignores unknown fields today).
- Server: `server/main.py` turn assembly + `server/diarize_local.py`
  transcript-utterance path; recording analysis stats.

Consumers to fix once tags exist:
- **Coaching:** never coach on a backchannel (kills "the coach answered my
  'yeah'"); backchannels don't reset CoachRepeatGate windows. Also wire
  `CoachRepeatGate` into `fastLoop.finalizeTurn` (long-owed —
  docs/research/2026-08-30-nudge-quality).
- **Speaker-ID:** backchannels never found clusters (extends the existing
  sub-1.5 s guard with a semantic reason) and are excluded from
  `poolSpeakerAudio` blending.
- **Stats:** airtime/turn counts/Growth use primary turns only; keep raw
  counts in dev-mode diagnostics.

Acceptance: fixture-replay on family/poker/maggiano transcripts shows
backchannels tagged (spot-check vs rubric); turn count on a real session
drops to plausible; no coach responses to ≤3-word cue utterances; full
mobile + server suites green.

## Workstream 2 — gap/overlap measurement (dark first, no nudges yet)

- Per session compute: median response gap after partner turns (self),
  gap distribution, brief-overlap rate, and **sustained-overlap seconds**
  (self speaking ≥2 s while partner already speaking — steamrolling proxy).
- Surface in session summary (dev mode first), Growth later. NO nudge on
  brief overlap ever (CANDOR: harmless); a steamroll vector may join
  `nudgePolicy` ONLY after we see real distributions from owner sessions.
- Long response gaps: post-session insight ("you went quiet after X"),
  not a live nudge.

## Workstream 3 — vocal activation vector (instant tier)

Train logistic regression on **RAVDESS (public, no license issue)**:
features = F0 mean/max/SD, log-energy mean/max/SD, voiced/unvoiced
duration — all already computed by `apps/mobile/src/live/prosody.ts`.
Export weights as constants (no runtime model file). Add `activation`
vector to `nudgePolicy.ts` instant tier alongside `yelling` (same
hysteresis machine). Offline eval on our fixtures before enabling;
ship dark behind dev mode if ambiguous.

## Workstream 4 — outcome engine (pre/post mood)

One-tap mood check (CANDOR's single item: positive↔negative right now)
before/after sessions and journal days; store per episode. This is the
therapy-evidence primitive and makes our data directly comparable to
CANDOR's surveys.csv for the joint publication.

## Workstream 5 — research tasks (BY-NC data, tmp/ only)

- Download OSF `surveys.csv` + `transcripts--naturalturn.csv` →
  `tmp/candor/` (gitignored). Mine: backchannel vocabulary + frequencies,
  gap/overlap distributions, turn-length norms → constants with citations.
- Re-score our diarization fixtures with NaturalTurn-merged turns
  (does merging help the transcript-path maggiano score?).

## Phasing

1. WS1 phone tagging + coaching/speaker-ID consumers (biggest UX win).
2. WS5 mining (feeds WS1 vocabulary + WS2 thresholds) — parallel.
3. WS2 dark metrics → summary.
4. WS3 activation vector.
5. WS4 mood check-in (product decision on placement with owner).
6. Server parity for WS1.

Each lands as its own commit(s) with tests; preview OTA per phase; nothing
in this plan requires a native build.
