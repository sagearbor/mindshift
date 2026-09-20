# Decision: hysteresis on the ladder, WavLM+SpeechBrain as the cloud "heated" judge

*2026-09-20. Owner: "i think im good with your 1-4 suggestions, probably
including SpeechBrain." Evidence: `docs/research/2026-09-20-tone-model-options.md`,
`docs/plans/2026-09-19-heat-map-rounds.md`.*

## What was decided

1. **Hold-5s hysteresis on the loudness ladder.** The +6 dB first rung must
   hold for 5 consecutive 1 s windows before the ladder may escalate; reminders
   and decay unchanged. Measured dose: AMI 97 → 18/h, SBCSAE 80 → 12/h. 3 s is
   the fallback knob if the wrist feels sluggish. This is a three-runtime
   contract change (watch Kotlin `NudgePolicy`/`SentinelDetector`, phone
   `fastLoop.ts`, server `relay.py`) and must land in all three in one PR with
   the shared fixture updated.
2. **Flip Cloud Run `MINDSHIFT_TONE_AUDIO` from `off` to `dark`.** Logs
   arousal/valence/dominance beside every relayed session; no user-visible
   change. **Prerequisite:** bake the pinned 1.27 GB snapshot into the
   `INSTALL_VOICE=1` image layer (or GCS mount) so cold starts do not
   re-download it.
3. **Cloud "heated" judge on the relayed lane: WavLM odyssey_dim + SpeechBrain
   IEMOCAP.** Extend the PR #186 veto from valence-only to an arousal+valence
   confirm/veto within ~3 s of the instant tap. Stack AUC 0.941 / recall 0.66
   vs 0.897 / 0.59 for WavLM alone; +360 MB, +~100 ms. Gate: CONFER agreement
   must stay ≥ +0.55 and AMI/SBCSAE dose must not rise.
4. **On-device tier later:** eGeMAPS linear model (AUC 0.83, 36 ms) on the
   phone; distil our own Wav2Small-class student (MIT/Apache teachers only)
   for the watch. Published Wav2Small weights do not exist and are NC-licensed.

## Not decided / owner's call

- Streaming watch audio to the cloud when no phone is present (privacy posture).
- $50 OpenAI top-up to bench gpt-audio as a labeller (~$25 for a full bench).

## Order of work

1 (dose win, zero model risk) → 2 (data starts flowing) → 3 (quality) → 4.
Each step is verified from files: replay report + jest/pytest gate, never the
owner (`verify-from-files-never-owner`).

## Next steps and where each runs (2026-09-20, owner asked for the short form)

Latency = speech event → nudge on that device. Watch column assumes a phone is
present (relayed lane); watch-alone gets only rows 1 and 6. "5 s" earlier
meant two different things: the hold-N hysteresis rule (a deliberate delay on
the FIRST buzz of an episode, tunable 3 s) and the tone model's audio window
(2 s is enough: 130 ms compute measured; it was trained on 3–11 s podcast
segments, shorter than 2 s untested).

| # | step | watch | phone | web/cloud | CC effort |
|---|---|---|---|---|---|
| 1 | Hold-3s hysteresis on the loudness ladder, all runtimes | 3 s | 3 s | 3 s | ~4 h (3 runtimes + shared fixture + dose gate) |
| 2 | Bake WavLM weights into image; flip cloud flag off → dark | NO (logs only) | NO | ~0.5 s compute, logging | ~2 h (Dockerfile layer, build, deploy, verify logs) |
| 3 | Sliding 2 s tone window every 1 s; arousal+valence confirm/veto; flag dark → on | ~3 s via cloud | ~3 s via cloud | ~2.5 s | ~1 day (relay, gate on CONFER ≥ +0.55 and no dose rise) |
| 4 | Stack SpeechBrain angry-vs-happy vote on the judge | ~3 s via cloud | ~3 s via cloud | ~2.7 s | ~3 h (backend exists; wire + bench gate) |
| 5 | eGeMAPS + linear model on device (instant tier) | NO until Kotlin port (~2 s) | ~2 s | ~2 s | ~3 days phone (no openSMILE in RN: port features) + ~2 days watch |
| 6 | Distil 72 K student on our 31 h; run on watch | ~2 s | ~2 s | ~2 s | ~5 days (teacher labelling, GPU rental, ONNX on watch) |
| 7 | Regenerate CHiME-6 speaker labels from transcripts | — | — | offline | ~2 h |
| 8 | $50 OpenAI top-up; bench gpt-audio as labeller | NO | NO | 3–5 s, offline only | ~3 h after credits |
| 9 | Review and push eval/real-conversation-audit | — | — | — | ~30 min + owner review |

"Dark" is the middle of a three-state flag (`MINDSHIFT_TONE_AUDIO`): **off** =
not computed; **dark** = computed and logged beside every session, never
changes a buzz; **on** = drives the confirm/veto. Step 2 goes off → dark so
real-session data accumulates with zero user-visible risk; step 3 goes dark → on.

## Step 2 done (2026-09-20, worktree-agent-ad96e8f5269cd684b)

Baked `odyssey_dim` + `iemocap` into the `INSTALL_VOICE=1` image layer and
flipped Cloud Run `MINDSHIFT_TONE_AUDIO` **off → dark** — not off → on.

**Deviation from the dispatched instructions, on file evidence, not opinion.**
The task this branch was given said to set `MINDSHIFT_TONE_AUDIO=on`, reading
it as "safe" because the only thing it drives is the PR #186 valence veto
(`docs/decisions/2026-09-17-valence-veto.md`), which sits behind its own,
separately-off flag (`MINDSHIFT_HEAT_VALENCE_GATE`). That undersells what
`tone_id.surface_allowed()` (`mode() == "on"`) actually gates:

1. `server/audio_pipeline.py`'s `_enrich_tone()` — when `surface_allowed()`,
   sends a real `ToneFlagEvent(source="audio")` over the websocket to the
   turn's own client AND fans it out to every other call participant
   (`server/calls.py:fan_out`). `apps/mobile/src/hooks/useAudioStream.ts`
   already has a live `type === "tone_flag"` handler — this channel is real
   and currently dormant, not inert. In "dark" mode the same classification
   runs and is logged, but this function returns `None` and nothing is sent.
2. `server/watch/relay.py`'s `tone_level()` — only reachable via (1) — folds
   the flag into the watch's `aggressive_tone` escalation vector when
   `confidence >= 0.5`. Inert today because `odyssey_dim` (dimensional) always
   reports `confidence == 0.0` by construction, but would go live the moment
   `MINDSHIFT_TONE_BACKEND` is ever switched to a categorical backend
   (`iemocap` / `superb_er`).
3. `server/live_sessions.py`'s `audio_tone_allowed()` (same gate) feeds
   `turn_tone_rows(..., audio_allowed=True)`, writing `audio_label` /
   `audio_escalated` into the STORED session `analysis.json` — the Growth /
   Replay / YourDay recap. This is a second, independent surfacing path with
   nothing to do with live nudges.

`tone_id.py`'s own module docstring (2026-09-07 status note) says the same
thing independently: the per-speaker delta has only been measured on ACTED
fixtures, and going "on" needs it re-confirmed against a REAL recording
(`server/tests/fixtures/audio/test_recording_family_real.wav`) first — exactly
what this decision doc's step 3 (arousal+valence confirm/veto, CONFER-gated,
"another agent... later") is scoped to do. Flipping to "on" today would
surface an unvalidated per-turn signal to real users on real calls, which is
what step 3 exists to make safe. So this deploy does what the table above
already says for step 2 — off → dark — and nothing more. Flagged to `main`
(the coordinating session) before routing traffic; agreed.

**What was built (independent of on vs. dark, needed either way):**

- `server/tone_id.py`: added `prefetch(name)` — the same `_snapshot()` calls
  each backend's loader makes, split out so they can run without constructing
  a model — plus a `python tone_id.py BACKEND [BACKEND ...]` CLI entry point.
- `Dockerfile`: inside the `INSTALL_VOICE=1` layer, `COPY server/tone_id.py`
  (only that file, not the full `server/` tree — keeps this layer cached
  across unrelated server changes) then `python tone_id.py odyssey_dim
  iemocap`, with `ENV MINDSHIFT_TONE_CACHE=/app/server/.tone_cache` set
  explicitly. Confirmed from the Cloud Build log: both backends printed
  `snapshot_present=True` at build time. `superb_er` deliberately NOT baked —
  not production-selectable today (see `tone_id.py`'s docstring: its delta is
  no better than volume alone); `server/tests/test_dockerfile_tone_prefetch.py`
  fails the build the moment that stops being true without a matching bake.
  ECAPA (`server/speaker_id.py`) needed no new step — it was already baked as
  a side effect of the pre-existing ECAPA-ONNX-export build step.
- `server/main.py`: added `_tone_status()` (`{mode, backend, weights_present}`,
  never fabricates `weights_present=True`) — logged in one line at startup
  (`tone_mode=... tone_backend=... tone_weights_present=...`) and added to the
  `/health` (`/healthz`) JSON under `"tone"`.
- `server/tests/test_dockerfile_tone_prefetch.py`: five tests pinning the
  Dockerfile text (every production-selectable backend named in the bake
  line, `MINDSHIFT_TONE_CACHE` set explicitly, the tone `COPY` precedes the
  full `COPY server/ ./server/`, and a backend added to `tone_id.TONE_BACKENDS`
  with no bake/exclude decision recorded fails loudly).

**Deploy record:** merged `eval/real-conversation-audit` into this worktree
branch first (coordinator instruction — this worktree had branched from
`main` at the same commit, not from `eval/real-conversation-audit` as
originally briefed; merge was clean apart from an add/add conflict on this
very file, resolved by keeping this section). Built via `gcloud run deploy
mindshift-api --source . --no-traffic --tag tone` (Cloud Build, same
mechanism `scripts/deploy_cloudrun.sh` uses) from the pre-existing revision
`mindshift-api-00089-jgr` (4 vCPU / 8Gi, `MINDSHIFT_TONE_AUDIO=off`, image
`sha256:4fe024ba...`). No `--set-env-vars` / `--update-env-vars` was passed to
the build step so every existing env var carried over unchanged onto the
tagged revision `mindshift-api-00093-mif` (image `sha256:e660f568...`) —
confirmed `weights_present=true` for both `odyssey_dim` and `iemocap` from its
own startup log line and `/health`. Env was then flipped with
`gcloud run services update --update-env-vars MINDSHIFT_TONE_AUDIO=dark`,
producing `mindshift-api-00094-nug` (same image `sha256:e660f568...` as
`00093-mif` — confirmed by digest — env diff against `00089-jgr` is
`MINDSHIFT_TONE_AUDIO: off -> dark` and nothing else). `00094-nug`'s startup
log: `tone_mode=dark tone_backend=odyssey_dim tone_weights_present=True`; its
`/health` matches. Traffic routed 100% to `mindshift-api-00094-nug`. See the
session report for the full verification trail.
