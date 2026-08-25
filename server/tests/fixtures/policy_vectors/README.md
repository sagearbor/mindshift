# Policy vectors — cross-language golden tests

Language-neutral JSON fixtures for the small pieces of "brain" that have to
behave identically on more than one runtime: the Python server (reference),
the Wear OS watch (Kotlin, offline fallback), and the phone's realtime path
(TypeScript, future). Each file is self-describing — its top-level `_schema`
key spells out every field — so a port in another language can consume it
without reading the Python driver.

| File | Subject | Python driver |
| --- | --- | --- |
| `nudge_policy.json` | `server/nudge_policy.py` `NudgePolicy` (mirror: `apps/watch/shared/.../NudgeStateMachine.kt`) | `server/tests/test_nudge_policy_vectors.py` |
| `vad_segments.json` | `server/watch/diarize.py` `speech_segments` (energy VAD + merge/drop) | `server/tests/test_vad_vectors.py` |
| `tone_escalation.json` | `server/watch/relay.py` phone turn -> `VectorEvent`s -> `NudgePolicy` (tone + loudness, max-combined) | `server/tests/watch/test_tone_escalation_vectors.py` |
| `pleasantness.json` | `server/pleasantness.py` PRD §6 scoreboard: per-turn score from text tone + prosody + turn balance, per-person current/series, lead (mirror: `apps/mobile/src/live/pleasantness.ts`, `apps/mobile/__tests__/livePleasantness.test.ts`) | `server/tests/test_pleasantness_vectors.py` |

The Kotlin watch consumes `nudge_policy.json` too: `apps/watch/shared/build.gradle.kts`'s
`syncPolicyVectors` task copies this directory into the `:shared` JVM test
resources at build time (never a hand-maintained copy), and
`NudgeStateMachineVectorsTest.kt` replays every case tagged `watch`.

## Conventions (all files)

- Plain field names, no language-specific encodings. Timestamps and
  durations are **seconds as floats** (`1.0`, not `1`). Loudness is dBFS
  (float) or `null` for digital silence.
- `cases` is a list of `{name, description, config, inputs|signal, expected}`.
  `name` is unique and stable — the Python drivers name-check the required
  scenarios, so renaming one is a deliberate act.
- A case starts from a **fresh** instance. "Reset" is a session boundary,
  not an API call: `full_decay_then_fresh_escalation` shows that a channel
  decayed to 0 is indistinguishable from a new policy.
- `expected[i]` is the state/output after `inputs[i]` (one policy call per
  input step). For the VAD file `expected` is the full list of spans for the
  whole synthesized signal.

## `nudge_policy.json` specifics

- `applies_to` per case says which runtimes can run it. Every case runs on
  the server. Cases tagged `watch` are shaped for the Kotlin single-channel
  machine (channel `A` only, sensitivity 1.0, at most one `yelling` event
  per step, each carrying `db_over_baseline`); an empty step is fed as
  `0.0 dB` and `onLocalLoudness` must return `nudges[0].level` or `null`.
  The Python driver asserts `db_over_baseline` maps to the stated `level`
  under the shared `+6/+10/+14 -> 1/2/3` thresholds, so the two runtimes
  can't be reading different inputs.
- Cooldown is **strictly** `elapsed > cooldown_s` — `cooldown_is_strictly_greater_than`
  pins the tie. Rounding of `level * sensitivity` is **half-up**, not
  banker's — `sensitivity_scales_with_half_up_rounding` pins 0.5 -> 1.

## `tone_escalation.json` specifics

- Kept separate from `nudge_policy.json` on purpose: that file's inputs are
  already-levelled vector observations, these are raw phone turns
  (`is_self`, `rms_dbfs`, `text_tone`, optional `tone_flag`) that
  `relay.turn_local_to_vector_events` must first convert. Adding a `tone`
  variant to the other file would have changed its schema under the Kotlin
  consumer.
- Tone rungs are `max(frustration, defensiveness)` -> `>=85/70/55 -> 3/2/1`;
  the loudness lane reuses the `+6/+10/+14` ladder verbatim, and the two
  combine as the policy's per-channel max. `config.baseline_rms_db: null`
  means "cannot measure loudness" (never "baseline 0").
- An empty conversion (`events: []`) does NOT call the policy — the watch's
  own 1 s windows own cooldown de-escalation, so `levels` must hold across
  such a step even when `t` has moved past `cooldown_s`.

## `pleasantness.json` specifics

- Turns are fed IN ORDER to a fresh tracker per case; the loudness baseline
  and the 6-turn balance window are state, so a case's turns can't be
  reordered. `_schema` spells out every dimension's rule; `constants` are
  asserted equal to the module constants on both runtimes.
- Missing inputs are `null` and stay `null` in `expected.dims` — a turn is
  never scored from nothing, and turn balance alone never scores a turn.
- Rounding is half-up (`floor(x + 0.5)`), pinned because Python's `round`
  is banker's and the phone's `Math.round` is not.

## `vad_segments.json` specifics

- PCM is generated, not shipped: `signal` is a list of constant-loudness
  stretches (`{seconds, dbfs}`), each a 150 Hz sine (phase 0 per stretch) at
  the given RMS dBFS, or zeros for `null`. The reference generator is
  `synthesize()` in `server/tests/test_vad_vectors.py`; the schema text
  states the exact formula so a port reproduces it bit-for-bit.
- Boundaries are frame-aligned (0.25 s), so `tolerance_s` is only for float
  accumulation, not detection slack.

## Adding a case

Add it to the JSON with a `description` that names the behaviour and, where
it exists, the test it was derived from; run the Python driver; if the new
case belongs to the plan's required set, add its name to the driver's
`test_coverage_of_required_scenarios`. Never edit `expected` to match a
changed implementation without also updating the mirror(s) — that is the
whole point of the file.
