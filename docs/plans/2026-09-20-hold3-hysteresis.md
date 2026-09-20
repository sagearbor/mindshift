# hold-3s: the loudness ladder's first rung has to HOLD — 2026-09-20

**Shipped, on all three runtimes.** The escalation ladder used to climb off a
single 1 s window that read +6 dB over the speaker's own baseline. It now
requires that first rung to hold for **three consecutive windows** before the
ladder may climb. Nothing else changed: the rungs are still +6/+10/+14 dB →
levels 1/2/3, decay is still one level per 20 s of quiet, and the PRD §6
reminder cadence and its back-off are untouched.

## Why

`scripts/heat_map.py` replays the shipped chain over every recording in every
corpus we hold — 121 recordings, 32.2 hours — and records candidate variants
alongside the shipped number. Two of those columns are the whole argument:

| corpus | n | median dose, shipped ladder | median dose, first rung holds 3 s |
| --- | --- | --- | --- |
| **AMI** office meetings | 12 | 97.0 /h | **46.4 /h** |
| **SBCSAE** everyday American conversation | 60 | 80.2 /h | **36.1 /h** |

Nobody in either corpus is angry. The ladder they were measured on buzzes about
once every forty seconds through an ordinary meeting, and **roughly half of that
is instantaneous threshold crossing** — one loud second, which is a laugh, a
cough, a door, or the first syllable of an ordinary sentence.

That is also the oldest result in the alarm-design literature, which we are
rediscovering: Cvach (2012) puts 80–99% of instantaneous-threshold clinical
alarms in the "not clinically significant" bin, and continuous-arousal work
(O'Dwyer et al. 2017) tracks 3–5 s windows rather than instants for the same
reason. Three seconds is the shortest hold that captures most of the available
reduction; five seconds takes AMI to 18.4 /h and SBCSAE to 11.9 /h but starts
removing real escalations (see *What it costs*).

Aggregate over the full AMI corpus (the number
`server/tests/test_conversation_dose.py` gates on): **88.8 → 38.3 buzzes/hour.**

## What ships

One rule, one semantic, three implementations that the shared contract fixture
holds to the same behaviour:

| runtime | where | knob |
| --- | --- | --- |
| server | `server/nudge_policy.py` — `LoudnessHold`, applied inside `NudgePolicy.on_events` | env `MINDSHIFT_HEAT_HOLD_S` (default 3) |
| phone | `apps/mobile/src/live/nudgePolicy.ts` — same class, same place | `HEAT_HOLD_S` constant |
| watch | `apps/watch/shared/.../NudgeStateMachine.kt` — same class, inside `onLocalLoudness` | `HEAT_HOLD_S` constant; debug-only `GaugePrefs.heatHoldSeconds` (no Settings row) |

The gate is on the **loudness lane only**. `aggressive_tone` reads the words, is
the better signal, and climbs the same channel untouched — which is why the
scripted scenes' nudges are completely unaffected by this change.

A gated-out loud window reads as level 0 *for that call*: it neither escalates
nor refreshes the sustain clock, so cooldown decay runs exactly as it would have
on a quiet window. That is precisely how the measured replay models it
(`conversation_audit.replay`'s `gate` argument), which is what lets the numbers
above transfer to the product.

**Seconds, not calls.** The watch and the server's PCM path observe once per 1 s
window; the phone and `watch/relay.py` observe once per TURN. So an observation
carries how much audio it covers — a 4 s turn that read as loud is four windows
of hold — and every observation is worth at least one window. That last rule is
what makes `hold_s <= 1` byte-identical to the old ladder on every path, which is
how the pre-2026-09-20 expectations in the test suites are pinned.

### The escape hatch

`hold_s` of **0 or 1 reproduces the old ladder exactly**, measured, not asserted:
`test_a_hold_of_zero_or_one_reproduces_the_ladder_it_replaced` re-derives 97.0 /h
and 80.2 /h from the audio at both values and requires them to be equal.

## How it is gated

- `server/tests/fixtures/policy_vectors/nudge_policy.json` **schema v2** adds a
  required `config.hold_s` to every case and six `hold_*` cases: two loud windows
  then quiet never buzzes; the third consecutive window climbs; a quiet window
  makes the next rung earn the hold again; `hold_s` 1 and 0 reproduce the old
  ladder; and the tone lane is not gated. All three runtimes replay the same
  file (Python, Kotlin, TypeScript drivers).
- `server/tests/test_conversation_dose.py` re-measures both corpora from audio
  and asserts the medians above — and, in the case that needs no files at all,
  that the **shipped `NudgePolicy`** produces an escalation stream identical to
  `conversation_audit.sustained` at holds 0/1/3/5. Without that one, the dose
  numbers would be a statement about a script rather than about the product.
- `NudgeStateMachineHoldTest` (watch) and `liveNudgePolicy` / `liveFastLoop`
  (phone) pin the same behaviour on the two device runtimes, including the
  phone's instant haptic tier, which asks the hold before it buzzes.

## What it costs

Recorded because it is real, not rounded away.

- **A 2.4 s shout no longer buzzes on the spot.** `scene_ravdess_pair`'s angry
  turn is +29.8 dB over the speaker's own baseline — the most clear-cut loudness
  event in any fixture we hold — in a single 2.43 s fragment, 0.57 s short of the
  hold. It is still a *hit*: the words reach the same lane through
  `aggressive_tone` about a second later. The case is pinned in
  `apps/mobile/__tests__/replay.nudgeReport.test.ts` with that arithmetic
  spelled out, so it cannot be lost quietly.
- **CONFER (real televised arguments) catches even less.** Broadcast audio is
  level-compressed, so the ladder was already nearly blind to it; the hold makes
  that worse. The known-defect assertion in the dose gate now runs the shipped
  chain, hold included, so the number it reports is the honest one.

Both costs point the same way as round 1b of
`docs/plans/2026-09-19-heat-map-rounds.md`: **no amount of tuning the loudness
lane reaches the 12/h target.** hold-3s is the largest single reduction available
from the existing signal, and it is still 3-4x over target. The next move is a
detector that is not loudness.
