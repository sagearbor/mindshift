# The nudge chain, audited against 3.8 hours of real conversation — 2026-09-18

The owner stopped being able to test on a device, which was correct: his own
standing rule is *verify from files, never the owner*. This replaces manual
device testing for the questions it was being used for.

Corpus: **12 AMI meetings, 3.8 hours, 48 speakers** across three different
rooms and meeting series. AMI publishes each participant's **own headset**
alongside the meeting, sample-aligned, which gives ground truth for who spoke
when — and therefore for overlap — with no hand annotation, plus a faithful
single-microphone mix by summing the headsets. That mix is exactly what one
phone or one watch in the room hears. CC BY 4.0, kept in gitignored `tmp/`.

Why this corpus and not the ones we had: CREMA-D and RAVDESS are
single-utterance and acted — they answer "does this clip sound angry", which is
a question about clips. Our own TTS scenes turned out to be acoustically flat
(`2026-09-10-heat-rubric-and-buzz-dose.md`). Neither can answer anything about
conversation: turn-taking, crosstalk, or how often the thing goes off.

---

## 1. Dose — how often does the wrist buzz?

`scripts/conversation_audit.py` replays the shipped chain as a faithful port:
`SentinelDetector`'s seeding and baseline rules, the +6/+10/+14 dB ladder,
one-level-per-20 s decay, the pulse train, and the PRD §6 reminder.

| configuration | buzzes / hour | worst meeting |
| --- | --- | --- |
| before 2026-09-10 (pulse on, flat reminder) | **585** | 1,204/h — twenty a minute |
| shipped today (pulse off, reminder backs off) | **88.8** | 138/h |

**Nobody in these recordings is angry.** They are ordinary work meetings.

So this week's two fixes are worth **6.6×** — considerably more than the 3.3×
a two-meeting sample suggested. And 88.8/hour is still about one buzz every
forty seconds, which is not a coaching cue; it is wallpaper. The gate records
95/h as a regression ceiling and 12/h as the product target, with a
known-defect test asserting the gap still exists so that closing it forces the
numbers and this document to move together.

## 2. Does a buzz know *who* was talking?

The headsets give ground truth, so this is answerable for the first time.

| | shipped |
| --- | --- |
| buzzes landing while the **wearer** spoke | 21.7% |
| chance alone (their share of the talking) | 17.1% |

**A buzz carries essentially no information about who was speaking.** The
base-rate comparison is what makes that rigorous — in a four-way meeting almost
any buzz lands while someone else is talking, so the raw percentage alone would
have proved nothing.

This is not a bug in the detector. The watch's own microphone path has **no
identity at all**: loudness on a single mic cannot tell "you got loud" from
"someone near you got loud". The wrist is about as likely to tell its wearer
off for a colleague.

## 3. Does the voiceprint find its owner in a real room?

`scripts/identity_audit.py`: enrol from the wearer's **own clean headset**
(what phone voice training produces — a print built from the mix would already
contain the people we then ask it to reject), then match against the
single-microphone mix. 1,758 scored turns, 16 wearers.

**The model is fine.** AUC 0.891; median cosine 0.59 for the wearer's own turns
against 0.08 for everyone else's. This is a calibration defect, not a reason to
replace ECAPA.

| threshold | finds you | reads someone else as you |
| --- | --- | --- |
| 0.50 | 66.7% | 0.8% |
| 0.55 | 59.6% | 0.5% |
| **0.65 (shipped)** | **35.5%** | **0.1%** |
| 0.70 | 25.6% | 0.0% |

The shipped bar is excellent at the expensive failure — it almost never blames
you for someone else. But it misses **two thirds of your own turns**, and the
reason is arithmetic: the wearer's own *clean* turns have a median cosine of
**0.633**, and the bar sits at **0.65**. It is set just above the middle of the
distribution it is supposed to accept, so it rejects about half of it by
construction.

Splitting the misses by cause (weighing by volume, not by the gap between
medians — an earlier pass of this analysis got that wrong):

- **48% of misses are clean turns** just under the bar → the threshold is too
  strict for a real room.
- **8% are heavily overlapped turns**, which score a median of **0.103** —
  noise, not a marginal call. A single mic hands ECAPA a blend of two voices.
  No threshold rescues that; those spans should be **excluded**, not
  thresholded.

---

## What this adds up to

Two independent failures, both now measured rather than argued about:

- The wrist's own lane knows **neither who is loud** (§2) **nor what loud
  means** — loudness cannot separate anger from enthusiasm, AUC 0.799
  (`2026-09-10`).
- Where identity *is* available, it works well but is **tuned past its own
  data** (§3).

Suggested order, none of it done unilaterally because all of it changes what
fires:

1. **Exclude overlapped spans from matching.** Unambiguous — those embeddings
   are noise and currently produce confident-looking near-zero scores.
2. **Re-tune `MATCH_THRESHOLD` from the distribution**, ~0.55 rather than 0.65.
   Nearly doubles self-recall for 0.4 points of false accept.
3. **Stop the watch nudging on loudness alone** — the §2 finding is fatal for
   that lane on its own.
4. Then re-run this audit; the dose target is 12/h and it is currently 88.8.

## Reproduce

```bash
python scripts/ami_corpus.py --keep-headsets     # ~2 GB into tmp/, CC BY 4.0
python scripts/conversation_audit.py
pip install -r requirements-voice.txt
python scripts/identity_audit.py --meetings 4
pytest tests/test_conversation_dose.py tests/test_identity_in_a_room.py
```

Both gates skip where the corpus or ECAPA is absent, including CI. The dose
*arithmetic* stays pinned in CI against committed fixtures by
`PulseDoseTest`/`ReminderDoseTest`, so a logic regression still fails there.

---

## Postscript: where the valence veto does and does not reach

PR #186 (2026-09-17) shipped the valence veto — Option B from the 09-10
write-up — and it is the right change. This audit does not contradict it, but
it does bound it, and the bound is worth stating before anyone expects the veto
to fix the dose.

`valence_veto` is applied inside `relay.turn_local_to_vector_events`, i.e. on
**phone turns relayed to the watch**. It is absent from both of the paths that
produce the numbers above:

| | valence anywhere? |
| --- | --- |
| `server/watch/vectors.py` — the watch's OWN pcm → `yelling` vectors | no (0 references) |
| `NudgeStateMachine.kt` — the watch's on-device ladder | no (0 references) |

So the 88.8 buzzes/hour in §1 and the no-identity result in §2 are produced by
a lane the veto does not touch. Turning `MINDSHIFT_VALENCE_GATE` on would not
be expected to move either number, and if someone measures it and finds it
did, something is wired differently than this reading suggests.

That is not an argument against the veto — it removes real false buzzes on the
lane it governs, and §3's threshold finding is independent of it. It is an
argument that **Option A is still the outstanding item**: the watch's own lane
knows neither who is loud nor what loud means, and no amount of tuning the
relayed lane changes that.

---

## Correction: §3's first recommendation was wrong, and the real mechanism is worse

The original recommendation above said *"exclude overlapped spans from
matching"*. That does not survive contact with a constraint this repo already
established: **detecting overlap on a single microphone is the thing we proved
we cannot do** — the probe scored 56% against an 80% bar (2026-09-07), which is
why it is off. There is no signal available to exclude on.

Worse, the premise was wrong too. An unmatched turn is not silent. From
`fastLoop.ts`:

```ts
const coachedAsSelf =
  verdict.isSelf === true ||
  (verdict.isSelf === null && fallback !== null && verdict.speaker === fallback);
```

A turn the voiceprint could not match (`isSelf === null`) is still coached as
the user whenever its **diarized label** happens to equal the "you speak first"
convention (`Speaker A`). So failing to match does not fail safe — it falls
through to a label convention.

Now put that next to §3's numbers. In a four-way meeting at the shipped
threshold:

- **64.5%** of the wearer's own turns fail to match,
- **99.9%** of everybody else's turns fail to match.

So very nearly every turn arrives at the fallback, and attribution is decided
almost entirely by whether the diarizer called that turn `Speaker A`. On one
microphone, in a room of four, that is close to arbitrary.

**This is the mechanism behind §2.** The 21.7%-against-a-17.1%-base-rate result
is not a mystery: the voiceprint is mostly abstaining, and the thing actually
deciding "is this you" is a naming convention. The two findings were never
independent.

Revised, in place of the original item 1:

1. **Fix the threshold first** (§3) — it is what pushes attribution onto the
   fallback in the first place. Raising match rate from 35% shrinks the
   fallback's blast radius before anything else is touched.
2. **Then decide what an unmatched turn should do.** Today it guesses. The
   honest alternatives are to stay silent (fewer nudges, some missed) or to
   require a match before coaching at all — which is Option A by another route.
   Either is a product decision; the current behaviour should at least be a
   deliberate one rather than a fallback nobody re-examined.

Overlapped spans need no special handling: they score ~0.10 against every
print, so they simply fail to match, and are then subject to exactly the
fallback problem above — which is the thing to fix, not the overlap.


---

## Acted on: `MATCH_THRESHOLD` 0.65 → 0.60 (2026-09-18)

§3 said the bar was set just above the middle of the distribution it exists to
accept. That is now changed, and the value was chosen from the ceiling of
*"not the same voice"* rather than from the recall curve.

**Why not simply as low as recall keeps improving.** Lower bars keep buying
recall, but three independent measurements agree on where genuine
cross-speaker similarity tops out:

| measurement | ceiling |
| --- | --- |
| the original calibration table's merged/degraded artifacts | 0.558 |
| 91 CREMA-D speakers, 8,190 pairs, 24,570 cross comparisons | **0.567** |
| AMI, impostor turns with no overlap | below 0.60 |

Nothing that is *not* the same voice has been observed above ~0.57. **0.60 sits
just above all three; 0.55 does not** — it would start accepting scores that
have been measured between different people, which is the cardinal sin this
matcher exists to avoid.

**Why the AMI "false accepts" below the bar did not block it.** They are not
voice confusion. Impostor turns scoring ≥0.55 have a median overlap of 0.284
against 0.059 for other-speaker turns generally — the wearer was talking over
them, so their voice genuinely is in that audio. On turns with **no** overlap
the false-accept rate is **zero**.

**Independent confirmation, from fixtures not used to choose the value.** Every
scene in the replay pack improved:

| scene | attribution | self attribution | speakers detected (truth) |
| --- | --- | --- | --- |
| couple_escalation | 11/13 → **12/13** | 6/7 → **7/7** | 3 (3) |
| family3 | 9/15 | 3/5 | 6 → **5** (3) |
| meeting4 | 14/17 → **16/17** | 2/5 → **4/5** | 8 → **5** (4) |

`meeting4`'s "documented ceiling" — *the shout and the apology don't match the
calm print, so the strong nudge is MISSED* — is largely lifted, and speaker
over-fragmentation fell from 8 clusters to 5 against a ground truth of 4.

**What it does not fix.** Self-recall in a real room goes from ~35% to ~48%.
Better, still a defect: the median clean turn scores ~0.63, so half the
distribution remains near the line. `test_but_it_misses_most_of_your_own_turns_
in_a_real_room` keeps asserting the defect until recall clears 55%.

**Reverting is one env var:** `MINDSHIFT_VOICE_MATCH_THRESHOLD=0.65`.
