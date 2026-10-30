# Recording annotation format (`mindshift-annotation/v1`)

The owner records real conversations on his phone and drops each one into
`tmp/recordings/inbox/<name>/` with two small sidecar files. The recording-replay
pipeline (`scripts/recording_replay.py`) turns that into a timeline report of what
the live coach would have done and a re-runnable regression fixture. This page is
the contract for the sidecars.

The strict machine-readable schema is
[`server/tests/fixtures/annotation/mindshift-annotation-v1.schema.json`](../server/tests/fixtures/annotation/mindshift-annotation-v1.schema.json).
The pipeline validates **leniently**. LLM output is messy, so it strips code
fences, repairs a truncated tail, tolerates missing keys and wrong types, and
**reports** every problem in the run report rather than crashing.

## Files per recording

```
tmp/recordings/inbox/<name>/
  <name>.m4a | .mp3 | .wav | .ogg | .webm | .aac | .flac   the audio (any format; ffmpeg -> 16 kHz mono PCM16)
  <name>.notes.txt                                          the owner's notes (below)
  <name>.annotation.json                                    optional: an audio LLM's annotation
  <name>.annotation.<annotator>.json                        optional: more annotators (e.g. .gemini, .gpt)
  <name>.annotation.json.part2                              optional: the continuation of a cut-off reply
```

`tmp/` is gitignored. These are private family recordings and are never committed.

## The notes file

Only the first three lines matter. The parser accepts any order, any case, a
leading bullet, and `:`, `-` or `=` after the key.

```
who: I'm S? / "the man with the low voice"; other voice is my son (12)
setting: dinner at home, son talking about his school day and a quiz
phone: on the table, no earpiece
moments (optional, one per line, mm:ss — what a good coach would have said):
03:10 I interrupted him — "let him finish"
```

* `who:` says which voice is the owner (the wearer). `I'm S2` names an annotation
  speaker directly. A description ("the man with the low voice") is matched
  against each annotation speaker's `voice_description` / `approx_age`. The owner's
  enrolled voiceprint (`tmp/private_fixtures/owner_profile.json`) is a third,
  independent vote. The report shows each vote and whether they agree.
* `setting:` goes to the live session as `session_context`.
* `phone:` is shown in the report (placement affects what the phone hears).
* Moment lines are `mm:ss` or `h:mm:ss`, then an optional dash, then free text.
  They are timed by the owner from a player, so they are **not** re-timed.
* Optional flag lines: `relationship: child|partner|parent|coworker|friend|other`,
  `mode: earpiece|speaker|therapist|room`, `library: <id>, <id>`. Command-line flags
  override them.

## The prompt (paste into Gemini 2.5 Pro / ChatGPT with the audio)

> You are annotating an audio recording of a real conversation for a research tool
> that coaches people to communicate better. Listen to the WHOLE file carefully, then
> output ONE JSON object and nothing else (no prose, no markdown fences).
>
> RULES
> - Timestamps are seconds from the start of the file, with 1 decimal place. Be as
>   accurate as you can; it is fine to be approximate, but never make segments up.
> - Never invent words. If you can't make something out, write [inaudible]; if you're
>   unsure, use your best guess and set "confidence" low.
> - Do NOT guess who people are. Label voices S1, S2, S3… in order of first appearance
>   and describe each voice so a human can match it.
> - A new segment starts whenever the speaker changes or a speaker pauses more than
>   ~1 second. Keep segments under ~15 seconds.
> - Judge emotion from the VOICE (tone, loudness, pace) and from the WORDS separately.
>   They often disagree, e.g. excited vs angry are both loud; record both.
> - Mark overlapping speech (two people talking at once) and short listener sounds
>   ("mm-hm", "yeah").
> - Also mark "coach_moments": points where a helpful, private coach in ONE speaker's
>   ear could have helped that speaker be a more positive presence (e.g. they
>   interrupted, escalated, didn't acknowledge, went silent, or did something well
>   worth reinforcing). Give a nudge of 10 words or fewer. Never script what the
>   other person should say.
>
> OUTPUT FORMAT (exactly these keys; use null where unknown)

```json
{
  "format": "mindshift-annotation/v1",
  "annotator": {"model": "<your model name>", "notes": "<anything about audio quality or your confidence>"},
  "audio": {"duration_s": 0.0, "quality": "clean|ok|noisy|very_noisy", "environment": "<e.g. quiet room, train, restaurant>"},
  "speakers": [
    {"id": "S1", "voice_description": "<e.g. adult male, low voice, fast talker>", "approx_age": "child|teen|adult|older_adult|null", "talk_share_pct": 0}
  ],
  "segments": [
    {
      "start": 0.0, "end": 0.0, "speaker": "S1",
      "text": "<verbatim words>",
      "is_backchannel": false,
      "overlaps_with": [],
      "vocal": {"emotion": "neutral|calm|happy|excited|amused|frustrated|angry|sad|anxious|tired|sarcastic", "intensity": 0, "volume": "quiet|normal|raised|shouting", "pace": "slow|normal|fast"},
      "text_emotion": "neutral|positive|negative|hostile|supportive|questioning",
      "confidence": 0.0
    }
  ],
  "events": [
    {"t": 0.0, "type": "interruption|long_pause|laughter|escalation|de_escalation|repair|apology|question|topic_change|praise", "speakers": ["S1"], "note": "<short>"}
  ],
  "coach_moments": [
    {"t": 0.0, "for_speaker": "S1", "what_happened": "<short>", "ideal_nudge": "<=10 words", "kind": "warning|encouragement", "priority": 1}
  ],
  "summary": {"tone_arc": "<one or two sentences on how the mood moved>", "overall_heat": 0, "peak_heat_t": null}
}
```

> SCALES: intensity and overall_heat are 0 = none, 1 = mild, 2 = clear, 3 = strong.
> priority 1 = most important. talk_share_pct across speakers should sum to about 100.

If the reply is cut off, send: *"Your JSON was cut off. Continue EXACTLY where it
stopped, with no repetition and no commentary, so the two parts concatenate into
valid JSON."* Save the continuation as `<name>.annotation.json.part2` (or append
it to the same file). The pipeline concatenates the parts, and if the result is
still truncated it closes the open brackets and reports what was lost.

## What the pipeline does with an annotation

1. **Lenient parse.** It strips fences and prose around the object, concatenates
   `.partN` files, repairs a truncated tail, coerces numeric strings, and fills
   missing keys with `null`. Each fix is listed under "Inputs" in the report.
2. **Re-timing.** LLM timestamps drift. Each segment's words are aligned to
   Deepgram's word timings (global token alignment). Segments are then re-timed
   from the matched words, and the report gives the share of words matched and the
   median shift. `events[].t` and `coach_moments[].t` move by the same
   piecewise-linear correction.
3. **Speaker mapping.** Annotation `S1…` ids are mapped to Deepgram's
   `Speaker A…` labels by time overlap. The owner's voice comes from `who:` (above).
4. **Ground truth for scoring.** Re-timed segments are the identity truth (who is
   speaking when), `coach_moments` are moment markers, and `vocal` emotion and
   intensity drive the heat lane.
5. **Several annotators.** All are aligned and scored against Deepgram and each
   other. The primary one (`<name>.annotation.json`, else the first alphabetically)
   is used for truth, and the rest are compared in the report.

`vocal.emotion` is also the default stand-in for the phone's on-device tone read
(Gemini Nano is not replayable on a laptop). Pass `--phone-tone neutral` to turn
that off. See the report's "What is and isn't measured" section.
