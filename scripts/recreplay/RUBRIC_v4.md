# Coach-line grading rubric v4 (= v3 + owner calibration + no-hindsight; see the end of the v3 text)

v2 (RUBRIC.md) is untouched. v3 exists because v2 equated "generic" with
"wrong". The owner's view (2026-10-10): "Take a breath" or "let them finish"
can be reasonable things to say. They are reasonable WHEN THE MOMENT CALLS FOR
THEM. v3 therefore scores timing and specificity separately and asks one
overall question: was the wearer better off hearing it?

You grade ONE row's `grader_view` only (never `meta`). Same fields as v2.
Grade the CONTENT the same whether or not it was spoken.

## Scores

**timing (0-3): does the moment genuinely call for the coach to speak at all?**
* 3: A clear trigger just happened or is under way: the wearer is interrupting
  or talking over someone, a long wearer monologue (about 45 s or more) with
  nobody else getting in, heat rising (sarcasm, raised stakes, repeated
  overlap), a dropped question, an explicit floor request by the other person
  ("can I answer?", "let me finish"), a repair worth reinforcing, or the
  wearer trailing off after a ramble.
* 2: Plausible but softer trigger (mild heat, a long-ish wearer turn, a natural
  engagement opening that the wearer has not taken).
* 1: Calm or neutral talk; nothing to fix or reinforce. A line here adds
  nothing but it is not necessarily misleading.
* 0: The line contradicts or misreads the moment: laughter or friendly banter
  treated as conflict, "let them finish" when nobody was cut off AND the wearer
  is simply listening, advice about something already done (re-asking a
  question the wearer just asked), advice landing on the wrong person.

**specificity (0-3): how tied is the wording to what was actually said?**
* 3: Names the concrete thing in the transcript (a number, a phrase, the exact
  question left unanswered).
* 2: Tied to the situation type (monologue, interruption) but not a specific
  fact.
* 1: Generic ("take a breath", "pause and listen"); would fit almost any turn.
* 0: Vague to the point of confusing, or points at the wrong content.
Generic is NOT automatically bad: specificity 1 with timing 3 is a good line.

**helpful (binary 1/0): net helpful vs net distracting.**
Would the wearer, hearing this in their ear right now, be better off than in
silence? 1 = yes or fine (short, kind, usable, harmless-to-useful). 0 = net
distracting or wrong.
Force helpful = 0 whenever any of these is true (always bad):
* it hallucinates a fact, name or event not in the context (e.g. a name that
  nobody said, "you explained the system" when the wearer did not);
* it says an internal label aloud (Speaker A/B/C, S1, "the wearer");
* it speaks AS the wearer or scripts the wearer's reply as if the coach were
  the wearer, or treats another speaker's words as the wearer's ("your lot-rent
  model" when it was the other person's);
* "you" points at the wrong person (the line coaches the other speaker);
* it misreads laughter/banter as conflict or tells the wearer to stop when
  they are only listening.
Also helpful = 0 when timing is 0 or when timing is 1 and the line is a
cliche that interrupts the wearer's listening.

**should_be_silent (boolean, derived):** true when helpful = 0 AND timing <= 1
(the moment did not warrant any speech). False when helpful = 1, or when the
moment really did call for something but this line was the wrong one
(helpful = 0, timing >= 2). Silence is correct in: calm or neutral talk,
during laughter, after backchannels ("mm-hm", "yeah"), when the answered turn
is a fragment, and whenever the line would confuse who is who.

**why**: 20 words or fewer naming the decisive evidence.

## Rules of thumb
1. Roles may be wrong (the phone mislabels turns). Use `context_before` to see
   who really spoke. If roles are guessed (non-ground-truth basis), do not
   penalise role confusion by itself; judge the moment from the text.
2. An explicit floor request by the other person is a valid cut-off moment
   (timing 3 for "let them answer / stop talking"), even if the line is generic.
3. A line that tells the wearer to wait or listen while the other person is
   already speaking and the wearer is silent is NOT a trigger. Timing 1.
4. A phone-truncated turn ("The", "it's") is not an interruption.
5. A cue that arrives while the wearer is already mid-ramble and the cue is a
   question to finish a sentence the transcript merely truncated: timing 1.
6. When torn between two scores, take the lower. Do NOT apply a blanket
   penalty for generic wording: ask first whether the moment called for it.
7. Greek (CONFER) debates: heat is the default, so an interruption or long
   monologue cue is fine, but only if the transcript shows the wearer holding
   the floor or talking over someone.

## Worked examples (real rows)

1. final-H1 sbcsae_SBC042#5, "Take a breath. Say what you need from her right
   now." Wearer says "Since when has she taken her punishment" in a family
   argument. timing 3, specificity 2, helpful 1. Heat rising, concrete ask. v2
   called it generic and silent.
2. final-H0 chime6_S09_t3975#19, "Take a breath. What's the one thing you want
   them to know?" Wearer rambles, gets cut with "No, I'll get there", trails
   off "I don't know". timing 3, spec 2, helpful 1.
3. final-H1 confer_20111031_seq2#4, "Stop. Let them speak - you've filled the
   air for three turns." Wearer has talked over a minute and said they want to
   hear the other's view. timing 3, spec 2, helpful 1. v2 scored grounded 0 for
   the word "turns"; v3 does not (natural English, not a system label).
4. final-H0 chime6_S09_t3975#10, "Slow down. Take a breath before your next
   sentence." Calm dinner banter, wearer said "I had no time". timing 1,
   spec 1, helpful 0, should_be_silent true. Same words as ex. 1, wrong moment.
5. final-H0 chime6_S02_t975#45, "Let Speaker C finish." Internal label.
   helpful 0 always; timing 0; should_be_silent true.
6. final-H1 ami_TS3006a#37, "Stop. Take a breath. This sounds like
   frustration - name it." The others are joking ("You're fired" / "Not yet").
   timing 0, helpful 0.
7. final-H0 ami_TS3006a#24, "Pause. Give the room 3-5 seconds of silence
   first." Wearer just asked "anything more you want to add?". timing 3, spec 2,
   helpful 1. Generic-ish but exactly right.
8. final-H0 chime6_S02_t1305#13, "Let them respond - you've said enough about
   the system." Wearer overlapped the other speaker, so the moment is real
   (timing 2) but the line asserts the wearer "explained a system". helpful 0,
   should_be_silent FALSE (right moment, wrong line).

## v4 additions (owner calibration 2026-10-11 + the no-hindsight view)

You are shown ONLY what had been said when the coach line arrived. A turn
marked "(still speaking)" was cut at that moment; do not guess what came
after. Judge the line as the wearer would have heard it, live.

SILENT IS THE DEFAULT ANSWER. A line is helpful only if the wearer is clearly
better off hearing it than in silence. When torn, helpful = 0.

Owner rules (these override your taste):
* Praise or reinforcement lines ("good, hold that tone", "nice") are never
  helpful, even if the wearer did well.
* Conversational-engagement prompts in calm talk ("ask a detail about the
  trip", "ask what drew them there", "be curious") are never helpful: the
  owner wants silence in calm chat.
* A generic regulate cue ("take a breath", "say what you need") IS acceptable
  when there is real tension or turn pressure in the exchange; do not mark it
  down for being generic then.
* Reading the OTHER person's state ("they sound guarded; ask what would help")
  is worth more than a procedural instruction ("wait 3 seconds").
* A listen/pause/"let them finish" cue is only right if the wearer actually
  talked over someone or held the floor a long time.

## Output (one JSON object per row)
{"id": "...", "timing": 0, "specificity": 0, "helpful": 0,
 "should_be_silent": false, "why": "<=20 words"}

## Aggregation
* helpful_pct = share of SPOKEN lines with helpful = 1.
* should_silent_pct = share of SPOKEN lines with should_be_silent = true.
* mean timing, mean specificity over spoken lines.
* Report n and, where relevant, the comparison with v2 on the same rows.
