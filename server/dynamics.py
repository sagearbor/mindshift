"""Post-hoc conversation-dynamics statistics ("the impartial third chair").

Pure, unit-testable functions that turn a per-turn heat series (scored once, in
batch, by the LLM) into the statistical dynamics of a conversation: talk share,
interruptions, heat stats, Gottman-style repair accounting, emotional coupling
between the two most-active speakers, de-escalation leadership, and trigger
escalations.

Design rules honored here (house style):
* Every function takes plain parallel arrays (speakers/heats/markers/…), never
  Pydantic models — so they are trivially unit-testable and carry no framework
  coupling.
* Nothing is ever fabricated. When the data cannot support a statistic (no
  timestamps, too few turns, no variation) the function returns ``None`` and an
  honest human-readable description, never a padded or guessed number.
* All human-readable ``description`` strings are generated from the numbers in
  Python — there is no second LLM call.
"""

from __future__ import annotations

# The Gottman "Four Horsemen" plus the two constructive markers. This is the
# EXACT vocabulary the LLM is constrained to; anything outside it is dropped
# upstream before these functions ever see a marker list.
HORSEMEN = ("criticism", "contempt", "defensiveness", "stonewalling")

# Thresholds (spec §"Split of labor"). Named so the intent reads at the call
# site and a future product tweak is a one-line change.
SPIKE_DELTA = 20          # is_spike: heat >= own previous + 20
REPAIR_ACCEPT_DROP = 10   # repair accepted: other party then drops >= 10
DEESCALATION_DROP = 15    # a de-escalation event: own heat drops >= 15
DEESCALATION_FOLLOW_DROP = 10  # a "follow": the other party then drops >= 10
COUPLING_MIN_TURNS = 6    # need this many turns from EACH main speaker
COUPLING_MIN_R = 0.3      # below this |r|, we do not name a leader


# ---------------------------------------------------------------------------
# Talk share
# ---------------------------------------------------------------------------

def talk_share(speakers: list[str], char_counts: list[int]) -> dict[str, float]:
    """Fraction of the total characters spoken by each speaker (0..1, rounded).

    Uses characters (not turns) so a speaker who takes few but long turns is
    credited fairly. An all-empty transcript yields 0.0 for every speaker
    rather than a division by zero.
    """
    totals: dict[str, int] = {}
    for sp, count in zip(speakers, char_counts):
        totals[sp] = totals.get(sp, 0) + count
    grand_total = sum(char_counts)
    if grand_total == 0:
        return {sp: 0.0 for sp in totals}
    return {sp: round(chars / grand_total, 4) for sp, chars in totals.items()}


# ---------------------------------------------------------------------------
# Interruptions (timestamp-gated — never fabricated)
# ---------------------------------------------------------------------------

def count_interruptions(
    speakers: list[str],
    starts: list[float | None],
    ends: list[float | None],
) -> dict[str, int] | None:
    """Interruptions per speaker, or ``None`` when timestamps are unavailable.

    An interruption is a turn whose ``start_time`` precedes the *previous*
    turn's ``end_time`` and whose speaker differs — i.e. this speaker began
    talking over the person before them. Attribution is to the interrupter
    (the current turn's speaker).

    Requires a start AND end on EVERY turn: overlap is meaningless if any
    boundary is missing, so the honest answer is ``None`` (never a fabricated
    zero) whenever timestamps are absent.
    """
    if any(s is None for s in starts) or any(e is None for e in ends):
        return None
    counts = {sp: 0 for sp in speakers}
    for i in range(1, len(speakers)):
        if speakers[i] != speakers[i - 1] and starts[i] < ends[i - 1]:
            counts[speakers[i]] += 1
    return counts


# ---------------------------------------------------------------------------
# Per-turn spike flags
# ---------------------------------------------------------------------------

def spike_flags(speakers: list[str], heats: list[int]) -> list[bool]:
    """Per-turn ``is_spike``: heat jumped >= ``SPIKE_DELTA`` over the same
    speaker's OWN previous turn. A speaker's first appearance can never be a
    spike (there is nothing to compare against)."""
    flags: list[bool] = []
    last_heat: dict[str, int] = {}
    for sp, heat in zip(speakers, heats):
        prev = last_heat.get(sp)
        flags.append(prev is not None and heat >= prev + SPIKE_DELTA)
        last_heat[sp] = heat
    return flags


# ---------------------------------------------------------------------------
# Per-speaker heat stats
# ---------------------------------------------------------------------------

def speaker_heat_stats(
    speakers: list[str], heats: list[int],
) -> dict[str, dict[str, float | int]]:
    """avg / peak / peak-turn-index / population-variance of each speaker's
    heat series, plus the speaker's turn count."""
    series: dict[str, list[tuple[int, int]]] = {}
    for i, (sp, heat) in enumerate(zip(speakers, heats)):
        series.setdefault(sp, []).append((i, heat))

    stats: dict[str, dict[str, float | int]] = {}
    for sp, pairs in series.items():
        values = [h for _, h in pairs]
        mean = sum(values) / len(values)
        peak = max(values)
        # First index attaining the peak — stable and predictable for tests.
        peak_index = next(i for i, h in pairs if h == peak)
        variance = sum((h - mean) ** 2 for h in values) / len(values)
        stats[sp] = {
            "turns": len(values),
            "avg_heat": round(mean, 2),
            "peak_heat": peak,
            "peak_turn_index": peak_index,
            "heat_variance": round(variance, 2),
        }
    return stats


# ---------------------------------------------------------------------------
# Gottman repair accounting
# ---------------------------------------------------------------------------

def _previous_heat_for(
    speaker: str, before_index: int, speakers: list[str], heats: list[int],
) -> int | None:
    """Heat of ``speaker``'s most recent turn strictly before ``before_index``,
    or ``None`` if this is their first turn."""
    for k in range(before_index - 1, -1, -1):
        if speakers[k] == speaker:
            return heats[k]
    return None


def count_repairs(
    speakers: list[str], heats: list[int], markers: list[list[str]],
) -> tuple[dict[str, int], dict[str, int]]:
    """``(repair_attempts, repairs_accepted)`` per speaker.

    A repair is *accepted* when, after a ``repair_attempt`` turn by speaker S,
    the OTHER party's very next turn is >= ``REPAIR_ACCEPT_DROP`` cooler than
    that other party's own previous turn — i.e. the olive branch measurably
    lowered their temperature.
    """
    attempts = {sp: 0 for sp in speakers}
    accepted = {sp: 0 for sp in speakers}
    for i, (sp, mk) in enumerate(zip(speakers, markers)):
        if "repair_attempt" not in mk:
            continue
        attempts[sp] += 1
        # The other party's next turn (first turn after i by a different speaker).
        for j in range(i + 1, len(speakers)):
            if speakers[j] == sp:
                continue
            prev = _previous_heat_for(speakers[j], j, speakers, heats)
            if prev is not None and heats[j] <= prev - REPAIR_ACCEPT_DROP:
                accepted[sp] += 1
            break
    return attempts, accepted


def count_horsemen(
    speakers: list[str], markers: list[list[str]],
) -> dict[str, dict[str, int]]:
    """Per-speaker counts of each of the Four Horsemen markers."""
    horsemen = {sp: {h: 0 for h in HORSEMEN} for sp in speakers}
    for sp, mk in zip(speakers, markers):
        for marker in mk:
            if marker in HORSEMEN:
                horsemen[sp][marker] += 1
    return horsemen


# ---------------------------------------------------------------------------
# Emotional coupling (Pearson, lag-0 vs lag-1, LOCF interpolation)
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, or ``None`` when it is undefined (fewer than two
    points, or a constant — zero-variance — series)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / ((sxx * syy) ** 0.5)


def _locf_series(sp: str, speakers: list[str], heats: list[int]) -> list[int | None]:
    """A speaker's heat carried forward onto the full turn axis (last
    observation carried forward). Indices before the speaker's first turn are
    ``None`` (no observation yet — nothing to carry)."""
    out: list[int | None] = []
    current: int | None = None
    for i in range(len(speakers)):
        if speakers[i] == sp:
            current = heats[i]
        out.append(current)
    return out


def _rank_speakers(speakers: list[str]) -> list[tuple[str, int]]:
    """Speakers sorted by turn count (descending), name as a stable tiebreak."""
    counts: dict[str, int] = {}
    for sp in speakers:
        counts[sp] = counts.get(sp, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def compute_coupling(
    speakers: list[str], heats: list[int],
) -> dict[str, object]:
    """Do the two most-active speakers' heat curves move together?

    Correlates their LOCF-interpolated heat series at lag 0 (in step) and lag 1
    (one leading the other). ``strength`` is the maximum-magnitude correlation
    found; ``leader`` is named only when a lag-1 relationship wins AND
    ``|r| >= COUPLING_MIN_R`` — otherwise there is no clear leader. Returns
    honest ``None`` values when either main speaker has fewer than
    ``COUPLING_MIN_TURNS`` turns or the series carry no variation.
    """
    ranked = _rank_speakers(speakers)
    more_than_two = len(ranked) > 2
    a, a_count = ranked[0]
    b, b_count = ranked[1]
    pair_note = (
        f"Measured between the two most active speakers ({a}, {b}). "
        if more_than_two else ""
    )

    if a_count < COUPLING_MIN_TURNS or b_count < COUPLING_MIN_TURNS:
        return {
            "strength": None,
            "leader": None,
            "description": (
                pair_note
                + "Not enough data to measure coupling "
                f"(need at least {COUPLING_MIN_TURNS} turns from each)."
            ),
        }

    series_a = _locf_series(a, speakers, heats)
    series_b = _locf_series(b, speakers, heats)
    shared = [
        i for i in range(len(speakers))
        if series_a[i] is not None and series_b[i] is not None
    ]
    xa = [series_a[i] for i in shared]
    xb = [series_b[i] for i in shared]

    # candidates: (|r|, r, leader). leader None means "in step" (lag 0).
    candidates: list[tuple[float, float, str | None]] = []
    r0 = _pearson(xa, xb)
    if r0 is not None:
        candidates.append((abs(r0), r0, None))
    if len(shared) >= 3:
        # a leads b: a[i] against b[i+1].
        r_ab = _pearson(xa[:-1], xb[1:])
        if r_ab is not None:
            candidates.append((abs(r_ab), r_ab, a))
        # b leads a: b[i] against a[i+1].
        r_ba = _pearson(xb[:-1], xa[1:])
        if r_ba is not None:
            candidates.append((abs(r_ba), r_ba, b))

    if not candidates:
        return {
            "strength": None,
            "leader": None,
            "description": (
                pair_note
                + "Not enough variation in either speaker's intensity to "
                "measure coupling."
            ),
        }

    _, best_r, best_leader = max(candidates, key=lambda c: c[0])
    strength = round(best_r, 2)
    leader = best_leader if abs(best_r) >= COUPLING_MIN_R else None

    direction = "together" if best_r > 0 else "inversely"
    desc = (
        f"{a} and {b}'s emotional intensity moves {direction} "
        f"(r={strength:.2f})."
    )
    desc += f" {leader} tends to lead." if leader else " Neither clearly leads."
    return {"strength": strength, "leader": leader, "description": pair_note + desc}


# ---------------------------------------------------------------------------
# De-escalation leadership
# ---------------------------------------------------------------------------

def compute_deescalation(
    speakers: list[str], heats: list[int],
) -> dict[str, object]:
    """Who cools the conversation first, and does the other party follow?

    A de-escalation event is a turn whose heat is >= ``DEESCALATION_DROP``
    below the same speaker's previous turn. ``who_first`` is the speaker of the
    earliest such event. ``follow_rate`` is the fraction of that speaker's
    de-escalations that were answered, within two turns, by the OTHER party
    dropping >= ``DEESCALATION_FOLLOW_DROP`` against their own previous turn.
    Returns honest ``None`` values when there are no de-escalation events.
    """
    last_heat: dict[str, int] = {}
    events: list[tuple[int, str]] = []  # (turn_index, speaker)
    for i, (sp, heat) in enumerate(zip(speakers, heats)):
        prev = last_heat.get(sp)
        if prev is not None and heat <= prev - DEESCALATION_DROP:
            events.append((i, sp))
        last_heat[sp] = heat

    if not events:
        return {
            "who_first": None,
            "follow_rate": None,
            "description": (
                "No clear de-escalation moments — neither speaker measurably "
                "cooled the conversation."
            ),
        }

    who_first = events[0][1]
    first_party_events = [i for i, sp in events if sp == who_first]
    followed = 0
    for i in first_party_events:
        for j in (i + 1, i + 2):
            if j >= len(speakers) or speakers[j] == who_first:
                continue
            prev = _previous_heat_for(speakers[j], j, speakers, heats)
            if prev is not None and heats[j] <= prev - DEESCALATION_FOLLOW_DROP:
                followed += 1
                break
    follow_rate = round(followed / len(first_party_events), 2)

    desc = f"{who_first} was first to cool things down."
    desc += (
        f" The other party followed within two turns {follow_rate:.0%} "
        "of the time."
    )
    return {
        "who_first": who_first,
        "follow_rate": follow_rate,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Trigger escalations
# ---------------------------------------------------------------------------

def extract_triggers(
    speakers: list[str],
    heats: list[int],
    trigger_phrases: list[str | None],
) -> list[dict[str, object]]:
    """Turn every non-null ``trigger_phrase`` into a trigger record with the
    escalation it provoked.

    ``heat_delta`` measures how much the OTHER party's next turn rose against
    their own previous turn (mirroring ``is_spike``). When the responder is
    speaking for the first time — no previous turn of theirs to compare — the
    delta falls back to their reply minus the trigger turn's heat. Sorted by
    ``heat_delta`` descending so the most provocative surface first.
    """
    out: list[dict[str, object]] = []
    for i, (sp, phrase) in enumerate(zip(speakers, trigger_phrases)):
        if not phrase:
            continue
        heat_delta = 0
        for j in range(i + 1, len(speakers)):
            if speakers[j] == sp:
                continue
            prev = _previous_heat_for(speakers[j], j, speakers, heats)
            base = prev if prev is not None else heats[i]
            heat_delta = heats[j] - base
            break
        out.append({
            "phrase": phrase,
            "speaker": sp,
            "turn_index": i,
            "heat_delta": heat_delta,
        })
    out.sort(key=lambda t: (-t["heat_delta"], t["turn_index"]))
    return out
