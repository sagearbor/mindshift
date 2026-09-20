# Track 1 — the server-side HEAT JUDGE (2026-09-20). Step 3 of
# docs/decisions/2026-09-20-heat-judge-plan.md.
"""Does the speech actually sound HEATED? A second opinion on the dB ladder.

Why this exists
---------------
The +6/+10/+14 dB loudness ladder (``watch/vectors.py``, mirrored in
``watch/relay.py`` for the phone-relayed lane) measures AROUSAL and nothing
else. Anger and joy are both high-arousal, so it provably cannot tell
shouting at your wife from laughing with your friends: on CREMA-D the shipped
first rung fires on 44.5% of HAPPY speech, and no threshold repairs it
(angry-vs-happy AUC 0.799 from loudness alone).

PR #186 added the cheap half of the fix — ``relay.valence_veto``, a per-TURN
veto on the server's own tone verdict. This module is the other half: a
sliding 2 s window scored every 1 s, smoothed, so the judge has an opinion
about the CURRENT moment rather than about whichever turn the phone last
finalized. It confirms or vetoes the ladder's escalation; it can never
create one.

Shape (chosen from measurement, not taste)
------------------------------------------
* **2 s window.** ``tone_id.classify_pcm`` costs 130 ms on a 2 s clip (CPU,
  measured 2026-09-19). The Odyssey/WavLM head was trained on 3-11 s podcast
  segments; 2 s is the shortest span still inside "several syllables of
  connected speech", and it is what the plan's latency budget (~2.5 s
  speech-event -> nudge) can afford. Under 2 s is untested.
* **1 s hop.** The same cadence as ``SentinelDetector``'s windows, so a
  judge verdict always lines up with the dB window that asked for it. The
  continuous-arousal literature uses 3-5 s windows with a 1 s hop
  (O'Dwyer et al. 2017); we take the short end of the window and the same hop.
* **Exponential smoothing over the last 3 scores.** One window is ~2 s of
  one person's speech and the model is noisy at that length; three windows
  is 4 s of context. Weights are ``ALPHA * (1-ALPHA)**i`` over the most
  recent ``SMOOTHING_N`` scores, renormalised, so the current window still
  dominates (0.571 / 0.286 / 0.143 at ALPHA=0.5) but a single freak score
  cannot flip a verdict on its own.

Verdicts
--------
``confirm`` the tone agrees this is heated. ``veto`` the tone says it is not
(pleasant, or too calm to be an argument). ``unknown`` not enough evidence —
fewer than :data:`MIN_SCORES` windows, or an arousal between the calm floor
and the confirm bar.

The product rule lives in ``relay.py`` and is deliberately asymmetric:
**confirm and unknown both escalate.** A model outage, a cold start, or a
first-two-seconds session must never mute the product — the judge exists to
REMOVE a buzz that the ladder got wrong, never to gate the ladder behind a
model being up. Every failure path here degrades to ``unknown``.

Thresholds and their evidence
-----------------------------
See the module constants below; ``scripts/heat_judge_calibrate.py``
regenerates every number and ``server/tests/test_heat_judge.py`` pins them.

A second vote, later
--------------------
Step 4 of the plan stacks a SpeechBrain angry-vs-happy probability
(``tone_id.angry_vote``) on top of these dimensions — stack AUC 0.941 /
recall 0.66 vs 0.897 / 0.59 for WavLM alone. :class:`HeatJudge` takes its
scorer as a callable returning a dims dict, and :func:`verdict_from` reads
an optional ``angry`` key, so that vote is a scorer change plus one
constant — not a redesign. It is NOT implemented here.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

# --- shape ------------------------------------------------------------------
SAMPLE_RATE = 16000
#: Seconds of audio the model scores at once. 130 ms of CPU at this length.
WINDOW_S = 2.0
#: How often a new window is scored — the SentinelDetector window cadence.
HOP_S = 1.0
#: How many recent scores the smoother looks at.
SMOOTHING_N = 3
#: Exponential weight on the most recent score (see module docstring).
SMOOTHING_ALPHA = 0.5
#: Fewer scores than this and the judge has no opinion (-> ``unknown``).
MIN_SCORES = 2

# --- thresholds -------------------------------------------------------------
# Calibrated by scripts/heat_judge_calibrate.py on 2026-09-20 against two
# corpora pulling in opposite directions:
#   CONFER   ten human raters' continuous conflict rating at 1 Hz over 24
#            TV-debate recordings; a window is HEATED at >= 400/1000
#            (confer_corpus.HEATED_AT). 345 scored 5 s windows, 68 heated.
#   CREMA-D  the SAME two artefacts PR #186 calibrated the valence bar on —
#            tmp/corpora/valence_probe.json (tone_id's own dims) joined to
#            tmp/corpora/heat_rubric_cremad.csv (dB over each speaker's own
#            neutral). 250 clips per emotion; 116 happy and 208 angry clear
#            the +6 dB first rung, which is the only population a veto can
#            reach at all.
# Three gates, all at once:
#   <= 10% of human-rated HEATED windows vetoed  (silencing a real argument)
#   >= 50% of over-rung acted HAPPY vetoed       (the defect this module fixes)
#   <= 10% of over-rung acted ANGRY vetoed       (PR #186's own recall gate)
# 77 of 1,140 threshold pairs clear all three. At the pinned pair:
#   CONFER heated  8.8% vetoed / 89.7% confirmed;  calm windows 26.7% confirmed
#   CREMA-D over the rung  happy 56.9% vetoed, angry 9.6% vetoed (90.4% kept,
#                          against PR #186's valence-only 60.3% / 9.1%)
# Pinned by server/tests/test_heat_judge.py::test_the_thresholds_are_calibrated.

#: Above this smoothed valence the speech reads as PLEASANT and the loudness
#: nudge is vetoed. Same number and same scale as ``relay.VALENCE_VETO_MAX``
#: (PR #186, leave-one-speaker-out over 85 held-out CREMA-D speakers) — kept
#: as its own constant so this module does not import the relay (the relay
#: imports this one), pinned equal by a test.
VALENCE_VETO_MAX = 0.48
#: Below this smoothed arousal the speech is too FLAT to be an argument:
#: loud by the microphone, but delivered without activation. Vetoes.
#: 0.40 is the highest floor that still keeps >= 90% of the over-rung acted
#: anger; it buys +0.7 points of happy veto over the valence bar alone
#: (56.2% -> 56.9%) for +0.5 points of anger lost. A modest knob, kept
#: because the loud-but-flat case is real and costs almost nothing to catch.
CALM_AROUSAL_FLOOR = 0.40
#: At or above this smoothed arousal the judge CONFIRMS — and confirm
#: OUTRANKS the valence veto (see :func:`verdict_from`). 0.78 sits above the
#: 90th percentile of over-rung acted HAPPY arousal (0.756) and far below the
#: mean of CONFER's human-rated heated windows (0.839 +/- 0.048), which is
#: exactly the separation the bar is for. Between the floor and this bar the
#: verdict is ``unknown``, which escalates exactly like ``confirm`` does —
#: but ``confirm`` is now also the gate on SURFACING a tone flag to the
#: phone and on labelling a stored turn ``audio_escalated``, so it is
#: deliberately a high bar: roughly a quarter of over-rung acted anger
#: reaches it.
CONFIRM_AROUSAL = 0.78

VERDICT_CONFIRM = "confirm"
VERDICT_VETO = "veto"
VERDICT_UNKNOWN = "unknown"
#: How a verdict is written into the persisted per-second series.
#: ``LiveSession.series`` is typed ``dict[str, list[float]]`` — widening it
#: to carry strings would be a wire-model change for three words, and the
#: ordering veto < unknown < confirm is meaningful anyway: it is "how much
#: the tone model agrees the moment was heated", which plots.
VERDICT_CODES = {VERDICT_VETO: -1.0, VERDICT_UNKNOWN: 0.0, VERDICT_CONFIRM: 1.0}
#: Where the verdict rides on a ``ToneFlagEvent``. ``scores`` is already a
#: free-form ``dict[str, float]`` on the wire, so the verdict travels with
#: the reading it belongs to — through the relay, into the stored session,
#: and out the other side where ``live_sessions.turn_tone_rows`` decides
#: whether the turn may be labelled ``audio_escalated``. A flag with no such
#: key was never judged, and an unjudged flag is never an escalation.
HEAT_VERDICT_KEY = "heat_verdict"

# --- relay wiring -----------------------------------------------------------
#: The ladder rung that may WAIT for the judge: the first tap of an episode,
#: level 0 -> 1. Higher rungs are vetoed if a verdict is already in hand but
#: are never delayed — by rung 2 the wearer is 10 dB over their own baseline
#: and a late buzz is worse than an imperfect one.
FIRST_RUNG = 1
#: How long a first-rung escalation waits for a verdict before going ahead
#: anyway. One hop plus one model call, rounded up.
JUDGE_WAIT_S = 2.0
#: Poll interval while waiting. The scorer runs on another thread (and
#: possibly another loop); this is the cheapest correct way to notice it.
JUDGE_POLL_S = 0.05
#: Per-second series kept for a live session: 6 h at 1 Hz. Bounded because
#: this dict outlives any single turn and an all-day companion socket must
#: not grow without limit.
MAX_SERIES = 6 * 3600


@dataclass(frozen=True)
class HeatVerdict:
    """One judgement, plus everything the log line and the series need."""

    verdict: str
    arousal: float | None = None
    valence: float | None = None
    dominance: float | None = None
    n_scores: int = 0
    latency_ms: float | None = None
    reason: str = ""
    #: True only in ``MINDSHIFT_TONE_AUDIO=on``. In ``dark`` the verdict is
    #: computed and logged but must never change what the wearer feels.
    acts: bool = False

    def suppresses_escalation(self) -> bool:
        """The single question ``relay.py`` asks of this object."""
        return self.acts and self.verdict == VERDICT_VETO

    def log_fields(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "arousal": None if self.arousal is None else round(self.arousal, 4),
            "valence": None if self.valence is None else round(self.valence, 4),
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "n_scores": self.n_scores,
            "reason": self.reason,
            "acts": self.acts,
        }


UNKNOWN = HeatVerdict(VERDICT_UNKNOWN, reason="no scores")


# ------------------------------------------------------------------- pure --

def smooth(values: Iterable[float], *, alpha: float = SMOOTHING_ALPHA, n: int = SMOOTHING_N) -> float | None:
    """Exponentially weighted mean of the last ``n`` values, newest LAST.

    Weights ``alpha*(1-alpha)**i`` over i = 0 (newest) .. n-1, renormalised so
    a short history (the first two windows of a session) is still an average
    and not a fraction of one. Pure; the tests drive it directly.
    """
    tail = list(values)[-n:]
    if not tail:
        return None
    weights = [alpha * (1.0 - alpha) ** i for i in range(len(tail))]  # newest first
    total = sum(weights)
    return float(sum(w * v for w, v in zip(weights, reversed(tail))) / total)


def verdict_from(
    scores: list[dict[str, float]],
    *,
    latency_ms: float | None = None,
    acts: bool = False,
) -> HeatVerdict:
    """The rules, as a pure function of the recent raw dims. Order matters.

    1. Fewer than :data:`MIN_SCORES` windows -> ``unknown``. The judge has
       not heard enough to take a buzz away from anyone.
    2. Smoothed arousal at or above :data:`CONFIRM_AROUSAL` -> ``confirm``,
       **before** the valence veto is considered.
    3. Smoothed valence above :data:`VALENCE_VETO_MAX` -> ``veto``: loud and
       PLEASANT is the laughing-not-shouting case.
    4. Smoothed arousal below :data:`CALM_AROUSAL_FLOOR` -> ``veto``: loud
       but FLAT. A raised voice with no activation behind it is a projected
       voice, not an argument.
    5. Anything between the floor and the bar -> ``unknown`` (escalates).

    **Why confirm outranks the veto.** The obvious order — veto first, then
    confirm only when valence is also below the bar — was measured and
    rejected: PR #186's inherited 0.48 valence bar, applied that way, vetoes
    **20.6%** of CONFER's human-rated heated windows, and no choice of the
    two arousal knobs can move that number, because the veto never consults
    them. Letting a window at CONFER-heated activation (arousal 0.839 mean,
    against 0.756 for the 90th percentile of over-rung acted happy) escape a
    borderline pleasantness reading takes it to **8.8%** and costs 0.7 points
    of the happy veto. The corpora say the same thing the product does: a
    voice this activated is not someone laughing.
    """
    n = len(scores)
    if n < MIN_SCORES:
        return HeatVerdict(
            VERDICT_UNKNOWN,
            arousal=smooth([s["arousal"] for s in scores]) if scores else None,
            valence=smooth([s["valence"] for s in scores]) if scores else None,
            n_scores=n, latency_ms=latency_ms, acts=acts,
            reason=f"only {n} score(s), need {MIN_SCORES}",
        )
    arousal = smooth([float(s.get("arousal", 0.0)) for s in scores])
    valence = smooth([float(s.get("valence", 0.0)) for s in scores])
    dominance = smooth([float(s["dominance"]) for s in scores if "dominance" in s])
    common = dict(
        arousal=arousal, valence=valence, dominance=dominance,
        n_scores=n, latency_ms=latency_ms, acts=acts,
    )
    if arousal is not None and arousal >= CONFIRM_AROUSAL:
        return HeatVerdict(VERDICT_CONFIRM, reason=f"arousal {arousal:.3f} >= {CONFIRM_AROUSAL}", **common)
    if valence is not None and valence > VALENCE_VETO_MAX:
        return HeatVerdict(VERDICT_VETO, reason=f"valence {valence:.3f} > {VALENCE_VETO_MAX}", **common)
    if arousal is not None and arousal < CALM_AROUSAL_FLOOR:
        return HeatVerdict(VERDICT_VETO, reason=f"arousal {arousal:.3f} < {CALM_AROUSAL_FLOOR}", **common)
    return HeatVerdict(VERDICT_UNKNOWN, reason=f"arousal {arousal:.3f} in the undecided band", **common)


# ------------------------------------------------------------------ judge --

def _default_scorer(pcm: np.ndarray, sr: int) -> dict[str, float]:
    """``tone_id.classify_pcm`` -> the dims dict this module speaks.

    Imported lazily and per call: ``tone_id`` pulls in torch on first use
    (seconds, ~1.3 GB), and importing this module must stay free for every
    server that runs with the flag off.
    """
    import tone_id  # noqa: PLC0415  (deliberate: see docstring)

    result = tone_id.classify_pcm(pcm, sr)
    scores = result.get("scores") or {}
    return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}


Scorer = Callable[[np.ndarray, int], dict[str, float]]


@dataclass
class HeatJudge:
    """A rolling 2 s PCM buffer for ONE live session, scored once per second.

    Not thread-safe by accident: ``feed``/``score_now`` are called from the
    phone pipeline's loop (and its worker thread) while ``verdict`` is read
    from the watch socket's loop, so the mutable state is guarded by a lock
    and every reader gets a snapshot.
    """

    sample_rate: int = SAMPLE_RATE
    scorer: Scorer = _default_scorer
    clock: Callable[[], float] = time.monotonic
    window_s: float = WINDOW_S
    hop_s: float = HOP_S

    _buf: bytearray = field(default_factory=bytearray, repr=False)
    _scores: deque[dict[str, float]] = field(default_factory=lambda: deque(maxlen=SMOOTHING_N), repr=False)
    #: Bytes of audio that have arrived since the last score was taken.
    _since_hop: int = 0
    #: Total audio ever fed, in bytes — the session-relative "now".
    _total_bytes: int = 0
    #: Wall-clock stamp of the newest sample in the buffer, for latency.
    _window_end_at: float | None = None
    _last_latency_ms: float | None = None
    _failed_once: bool = False
    #: One score at a time. A model slower than the hop must drop windows,
    #: never queue them — a 4 s-old verdict is worse than no verdict.
    scoring: bool = False
    arousal_series: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SERIES), repr=False)
    valence_series: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SERIES), repr=False)
    judge_series: deque[float] = field(default_factory=lambda: deque(maxlen=MAX_SERIES), repr=False)
    #: RE-entrant on purpose: ``score_now`` reads the buffer under the lock
    #: and, on the "nothing to score yet" path, returns ``verdict()`` — which
    #: takes the same lock to snapshot the scores. A plain Lock deadlocks the
    #: whole session there, and it deadlocks on the FIRST second of every
    #: session, silently, with no error to see.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ---- ingest ----
    @property
    def _window_bytes(self) -> int:
        return int(self.window_s * self.sample_rate) * 2

    @property
    def _hop_bytes(self) -> int:
        return int(self.hop_s * self.sample_rate) * 2

    def feed(self, frame: bytes, sample_rate: int | None = None) -> bool:
        """Add one PCM16 frame. Returns True when a score is now DUE.

        Due means: the buffer holds a full window AND a full hop of new audio
        has arrived since the last score. A frame at the wrong sample rate is
        dropped rather than resampled — the phone's contract is 16 kHz mono
        int16 and a silent resample would hide a real wire bug.
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            return False
        if not frame:
            return False
        with self._lock:
            self._buf += frame
            self._total_bytes += len(frame)
            self._since_hop += len(frame)
            excess = len(self._buf) - self._window_bytes
            if excess > 0:
                del self._buf[: excess - (excess % 2)]
            self._window_end_at = self.clock()
            return len(self._buf) >= self._window_bytes and self._since_hop >= self._hop_bytes

    def score_now(self) -> HeatVerdict:
        """Score the current window (BLOCKING, ~130 ms) and fold it in.

        Never raises. A model that is missing, cold, or broken produces an
        ``unknown`` verdict and one WARNING for the life of the session — a
        line per second would drown the log and change nothing.
        """
        with self._lock:
            if len(self._buf) < self._window_bytes:
                return self.verdict()
            window = bytes(self._buf)
            started_from = self._window_end_at
            self._since_hop = 0
        pcm = np.frombuffer(window, dtype="<i2").astype(np.float32)
        try:
            dims = self.scorer(pcm, self.sample_rate)
        except Exception:
            if not self._failed_once:
                self._failed_once = True
                logger.warning(
                    "heat judge: scorer failed; degrading to %r for this session",
                    VERDICT_UNKNOWN, exc_info=True,
                )
            with self._lock:
                self._scores.clear()
            return self.verdict()
        if not isinstance(dims, dict) or "arousal" not in dims or "valence" not in dims:
            # A categorical backend (superb_er / iemocap) has no valence axis.
            # That is a configuration answer, not a failure: no opinion.
            if not self._failed_once:
                self._failed_once = True
                logger.warning("heat judge: scorer returned no arousal/valence dims (%r); no opinion", dims)
            return self.verdict()
        now = self.clock()
        latency_ms = None if started_from is None else max(0.0, (now - started_from) * 1000.0)
        with self._lock:
            self._scores.append({k: float(v) for k, v in dims.items()})
            self._last_latency_ms = latency_ms
            self.arousal_series.append(round(float(dims["arousal"]), 4))
            self.valence_series.append(round(float(dims["valence"]), 4))
        v = self.verdict()
        with self._lock:
            self.judge_series.append(VERDICT_CODES[v.verdict])
        return v

    # ---- read ----
    def verdict(self, *, acts: bool = False) -> HeatVerdict:
        with self._lock:
            scores = list(self._scores)
            latency = self._last_latency_ms
        return verdict_from(scores, latency_ms=latency, acts=acts)

    def series(self) -> dict[str, list]:
        with self._lock:
            return {
                "arousal": list(self.arousal_series),
                "valence": list(self.valence_series),
                "judge": list(self.judge_series),
            }


# --------------------------------------------------------------- registry --
# One judge per account with a live relayed lane, exactly like relay.py's
# _registry. Keyed by uid because that is the only identifier the phone
# pipeline and the watch socket share.

_judges: dict[str, HeatJudge] = {}
_judges_lock = threading.Lock()
_score_tasks: set[asyncio.Future] = set()
#: Series handed over by ``detach`` and not yet persisted. Bounded: a leak
#: here would be one entry per account that ever streamed audio.
RECENT_SERIES_MAX = 64
_recent_series: dict[str, dict[str, list]] = {}
_TONE_ID = None
_TONE_ID_TRIED = False
_WEIGHTS_WARNED = False


def _tone_id():
    """``tone_id`` if this deployment has it, else ``None`` — imported once.

    Deliberately NOT a module-level import: ``relay.py`` imports this module
    on every server start, and the import must stay free for a deployment
    that ships without the optional model deps. ``tone_id`` itself keeps
    torch behind function-level imports, so this costs nothing but a lookup
    after the first call.
    """
    global _TONE_ID, _TONE_ID_TRIED
    if not _TONE_ID_TRIED:
        _TONE_ID_TRIED = True
        try:
            import tone_id  # noqa: PLC0415
        except Exception:  # pragma: no cover — depends on the deployment
            tone_id = None
        _TONE_ID = tone_id
    return _TONE_ID


def _mode() -> str:
    """``off`` | ``dark`` | ``on``, from ``tone_id`` — fail SAFE to ``off``.

    ``tone_id`` is an optional import on some deployments (no torch); a
    server without it simply has no judge, which is the shipped behaviour.
    """
    tone_id = _tone_id()
    if tone_id is None:
        return "off"
    try:
        return tone_id.mode()
    except Exception:
        return "off"


def judge_for(uid: str, *, create: bool = False, **kwargs: Any) -> HeatJudge | None:
    if not uid:
        return None
    with _judges_lock:
        judge = _judges.get(uid)
        if judge is None and create:
            judge = HeatJudge(**kwargs)
            _judges[uid] = judge
        return judge


def weights_present() -> bool:
    """Is the pinned tone snapshot on this machine's disk?

    A filesystem check only — no torch, no download. The judge scores once
    per SECOND, so a server that would have to fetch 1.3 GB first must have
    no judge at all rather than a queue of stalled windows: plan step 2 bakes
    the snapshot into the image precisely so this is true in production.
    Logged once, because "no weights" is a deployment fact, not an event.
    """
    global _WEIGHTS_WARNED
    tone_id = _tone_id()
    check = getattr(tone_id, "snapshot_present", None)
    if not callable(check):
        return False
    try:
        present = bool(check())
    except Exception:
        present = False
    if not present and not _WEIGHTS_WARNED:
        _WEIGHTS_WARNED = True
        logger.warning("heat judge: tone weights not present on disk; no judge will run")
    return present


def attach(uid: str, **kwargs: Any) -> HeatJudge | None:
    """Start judging ``uid``'s relayed lane.

    Called by ``relay.register_live_session``: the judge exists to confirm or
    veto an escalation on a WRIST, so it starts when a wrist is actually
    live and not merely because audio is streaming. That scoping is what
    keeps a phone-only session (and every test that streams frames without a
    watch) from spinning up a 1.3 GB model for a verdict nobody can act on.

    ``None`` — no judge — in ``off`` mode, or when the weights are not on
    disk and the caller has not supplied its own ``scorer``.
    """
    if _mode() == "off":
        return None
    if "scorer" not in kwargs and not weights_present():
        return None
    return judge_for(uid, create=True, **kwargs)


def detach(uid: str) -> dict[str, list] | None:
    """Stop judging ``uid`` and hand back its per-second series, if any.

    The series is kept in a small bounded cache afterwards because the two
    sockets end independently: the phone's audio session (which feeds the
    judge) and the watch's live session (which PERSISTS the series) can
    close in either order, and the watch must still be able to save what
    was measured when the phone hung up first.
    """
    with _judges_lock:
        judge = _judges.pop(uid, None)
        if judge is None:
            return None
        series = judge.series()
        _recent_series[uid] = series
        while len(_recent_series) > RECENT_SERIES_MAX:
            _recent_series.pop(next(iter(_recent_series)))
    return series


def series_for(uid: str) -> dict[str, list] | None:
    """This uid's per-second series: the live judge's, else the last one it
    handed over on ``detach``. ``None`` when nothing was ever measured."""
    judge = judge_for(uid)
    if judge is not None:
        return judge.series()
    with _judges_lock:
        return _recent_series.get(uid)


def forget_series(uid: str) -> None:
    """Drop the post-detach copy once it has been persisted."""
    with _judges_lock:
        _recent_series.pop(uid, None)


def reset() -> None:
    """Tests only: forget every session."""
    with _judges_lock:
        _judges.clear()
        _recent_series.clear()


def feed_pcm(uid: str, frame: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Hand one PCM16 frame from the phone's stream to ``uid``'s judge.

    Called from ``audio_pipeline``'s receive loop, which is the hottest path
    in the server — so this appends bytes and returns. When a score comes
    due it is run on a worker thread (``asyncio.to_thread``): the model call
    is ~130 ms of CPU and must never sit on the event loop. Outside a running
    loop (replay, tests) it scores inline so the caller sees a deterministic
    result.

    Silent no-op for a uid with no judge ATTACHED — this never creates one.
    ``relay.register_live_session`` is the only thing that does, so audio
    from a phone with no live wrist is never scored.
    """
    if not frame:
        return
    judge = judge_for(uid)
    if judge is None:
        return
    if not judge.feed(frame, sample_rate):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        judge.score_now()
        return
    if judge.scoring:
        return
    judge.scoring = True
    task = loop.create_task(_score_off_loop(judge))
    _score_tasks.add(task)
    task.add_done_callback(_score_tasks.discard)


async def _score_off_loop(judge: HeatJudge) -> None:
    try:
        await asyncio.to_thread(judge.score_now)
    finally:
        judge.scoring = False


def judge_escalation(uid: str, rung: int):
    """The relay's ONE hook. See ``relay.push_turn_local``.

    Returns:
      ``None``            no judge (``off`` mode, or no audio for this uid).
      :class:`HeatVerdict`  a verdict available right now.
      an awaitable       only when the escalation is a FIRST-RUNG tap in
                         ``on`` mode and the verdict is still ``unknown``:
                         the caller awaits it (bounded by
                         :data:`JUDGE_WAIT_S`) on the watch socket's loop.
                         Higher rungs never get one — vetoed if a verdict is
                         already in hand, never delayed.
    """
    mode = _mode()
    if mode == "off":
        return None
    judge = judge_for(uid)
    if judge is None:
        return None
    acts = mode == "on"
    verdict = judge.verdict(acts=acts)
    if acts and rung == FIRST_RUNG and verdict.verdict == VERDICT_UNKNOWN:
        return _await_verdict(judge, uid=uid, deadline=judge.clock() + JUDGE_WAIT_S)
    log_verdict(uid, verdict)
    return verdict


async def _await_verdict(judge: HeatJudge, *, uid: str, deadline: float) -> HeatVerdict:
    """Poll until the judge has an opinion, or the deadline passes.

    Whatever we have at the deadline is what ships — and ``unknown``
    escalates, so a model that never answers costs the wearer at most
    :data:`JUDGE_WAIT_S` of delay on the first tap of an episode and nothing
    else. The scorer runs on another thread (possibly another loop), so
    polling rather than an Event keeps this free of cross-loop signalling.
    """
    while True:
        verdict = judge.verdict(acts=True)
        if verdict.verdict != VERDICT_UNKNOWN or judge.clock() >= deadline:
            log_verdict(uid, verdict)
            return verdict
        await asyncio.sleep(JUDGE_POLL_S)


def verdict_for_turn(uid: str, dims: dict[str, float] | None = None) -> HeatVerdict:
    """The verdict that gates SURFACING for one phone-finalized turn.

    ``MINDSHIFT_TONE_AUDIO=on`` flips ``tone_id.surface_allowed()``, which
    today would make ``audio_pipeline._enrich_tone`` send a
    ``ToneFlagEvent(source="audio")`` to the phone, fan it out to every other
    in-call participant, and hand it to the watch relay — on EVERY turn, with
    no judgement attached. This is the gate on all three: only ``confirm``
    surfaces (see :func:`surfaces`).

    Prefers the live rolling judge, which is the thing the module name
    promises: a 2 s window scored every second, smoothed, current as of the
    moment the turn closed. Falls back to the turn's OWN dims when no judge
    is running for this uid (no relayed lane, or the frame feed never
    started) — one turn-length reading, treated as its own steady state,
    which is what a turn IS to a caller that has nothing else. Returns
    ``unknown`` when there is neither, and ``unknown`` never surfaces.
    """
    mode = _mode()
    acts = mode == "on"
    judge = judge_for(uid)
    if judge is not None:
        verdict = judge.verdict(acts=acts)
        if verdict.verdict != VERDICT_UNKNOWN:
            return verdict
    if dims and "arousal" in dims and "valence" in dims:
        clean = {k: float(v) for k, v in dims.items() if isinstance(v, (int, float))}
        return verdict_from([clean] * MIN_SCORES, acts=acts)
    return HeatVerdict(VERDICT_UNKNOWN, reason="no judge and no dims", acts=acts)


def surfaces(verdict: HeatVerdict) -> bool:
    """May this verdict's tone flag be SHOWN to a user / stored as escalated?

    ``confirm`` only. ``unknown`` escalates the wrist (a missing judge must
    never mute the product) but does NOT surface: a haptic the wearer can
    attribute to their own volume is a much smaller claim than a labelled
    "you sounded angry" on the screen, in a call, or in a stored analysis
    that Growth/Replay/YourDay will render back weeks later.
    """
    return verdict.verdict == VERDICT_CONFIRM


def log_verdict(uid: str, verdict: HeatVerdict) -> None:
    """One structured line per verdict — the only trace the judge leaves.

    In ``dark`` mode this line IS the feature: computed, logged, behaviour
    untouched (plan step 2 -> 3).
    """
    fields = verdict.log_fields()
    logger.info(
        "heat judge: uid=%s verdict=%s arousal=%s valence=%s latency_ms=%s n=%d acts=%s reason=%s",
        uid, fields["verdict"], fields["arousal"], fields["valence"],
        fields["latency_ms"], fields["n_scores"], fields["acts"], fields["reason"],
    )
