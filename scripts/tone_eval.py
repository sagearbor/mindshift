#!/usr/bin/env python3
"""Measure server/tone_id.py — every audio-tone backend — on OUR labeled fixtures.

The owner's rule for PRD Tier 2 audio tone: build it, MEASURE it on our own
labeled audio, and ship it DARK (computed + logged behind
``MINDSHIFT_TONE_AUDIO``, never surfaced) unless it clears ~60% on our own
ground truth AND adds lift over the text-tone the product already has.
Round 1 (2026-08-24, ``--backend iemocap`` only) is written up in
``docs/research/tone-audio/2026-08-24-wav2vec2-iemocap-eval.md``; round 2
(this script, ``--backend all``) in ``2026-08-24-round2.md``. Both reports
keep their tables in ``docs/research/tone-audio/round2/`` next to the raw
per-turn model outputs, so the analysis can be re-run without the models.

WHAT IT RUNS

1. The LABELED PACK — five acted fixtures with a per-turn coarse label
   (``neutral | angry | sad | happy``): the two round-1 two-voice arguments
   (``gptaudio``, ``openai``; ``scripted_emotion`` collapsed by the EXPECTED
   table below) and the three Foundation-D scenes (``scene_couple_escalation``,
   ``scene_family3``, ``scene_meeting4``; ``emotion_coarse`` straight from the
   meta). 65 turns, 19 of them angry, 2–4 voices per fixture, Speaker A =
   "self" in every scene with a hand-authored expected-nudge timeline. Turn
   times are rebuilt from ``duration_sec`` + ``silence_gap_sec`` exactly as
   ``server/tests/test_diarize_regression_ladder.py::_build_turns`` does,
   audio goes through the production ``audio_ingest.decode_to_pcm_16k`` path,
   and every backend is driven through ``tone_id.classify_turns`` — the same
   function the server calls — so the numbers are the product's numbers.
2. The two REAL recordings (``family_real``, ``poker6_real``; speaker labels,
   NO emotion labels) as a sanity check that ordinary calm conversation does
   not come out "escalating".
3. For every backend: cost (cold load, per-turn CPU latency, realtime factor,
   on-disk size, resident memory after load), then the metrics:

   * 4-class accuracy (categorical backends only), raw argmax and with the
     logits z-scored per speaker over the whole fixture (offline).
   * ESCALATION — the product's question ("is this speaker's own voice
     heating up?"), truth = coarse label ``angry`` — scored three ways:
       raw    : the backend's per-turn arousal number as-is;
       delta  : arousal minus the speaker's running median over their
                PREVIOUS turns (``tone_id.EscalationTracker``, causal, the
                live-session shape; a speaker's first turn is unscored);
       zspk   : arousal z-scored per speaker over the whole fixture
                (offline upper bound for "per-speaker normalization").
     Each reports a threshold-free AUC, a leave-one-fixture-out (LOSO)
     threshold accuracy / balanced accuracy / precision / recall (the
     threshold is picked on the other four fixtures and applied to the
     held-out one, then pooled — never tuned on the turn it scores), and the
     same at the PINNED ``tone_id.ESCALATION_DELTA_THRESHOLD`` (in-sample,
     labelled as such). The delta is also re-run at -20 dB (the owner's "not
     a yelling detector" check: a per-speaker delta on a gain-invariant model
     should not move), and its AUC vs history length k = 1/2/3/5/all is
     tabled (how fast a baseline becomes useful).
   * Self-nudge timeline over the three scenes (expected = self angry turns):
     hits / misses / false nudges.
   * Per fixture × speaker: flagged-hot counts on angry vs non-angry turns.
   * TEXT-TONE BASELINE: the product's own prompt (apps/mobile/src/live/
     localLlm.ts) through server/llm_client at MINDSHIFT_MODEL, per turn with
     the preceding turns as history, scored by live_sessions' ``is_escalated``
     rule. Cached in ``round2/text_tone_baseline.json`` (``--refresh-text`` to
     re-query). This is the bar audio has to add lift over.
   * ``--fusion``: prosody (RMS/F0 per-speaker deltas from server/prosody.py)
     + audio delta + text-tone into a logistic regression with LOSO CV
     (needs scikit-learn). CV numbers only; the in-sample column is there to
     show how much a 5-fixture fit flatters itself.

BACKENDS: the three ``tone_id`` backends (``odyssey_dim``, ``superb_er``,
``iemocap`` — selected via MINDSHIFT_TONE_BACKEND for the run), plus two
script-only research rows: ``msp_dim`` (audeering's MSP-Podcast dimensional
model, CC-BY-NC-SA "research purpose only" — measured to know what the
dimensional idea buys, deliberately NOT a shippable backend) and ``prosody``
(RMS dB / F0 median, the volume-only floor audio must beat).

LABEL MAPPING for the round-1 fixtures (``scripted_emotion`` → coarse):
  calm_open/calm_guarded/calm_close/repair_hopeful → neutral;
  tense_rising/defensive_rising/shout_angry/cold_contempt → angry;
  hurt_sad/scared_shaky → sad (IEMOCAP has no fear; sad shares its negative
  valence + low-dominance voice quality — the valence side, stated). The
  scene metas carry ``emotion_coarse`` from the same fixed table.

USAGE
    tmp/venv-voice/bin/python scripts/tone_eval.py --backend all --fusion
    tmp/venv-voice/bin/python scripts/tone_eval.py --backend odyssey_dim
    tmp/venv-voice/bin/python scripts/tone_eval.py --report-only   # from cached raw JSON

Needs requirements-voice.txt installed and, on first run, network for the
pinned HF snapshots (HF_TOKEN optional). Exits 1 with a plain message if a
backend is unavailable — it never prints made-up numbers. ``--backend all``
loads every model in one process (~5 GB RSS on this machine); run backends
one at a time for a clean per-backend RSS figure.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SERVER_DIR = REPO_ROOT / "server"
AUDIO_DIR = SERVER_DIR / "tests" / "fixtures" / "audio"
OUT_DIR_DEFAULT = REPO_ROOT / "docs" / "research" / "tone-audio" / "round2"
sys.path.insert(0, str(SERVER_DIR))

import audio_ingest  # noqa: E402
import prosody  # noqa: E402
import tone_id  # noqa: E402

# Round-1 fixtures: scripted_emotion -> coarse (first = primary). See docstring.
EXPECTED: dict[str, tuple[str, ...]] = {
    "calm_open": ("neutral",),
    "calm_guarded": ("neutral",),
    "calm_close": ("neutral", "happy"),
    "repair_hopeful": ("neutral", "happy"),
    "tense_rising": ("angry",),
    "defensive_rising": ("angry",),
    "shout_angry": ("angry",),
    "cold_contempt": ("angry",),
    "hurt_sad": ("sad",),
    "scared_shaky": ("sad",),
}

LABELED_FIXTURES = ("gptaudio", "openai", "scene_couple_escalation", "scene_family3", "scene_meeting4")
SCENE_FIXTURES = LABELED_FIXTURES[2:]
REAL_FIXTURES = ("family_real", "poker6_real")
TONE_BACKENDS = tuple(tone_id.TONE_BACKENDS)
RESEARCH_BACKENDS = ("msp_dim", "prosody")
ALL_BACKENDS = TONE_BACKENDS + RESEARCH_BACKENDS
GAIN_DB = -20.0
LICENSE = {
    "odyssey_dim": "MIT", "superb_er": "Apache-2.0", "iemocap": "Apache-2.0",
    "msp_dim": "CC-BY-NC-SA-4.0 (research only)", "prosody": "ours (numpy)",
}

# Text-tone escalation rule — mirrors live_sessions.is_escalated (kept local
# so the script never imports the Firestore-touching module).
_ESC_DIMS = ("defensiveness", "sarcasm", "frustration")
_ESC_LABELS = frozenset({
    "defensive", "sarcastic", "frustrated", "angry", "anger", "hostile", "contempt",
    "contemptuous", "irritated", "annoyed", "critical", "aggressive", "dismissive",
    "frustration", "defensiveness", "sarcasm", "escalating", "escalation",
})
_TONE_DOMINANT_THRESHOLD = 60


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def _load_pcm(name: str) -> tuple[np.ndarray, int]:
    wav = AUDIO_DIR / f"test_recording_{name}.wav"
    return audio_ingest.decode_to_pcm_16k(wav.read_bytes(), wav.name)


def _load_meta(name: str) -> dict:
    return json.loads((AUDIO_DIR / f"test_recording_{name}_meta.json").read_text())


def _build_turns(name: str, meta: dict) -> list[dict]:
    """Labeled fixtures: verbatim logic of test_diarize_regression_ladder
    ._build_turns (kept local so a script never imports from the test tree).
    Real fixtures: the meta's own (approximate) boundaries."""
    if "approx_turns" in meta:  # poker6
        return [{"speaker": a["speaker"], "start_time": a["approx_start"], "end_time": a["approx_end"]}
                for a in meta["approx_turns"]]
    if "silence_gap_sec" not in meta:  # family_real
        return [{"speaker": t["speaker"], "start_time": t["start_time"], "end_time": t["end_time"]}
                for t in meta["turns"]]
    gap = meta["silence_gap_sec"]
    turns, t = [], 0.0
    for m in meta["turns"]:
        dur = m["duration_sec"]
        scripted = m.get("scripted_emotion")
        coarse = m.get("emotion_coarse") or (EXPECTED[scripted][0] if scripted in EXPECTED else None)
        turns.append({
            "speaker": m["speaker"], "text": m.get("text", ""), "scripted_emotion": scripted,
            "coarse": coarse, "start_time": round(t, 4), "end_time": round(t + dur, 4),
        })
        t += dur + gap
    return turns


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if sys.platform == "darwin" else ru / 1e3


# ---------------------------------------------------------------------------
# Backends — tone_id for the shippable ones, script-local adapters for research
# ---------------------------------------------------------------------------

class _MspDimResearch:
    """audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim — CC-BY-NC-SA,
    'research purpose only' per its model card. Loaded here ONLY to measure
    what a naturalistic dimensional model buys; never wired into tone_id."""

    SOURCE = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    REVISION = "6eba34a2485ea31cb03600241787c3a5edab8626"

    def load(self) -> None:
        import torch
        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x):
                x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)  # noqa: E702
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hs = torch.mean(self.wav2vec2(input_values)[0], dim=1)
                return hs, self.classifier(hs)

        d = snapshot_download(self.SOURCE, revision=self.REVISION,
                              local_dir=os.path.join(tone_id.cache_dir(), "msp-dim"),
                              allow_patterns=["*.json", "*.safetensors", "README.md"])
        self.torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(d)
        self.model = EmotionModel.from_pretrained(d).eval()
        self.dir = d

    def classify(self, pcm: np.ndarray, sr: int) -> dict:
        y = self.proc(pcm, sampling_rate=sr)["input_values"][0].reshape(1, -1)
        with self.torch.no_grad():
            _, logits = self.model(self.torch.from_numpy(y))
        a, d, v = logits.squeeze().tolist()
        out = tone_id._dims_to_result({"arousal": a, "dominance": d, "valence": v}, "odyssey_dim")
        out["backend"] = "msp_dim"
        out["model"] = f"{self.SOURCE}@{self.REVISION}"
        return out

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in Path(self.dir).rglob("*") if p.is_file() and ".cache" not in p.parts)


class _ProsodyResearch:
    """RMS dB + F0 median per turn (server/prosody.py) — the volume-only floor."""

    def load(self) -> None:
        pass

    def classify(self, pcm: np.ndarray, sr: int) -> dict:
        f0_median, f0_std, voiced = prosody.estimate_pitch(pcm, sr)
        rms_db = 20.0 * np.log10(max(prosody.rms_energy(pcm), 1e-6))
        return {
            "label": "n/a", "confidence": 0.0, "kind": "prosody", "backend": "prosody", "model": "prosody.py",
            "scores": {"rms_db": round(float(rms_db), 3), "f0_median": f0_median, "f0_std": f0_std,
                       "voiced_fraction": voiced},
            "arousal": round(float(rms_db), 4),
        }

    def size_bytes(self) -> int:
        return 0


def _tone_id_size_bytes(name: str) -> int:
    cache = Path(tone_id.cache_dir())
    if name == "iemocap":
        p = cache / "wav2vec2.ckpt"
        return p.stat().st_size if p.is_file() else 0
    sub = cache / tone_id.BACKEND_INFO[name]["subdir"]
    return sum(p.stat().st_size for p in sub.rglob("*") if p.is_file() and ".cache" not in p.parts)


def extract_backend(name: str, gain: bool = True) -> dict:
    """Run one backend over every fixture; return the raw per-turn record."""
    rss0 = _rss_mb()
    if name in TONE_BACKENDS:
        os.environ[tone_id.TONE_BACKEND_ENV] = name
        if not tone_id.is_available(name):
            raise SystemExit(f"backend {name!r} unavailable: install requirements-voice.txt into this venv")
        t0 = time.perf_counter()
        tone_id._load_model(name)
        load_s = time.perf_counter() - t0
        classify_fn, size = None, _tone_id_size_bytes(name)
        model_used = tone_id.model_id(name)
        threshold = tone_id.escalation_threshold(name)
    else:
        adapter = _MspDimResearch() if name == "msp_dim" else _ProsodyResearch()
        t0 = time.perf_counter()
        adapter.load()
        load_s = time.perf_counter() - t0
        classify_fn, size = adapter.classify, adapter.size_bytes()
        model_used = getattr(adapter, "SOURCE", "prosody.py")
        # research rows: thresholds are not pinned; LOSO only. The delta is
        # still annotated (tracker) with a placeholder threshold of +inf.
        threshold = float("inf")
    rss1 = _rss_mb()
    rec = {
        "backend": name, "model": model_used, "license": LICENSE[name], "load_s": load_s,
        "rss_before_mb": rss0, "rss_after_load_mb": rss1, "size_bytes": size,
        "pinned_threshold": None if threshold == float("inf") else threshold,
        "fixtures": {},
    }
    for fx in LABELED_FIXTURES + REAL_FIXTURES:
        meta = _load_meta(fx)
        pcm, sr = _load_pcm(fx)
        turns = _build_turns(fx, meta)
        # warm-up so the first latency isn't a cold-graph outlier
        if fx == LABELED_FIXTURES[0]:
            tone_id.classify_turns(pcm[: sr * 2], sr, [{"speaker": "warm", "start_time": 0.0, "end_time": 2.0}],
                                   classify_fn=classify_fn, threshold=threshold)
        results = tone_id.classify_turns(pcm, sr, turns, classify_fn=classify_fn, threshold=threshold)
        rows = []
        for t, r in zip(turns, results):
            rows.append({
                "index": r["index"], "speaker": t["speaker"], "seconds": r["seconds"],
                "text": t.get("text", ""), "scripted": t.get("scripted_emotion"), "coarse": t.get("coarse"),
                "skipped": r["skipped"], "latency_ms": r["latency_ms"], "tone": r["tone"],
            })
        if gain and fx in LABELED_FIXTURES:
            g = tone_id.classify_turns(pcm * (10 ** (GAIN_DB / 20.0)), sr, turns,
                                       classify_fn=classify_fn, threshold=threshold)
            for row, r in zip(rows, g):
                row["tone_gain"] = r["tone"]
        rec["fixtures"][fx] = {"self_speaker": meta.get("self_speaker"), "rows": rows}
        print(f"  {name}: {fx} ({len(rows)} turns)", flush=True)
    rec["rss_peak_mb"] = _rss_mb()
    return rec


# ---------------------------------------------------------------------------
# Text-tone baseline (the product's own prompt through server/llm_client)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a discreet real-time conversation coach whispering to one person "
    "during a conversation. Reply with ONLY a JSON object, no prose, no markdown: "
    '{"suggestion": string, "tone": {"warmth": 0-100, "defensiveness": 0-100, '
    '"sarcasm": 0-100, "sadness": 0-100, "frustration": 0-100, "label": string}}. '
    '"tone" scores the turn you were given. Keep "suggestion" under 18 words.'
)


def _text_user_prompt(history: list[tuple[str, str]], speaker: str, text: str, is_self: bool) -> str:
    hist = "\n".join(f"{s}: {t}" for s, t in history)
    who = "the coached person (YOU)" if is_self else speaker
    task = ("The coached person just said this. Give a single delivery nudge for them "
            "(6 words or fewer, e.g. \"ease up\", \"let them finish\").") if is_self else \
           f"Suggest what the coached person should say next to {speaker}, in a balanced stance."
    return (f"Earlier:\n{hist}\n\n" if hist else "") + f"Latest turn from {who}: \"{text}\"\n\n{task}"


def refresh_text_baseline(path: Path) -> dict:
    """Re-query MINDSHIFT_MODEL for every labeled turn (costs API calls)."""
    import llm_client

    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    model = os.environ.get("MINDSHIFT_MODEL", llm_client.DEFAULT_MODEL)
    client = llm_client.LLMClient(model)
    out: dict = {"model": model}
    for fx in LABELED_FIXTURES:
        meta = _load_meta(fx)
        turns = _build_turns(fx, meta)
        self_spk = meta.get("self_speaker") or "Speaker A"
        rows: dict = {}
        history: list[tuple[str, str]] = []
        for t in turns:
            user = _text_user_prompt(history, t["speaker"], t["text"], t["speaker"] == self_spk)
            tone, err, t0 = None, None, time.perf_counter()
            for _ in range(3):
                try:
                    txt = client.complete(_SYSTEM_PROMPT, user, temperature=0.0, max_tokens=200)
                    m = re.search(r"\{.*\}", txt, re.S)
                    obj = json.loads(m.group(0)) if m else None
                    tone = obj.get("tone") if isinstance(obj, dict) else None
                    if isinstance(tone, dict):
                        break
                    tone, err = None, f"unparseable: {txt[:80]!r}"
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)[:120]
            rows[str(len(rows))] = {"tone": tone, "error": None if tone else err,
                                    "latency_ms": (time.perf_counter() - t0) * 1000}
            history.append((t["speaker"], t["text"]))
            print(f"  text {fx} #{len(rows) - 1} {t['speaker']} {t['coarse']} -> {tone}", flush=True)
        out[fx] = rows
    client.close()
    path.write_text(json.dumps(out, indent=1))
    return out


def text_is_escalated(tone: dict | None) -> bool:
    if not tone:
        return False
    if str(tone.get("label") or "").strip().lower() in _ESC_LABELS:
        return True
    return any(isinstance(tone.get(d), (int, float)) and tone[d] >= _TONE_DOMINANT_THRESHOLD for d in _ESC_DIMS)


def text_score(tone: dict | None) -> float:
    if not tone:
        return 0.0
    return float(max((tone.get(d) or 0) for d in _ESC_DIMS))


# ---------------------------------------------------------------------------
# Metrics — numpy only (sklearn only for --fusion)
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count 0.5); nan when one class is absent."""
    y = np.asarray(y, bool); s = np.asarray(s, float)  # noqa: E702
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, bool); pred = np.asarray(pred, bool)  # noqa: E702
    tp = int((y & pred).sum()); fp = int((~y & pred).sum())  # noqa: E702
    fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())  # noqa: E702
    return {
        "n": int(y.size), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": (tp + tn) / y.size if y.size else float("nan"),
        "bal_acc": ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy — used ONLY on LOSO training folds."""
    y = np.asarray(y, bool); s = np.asarray(s, float)  # noqa: E702
    cands = np.unique(s)
    mids = np.concatenate([[cands[0] - 1e-6], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
    best, best_t = -1.0, float(mids[0])
    for t in mids:
        p = s >= t
        ba = binary_metrics(y, p)["bal_acc"]
        if ba > best:
            best, best_t = ba, float(t)
    return best_t


def loso_predict(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict]:
    pred = np.zeros(y.size, bool); thresholds = {}
    for held in sorted(set(groups)):
        tr, te = groups != held, groups == held
        t = best_threshold(y[tr], s[tr]); thresholds[held] = t
        pred[te] = s[te] >= t
    return pred, thresholds


def per_speaker_z(values: np.ndarray, speakers: list[str]) -> np.ndarray:
    v = np.asarray(values, float); out = np.zeros_like(v)  # noqa: E702
    for spk in set(speakers):
        idx = [i for i, s in enumerate(speakers) if s == spk]
        if len(idx) >= 2:
            sd = v[idx].std()
            out[idx] = (v[idx] - v[idx].mean()) / (sd if sd > 1e-9 else 1.0)
    return out


def causal_delta(values: np.ndarray, speakers: list[str], max_history: int | None = None) -> np.ndarray:
    """Same arithmetic as tone_id.EscalationTracker, over a fixture's turns
    (first turn of a speaker → 0, i.e. never flagged)."""
    tr = tone_id.EscalationTracker(max_history=max_history)
    out = []
    for v, s in zip(values, speakers):
        d = tr.observe(s, float(v))["delta"]
        out.append(0.0 if d is None else d)
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# Analysis over the cached raw records
# ---------------------------------------------------------------------------

def _labeled_rows(rec: dict) -> list[dict]:
    rows = []
    for fx in LABELED_FIXTURES:
        f = rec["fixtures"][fx]
        self_spk = f.get("self_speaker") or "Speaker A"
        for r in f["rows"]:
            if r["tone"] is None:
                continue
            rows.append({**r, "fixture": fx, "is_self": r["speaker"] == self_spk, "hot": r["coarse"] == "angry"})
    return rows


def self_nudges(rows: list[dict], pred: np.ndarray) -> dict:
    hit = miss = false = 0
    for r, p in zip(rows, pred):
        if r["fixture"] not in SCENE_FIXTURES or not r["is_self"]:
            continue
        if r["hot"]:
            hit += bool(p); miss += (not p)  # noqa: E702
        else:
            false += bool(p)
    return {"hits": hit, "misses": miss, "false": false}


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def _f2(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def analyze_backend(rec: dict) -> dict:
    rows = _labeled_rows(rec)
    y = np.array([r["hot"] for r in rows]); groups = np.array([r["fixture"] for r in rows])  # noqa: E702
    speakers_by_fx = {fx: [r["speaker"] for r in rows if r["fixture"] == fx] for fx in LABELED_FIXTURES}
    arousal = np.array([r["tone"]["arousal"] for r in rows], float)
    out: dict = {"backend": rec["backend"], "n": int(y.size), "n_hot": int(y.sum())}

    # cost
    lat = [r["latency_ms"] for fx in rec["fixtures"].values() for r in fx["rows"] if r["tone"] is not None]
    secs = [r["seconds"] for fx in rec["fixtures"].values() for r in fx["rows"] if r["tone"] is not None]
    out["cost"] = {
        "license": rec["license"], "model": rec["model"], "size_mb": rec["size_bytes"] / 1e6,
        "rss_after_load_mb": rec["rss_after_load_mb"], "rss_peak_mb": rec.get("rss_peak_mb"),
        "load_s": rec["load_s"], "median_ms": statistics.median(lat), "max_ms": max(lat),
        "rtf": statistics.mean(l / 1000.0 / s for l, s in zip(lat, secs)),
    }

    # 4-class (categorical only)
    if rows and rows[0]["tone"].get("kind") == "categorical":
        # the model's OWN argmax (the escalation layer may have relabelled a
        # flagged turn "escalating"; 4-class scores the classifier, not that)
        raw_ok = [max(r["tone"]["scores"], key=r["tone"]["scores"].get) == r["coarse"] for r in rows]
        # per-speaker z-scored logits argmax (offline); tone["logits"] is the
        # unnormalized 4-vector when the backend exposes it
        z_ok = []
        for fx in LABELED_FIXTURES:
            idx = [i for i, r in enumerate(rows) if r["fixture"] == fx]
            L = np.array([[rows[i]["tone"].get("logits", {}).get(lbl, np.log(max(rows[i]["tone"]["scores"][lbl], 1e-9)))
                           for lbl in tone_id.LABELS] for i in idx])
            Z = np.stack([per_speaker_z(L[:, j], speakers_by_fx[fx]) for j in range(4)], 1)
            for i, z in zip(idx, Z):
                z_ok.append(tone_id.LABELS[int(np.argmax(z))] == rows[i]["coarse"])
        out["four_class"] = {
            "raw": float(np.mean(raw_ok)), "zspk": float(np.mean(z_ok)),
            "per_fixture": {fx: float(np.mean([ok for ok, r in zip(raw_ok, rows) if r["fixture"] == fx]))
                            for fx in LABELED_FIXTURES},
        }

    # escalation: raw / delta / zspk
    def scored(s: np.ndarray) -> dict:
        pred, thr = loso_predict(y, s, groups)
        m = binary_metrics(y, pred)
        return {**m, "auc": auc(y, s), "loso_thresholds": thr, "nudges": self_nudges(rows, pred)}

    delta = np.concatenate([causal_delta(arousal[groups == fx], speakers_by_fx[fx]) for fx in LABELED_FIXTURES])
    zspk = np.concatenate([per_speaker_z(arousal[groups == fx], speakers_by_fx[fx]) for fx in LABELED_FIXTURES])
    out["escalation"] = {"raw": scored(arousal), "delta": scored(delta), "zspk": scored(zspk)}
    # pinned threshold (in-sample) — the number the server actually runs at
    t = rec.get("pinned_threshold")
    if t is not None:
        pred = np.array([bool(r["tone"].get("escalation", {}).get("flag")) for r in rows])
        out["escalation"]["pinned"] = {**binary_metrics(y, pred), "threshold": t, "nudges": self_nudges(rows, pred)}
        # per fixture x speaker at the pinned threshold
        tbl: dict = {}
        for r, p in zip(rows, pred):
            d = tbl.setdefault(f"{r['fixture']}·{r['speaker']}", {"angry": [0, 0], "not": [0, 0]})
            k = "angry" if r["hot"] else "not"
            d[k][0] += int(p); d[k][1] += 1  # noqa: E702
        out["per_speaker"] = tbl
    # -20 dB: delta AUC on the gain run
    if all("tone_gain" in r and r["tone_gain"] for r in rows):
        ag = np.array([r["tone_gain"]["arousal"] for r in rows], float)
        dg = np.concatenate([causal_delta(ag[groups == fx], speakers_by_fx[fx]) for fx in LABELED_FIXTURES])
        out["escalation"]["delta_gain20_auc"] = auc(y, dg)
        out["escalation"]["raw_gain20_auc"] = auc(y, ag)
    # history length
    out["delta_vs_history"] = {}
    for k in (1, 2, 3, 5, None):
        dk = np.concatenate([causal_delta(arousal[groups == fx], speakers_by_fx[fx], k) for fx in LABELED_FIXTURES])
        pred, _ = loso_predict(y, dk, groups)
        out["delta_vs_history"][str(k or "all")] = {"auc": auc(y, dk), "loso_acc": binary_metrics(y, pred)["acc"]}
    # real recordings: flagged count + arousal range
    out["real"] = {}
    for fx in REAL_FIXTURES:
        rs = [r for r in rec["fixtures"][fx]["rows"] if r["tone"] is not None]
        a = [r["tone"]["arousal"] for r in rs]
        out["real"][fx] = {
            "n": len(rs), "flagged": sum(bool(r["tone"].get("escalation", {}).get("flag")) for r in rs),
            "arousal_min": min(a) if a else None, "arousal_median": statistics.median(a) if a else None,
            "arousal_max": max(a) if a else None,
            "labels": {lbl: sum(r["tone"]["label"] == lbl for r in rs) for lbl in sorted({r["tone"]["label"] for r in rs})},
        }
    return out


def analyze_text(text: dict, rec_any: dict) -> dict:
    rows = _labeled_rows(rec_any)
    y = np.array([r["hot"] for r in rows]); groups = np.array([r["fixture"] for r in rows])  # noqa: E702
    tones = [text.get(r["fixture"], {}).get(str(r["index"]), {}).get("tone") for r in rows]
    pred = np.array([text_is_escalated(t) for t in tones]); s = np.array([text_score(t) for t in tones])  # noqa: E702
    lp, _ = loso_predict(y, s, groups)
    return {
        "model": text.get("model"), "rule": {**binary_metrics(y, pred), "auc": auc(y, s), "nudges": self_nudges(rows, pred)},
        "loso": {**binary_metrics(y, lp), "nudges": self_nudges(rows, lp)},
        "missing": int(sum(t is None for t in tones)),
    }


def analyze_fusion(recs: dict[str, dict], text: dict) -> dict | None:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:  # noqa: BLE001
        print("fusion skipped: scikit-learn not installed in this venv", file=sys.stderr)
        return None
    base = next(iter(recs.values()))
    rows = _labeled_rows(base)
    y = np.array([r["hot"] for r in rows]); groups = np.array([r["fixture"] for r in rows])  # noqa: E702
    spk = {fx: [r["speaker"] for r in rows if r["fixture"] == fx] for fx in LABELED_FIXTURES}

    def col(name: str, key: str) -> np.ndarray:
        rs = _labeled_rows(recs[name])
        return np.array([r["tone"]["scores"][key] if key in r["tone"]["scores"] else r["tone"]["arousal"] for r in rs], float)

    def delta_of(v: np.ndarray) -> np.ndarray:
        return np.concatenate([causal_delta(v[groups == fx], spk[fx]) for fx in LABELED_FIXTURES])

    feats: dict[str, np.ndarray] = {}
    tones = [text.get(r["fixture"], {}).get(str(r["index"]), {}).get("tone") or {} for r in rows]
    for d in _ESC_DIMS:
        feats[f"text_{d}"] = np.array([t.get(d) or 0 for t in tones], float)
    if "prosody" in recs:
        rms = col("prosody", "rms_db")
        f0 = np.array([v if v is not None else np.nan for v in col("prosody", "f0_median")], float)
        f0 = np.where(np.isnan(f0), np.nanmedian(f0), f0)
        feats["rms_delta"] = delta_of(rms); feats["f0_delta"] = delta_of(f0)  # noqa: E702
        feats["rms_session_z"] = np.concatenate([(rms[groups == fx] - rms[groups == fx].mean()) / (rms[groups == fx].std() or 1.0)
                                                 for fx in LABELED_FIXTURES])
    for name in recs:
        if name in ("prosody",):
            continue
        a = np.array([r["tone"]["arousal"] for r in _labeled_rows(recs[name])], float)
        feats[f"{name}_raw"] = a; feats[f"{name}_delta"] = delta_of(a)  # noqa: E702
    text_cols = [f"text_{d}" for d in _ESC_DIMS]
    sets: dict[str, list[str]] = {"text only": text_cols}
    if "prosody" in recs:
        sets["prosody only (rms delta, f0 delta)"] = ["rms_delta", "f0_delta"]
        sets["volume only (rms session-z)"] = ["rms_session_z"]
    for name in recs:
        if name == "prosody":
            continue
        sets[f"{name} delta only"] = [f"{name}_delta"]
        sets[f"{name} raw + delta"] = [f"{name}_raw", f"{name}_delta"]
        sets[f"text + {name} (raw + delta)"] = text_cols + [f"{name}_raw", f"{name}_delta"]
        if "prosody" in recs:
            sets[f"text + prosody + {name} (raw + delta)"] = text_cols + ["rms_delta", "f0_delta", f"{name}_raw", f"{name}_delta"]
    if "prosody" in recs:
        sets["text + prosody"] = text_cols + ["rms_delta", "f0_delta"]
    out = {}
    for label, cols in sets.items():
        X = np.stack([feats[c] for c in cols], 1)
        cv_pred = np.zeros(y.size, bool); cv_score = np.zeros(y.size)  # noqa: E702
        for held in LABELED_FIXTURES:
            tr, te = groups != held, groups == held
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
            cv_score[te] = clf.decision_function((X[te] - mu) / sd); cv_pred[te] = cv_score[te] >= 0  # noqa: E702
        mu, sd = X.mean(0), X.std(0) + 1e-9
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit((X - mu) / sd, y)
        out[label] = {**binary_metrics(y, cv_pred), "auc": auc(y, cv_score), "nudges": self_nudges(rows, cv_pred),
                      "in_sample_acc": float((clf.predict((X - mu) / sd) == y).mean()),
                      "features": cols, "coef": dict(zip(cols, [round(float(c), 3) for c in clf.coef_[0]]))}
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_tables(summary: dict) -> str:
    L: list[str] = []
    add = L.append
    add(f"# Round-2 tone tables ({summary['date']}) — generated by `scripts/tone_eval.py`")
    add("")
    add(f"Labeled pack: {summary['n']} turns ({summary['n_hot']} angry) over {len(LABELED_FIXTURES)} fixtures; "
        f"self turns in the 3 scenes carry the expected-nudge timeline. Machine: {summary['machine']}.")
    add("")
    add("## Cost")
    add("")
    add("| backend | license | size MB | RSS after load MB | cold load s | median ms/turn | max ms/turn | RTF |")
    add("|---|---|---|---|---|---|---|---|")
    for name, a in summary["backends"].items():
        c = a["cost"]
        add(f"| {name} | {c['license']} | {c['size_mb']:.0f} | {c['rss_after_load_mb']:.0f} | {c['load_s']:.1f} | "
            f"{c['median_ms']:.0f} | {c['max_ms']:.0f} | {c['rtf']:.3f} |")
    add("")
    if summary.get("text"):
        t = summary["text"]
        add(f"## Text-tone baseline (`{t['model']}`, product prompt, `is_escalated` rule)")
        add("")
        r, lo = t["rule"], t["loso"]
        add(f"- Rule (label ∈ ESCALATION_LABELS or any escalation dim ≥ 60): acc {_pct(r['acc'])}, bal-acc "
            f"{_pct(r['bal_acc'])}, precision {_pct(r['precision'])}, recall {_pct(r['recall'])}, AUC (max escalation "
            f"dim) {_f2(r['auc'])}; self nudges {r['nudges']['hits']} hit / {r['nudges']['misses']} miss / "
            f"{r['nudges']['false']} false.")
        add(f"- LOSO threshold on the max escalation dim: acc {_pct(lo['acc'])}, bal-acc {_pct(lo['bal_acc'])}, "
            f"precision {_pct(lo['precision'])}, recall {_pct(lo['recall'])}; nudges {lo['nudges']['hits']}/"
            f"{lo['nudges']['misses']}/{lo['nudges']['false']}.")
        add("")
    add("## 4-class accuracy (categorical backends; coarse neutral/angry/sad/happy)")
    add("")
    add("| backend | raw argmax | per-speaker z-scored logits argmax (offline) | per fixture (raw) |")
    add("|---|---|---|---|")
    for name, a in summary["backends"].items():
        if "four_class" in a:
            fc = a["four_class"]
            add(f"| {name} | {_pct(fc['raw'])} | {_pct(fc['zspk'])} | "
                + ", ".join(f"{k}: {_pct(v)}" for k, v in fc["per_fixture"].items()) + " |")
    add("")
    add("## Escalation (truth = angry): AUC threshold-free; LOSO = leave-one-fixture-out threshold; pinned = the server's constant (in-sample)")
    add("")
    add("| backend | signal | AUC | LOSO acc | bal-acc | prec | rec | self nudges hit/miss/false | delta AUC @-20 dB |")
    add("|---|---|---|---|---|---|---|---|---|")
    for name, a in summary["backends"].items():
        e = a["escalation"]
        for sig in ("raw", "delta", "zspk"):
            m = e[sig]
            g = _f2(e.get("delta_gain20_auc")) if sig == "delta" else (_f2(e.get("raw_gain20_auc")) if sig == "raw" else "")
            add(f"| {name} | {sig} | {_f2(m['auc'])} | {_pct(m['acc'])} | {_pct(m['bal_acc'])} | {_pct(m['precision'])} | "
                f"{_pct(m['recall'])} | {m['nudges']['hits']}/{m['nudges']['misses']}/{m['nudges']['false']} | {g} |")
        if "pinned" in e:
            p = e["pinned"]
            add(f"| {name} | delta @ pinned {p['threshold']} | — | {_pct(p['acc'])} | {_pct(p['bal_acc'])} | "
                f"{_pct(p['precision'])} | {_pct(p['recall'])} | {p['nudges']['hits']}/{p['nudges']['misses']}/{p['nudges']['false']} | |")
    add("")
    add("## Causal delta vs history length (AUC / LOSO acc)")
    add("")
    add("| backend | k=1 | k=2 | k=3 | k=5 | all | zspk (offline) |")
    add("|---|---|---|---|---|---|---|")
    for name, a in summary["backends"].items():
        cells = [f"{_f2(v['auc'])} / {_pct(v['loso_acc'])}" for v in a["delta_vs_history"].values()]
        z = a["escalation"]["zspk"]
        add(f"| {name} | " + " | ".join(cells) + f" | {_f2(z['auc'])} / {_pct(z['acc'])} |")
    add("")
    add("## Per speaker at the pinned threshold (flagged on angry turns / flagged on non-angry turns)")
    add("")
    for name, a in summary["backends"].items():
        if "per_speaker" in a:
            add(f"- **{name}**: " + "; ".join(
                f"{k.replace('test_recording_', '').replace('scene_', '').replace('Speaker ', '')}: "
                f"angry {v['angry'][0]}/{v['angry'][1]}, not {v['not'][0]}/{v['not'][1]}"
                for k, v in a["per_speaker"].items()))
    add("")
    add("## Real recordings (no labels; calm conversation must not flag)")
    add("")
    for name, a in summary["backends"].items():
        for fx, r in a["real"].items():
            add(f"- {name} / {fx}: {r['flagged']}/{r['n']} turns flagged escalating; arousal "
                f"{_f2(r['arousal_min'])} … {_f2(r['arousal_median'])} … {_f2(r['arousal_max'])}; labels {r['labels']}")
    add("")
    if summary.get("fusion"):
        add("## Fusion — logistic regression, leave-one-fixture-out CV (5 folds), standardized features")
        add("")
        add("| features | CV AUC | CV acc | bal-acc | prec | rec | self nudges | in-sample acc (optimistic) |")
        add("|---|---|---|---|---|---|---|---|")
        for label, m in summary["fusion"].items():
            add(f"| {label} | {_f2(m['auc'])} | {_pct(m['acc'])} | {_pct(m['bal_acc'])} | {_pct(m['precision'])} | "
                f"{_pct(m['recall'])} | {m['nudges']['hits']}/{m['nudges']['misses']}/{m['nudges']['false']} | {_pct(m['in_sample_acc'])} |")
        add("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="all",
                    help="one of %s, or 'all' (default)" % ", ".join(ALL_BACKENDS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--no-gain", action="store_true", help="skip the -20 dB re-run")
    ap.add_argument("--fusion", action="store_true", help="LOSO logistic fusion (needs scikit-learn)")
    ap.add_argument("--refresh-text", action="store_true", help="re-query the LLM text-tone baseline (API calls)")
    ap.add_argument("--report-only", action="store_true", help="render from cached raw_*.json, run no models")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if tone_id.mode() == "off":
        os.environ[tone_id.TONE_AUDIO_ENV] = "dark"  # measuring, not surfacing

    names = list(ALL_BACKENDS) if args.backend == "all" else [args.backend]
    for n in names:
        if n not in ALL_BACKENDS:
            print(f"unknown backend {n!r}; choose from {ALL_BACKENDS} or all", file=sys.stderr)
            return 2
    if not args.report_only:
        for n in names:
            print(f"== {n}", flush=True)
            rec = extract_backend(n, gain=not args.no_gain)
            (args.out_dir / f"raw_{n}.json").write_text(json.dumps(rec, indent=1, default=float))
            lat = [r["latency_ms"] for fx in rec["fixtures"].values() for r in fx["rows"] if r["tone"] is not None]
            print(f"   load {rec['load_s']:.1f}s, RSS {rec['rss_after_load_mb']:.0f} MB, "
                  f"size {rec['size_bytes'] / 1e6:.0f} MB, latency median {statistics.median(lat):.0f} ms", flush=True)

    text_path = args.out_dir / "text_tone_baseline.json"
    if args.refresh_text or not text_path.is_file():
        print("== text-tone baseline (LLM)", flush=True)
        text = refresh_text_baseline(text_path)
    else:
        text = json.loads(text_path.read_text())

    recs = {}
    for n in ALL_BACKENDS:
        p = args.out_dir / f"raw_{n}.json"
        if p.is_file():
            recs[n] = json.loads(p.read_text())
    if not recs:
        print("no raw_*.json to report on", file=sys.stderr)
        return 1
    summary: dict = {
        "date": date.today().isoformat(),
        "machine": f"{platform.machine()} / {platform.platform()} / python {platform.python_version()}",
        "backends": {n: analyze_backend(r) for n, r in recs.items()},
    }
    try:
        import torch
        summary["machine"] += f" / torch {torch.__version__} ({torch.get_num_threads()} threads)"
    except Exception:  # noqa: BLE001
        pass
    first = next(iter(recs.values()))
    summary["n"] = summary["backends"][first["backend"]]["n"]
    summary["n_hot"] = summary["backends"][first["backend"]]["n_hot"]
    summary["text"] = analyze_text(text, first)
    if args.fusion:
        summary["fusion"] = analyze_fusion(recs, text)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    tables = render_tables(summary)
    (args.out_dir / "tables.md").write_text(tables)
    print(tables)
    print(f"\nwrote {args.out_dir / 'tables.md'}\n      {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
