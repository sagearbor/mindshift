"""Experiment 1 — adaptive S-norm (AS-Norm) for self-identification.

Question: the owner's voice across settings scores 0.24-0.45 raw cosine
against a one-setting print while different people score 0.11-0.28, so no
absolute threshold works (speaker_id.py CROSS_MATCH note, 2026-08-27). Does
normalising every score against a background COHORT open a gap?

    s_norm = 0.5 * ((s - mu_e) / sd_e + (s - mu_t) / sd_t)

where mu_e/sd_e are the mean/sd of the enrollment print's TOP-N cosines to
the cohort and mu_t/sd_t the same for the test embedding (Matejka et al.
2017). N in {10, 20, 30, all} (N=all is plain S-norm).

Two cohorts (pinned ECAPA, speaker_id.embed_pcm / embed_pcm_batch):
  fixtures        every NON-owner voice in the checked-in fixtures: one
                  POOLED vector per (fixture, speaker) plus up to
                  CAP_PER_VOICE non-overlapping 1.5 s speech windows per
                  distinct voice (TTS voices onyx/coral/ballad/nova/marin
                  recur across fixtures — deduplicated by voice id)
  fixtures+libri  the same plus 40 LibriSpeech dev-clean speakers (pooled +
                  6 windows each; exp1_libri_cohort.py)
Scoring is leave-one-voice-out: when a non-owner probe's own voice is in the
cohort it is excluded, as a stranger in a real recording would be. The owner
is never in the cohort.

Prints (enrollment side): maggiano-only (rubric dad audio pooled; and the 3
stored maggiano samples' centroid — what the app actually enrolled),
family-only (Sage pooled), poker-only (Player6 pooled), the stored guided
sample alone, the stored 5-sample per-recording blend (speaker_id.
blend_samples: maggiano x3 -> 1 centroid, guided, family = 3 settings), and
a 2-setting maggiano+family blend from this script's own pools.

Probes (test side): the owner pooled in each of the 3 recordings (+ his
1.5 s windows), every non-owner pooled speaker (maggiano mom/asher, family
Asher, poker P1-P5, the TTS voices per fixture) (+ their windows). A probe
from the recording a print was built from is flagged IN-SAMPLE and excluded
from the "owner's lowest cross-setting score".

Outputs results.json["exp1"] and a table on stdout. Embeddings are cached in
cache/exp1_emb.npz (gitignored: it holds vectors from the private clip).
"""
from __future__ import annotations

import json

import numpy as np

import common as C
from common import speaker_id

CAP_PER_VOICE = 20
NS = [10, 20, 30, None]
OWNER_PROFILE = C.PRIVATE / "owner_profile.json"

# (fixture, GT label) -> distinct voice id. Owner labels are NOT listed.
VOICES = {
    "poker6": {f"Player{i}": f"poker_P{i}" for i in range(1, 6)},
    "family_real": {"Asher": "family_Asher"},
    "openai": {"Speaker A": "tts_coral", "Speaker B": "tts_onyx"},
    "gptaudio": {"Speaker A": "tts_coral", "Speaker B": "tts_onyx"},
    "scene_couple": {"Speaker A": "tts_onyx", "Speaker B": "tts_coral"},
    "scene_family3": {"Speaker A": "tts_onyx", "Speaker B": "tts_ballad", "Speaker C": "tts_nova"},
    "scene_meeting4": {"Speaker A": "tts_onyx", "Speaker B": "tts_marin", "Speaker C": "tts_ballad",
                       "Speaker D": "tts_nova"},
}
OWNER = {"maggiano3": "dad", "family_real": "Sage", "poker6": "Player6"}
PRIVATE_NON_OWNER = {"maggiano3": {"mom": "magg_mom", "asher": "magg_asher"}}


def build_embeddings() -> dict:
    f = C.CACHE / "exp1_emb.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    C.torch_threads()
    speaker_id._load_model()
    pooled, windows = [], []          # rows: (fixture, label, voice, kind) + vec
    pooled_v, windows_v = [], []
    for name in list(VOICES) + ["maggiano3"]:
        pcm, sr = C.load_audio(name)
        pools = C.speaker_pcm(name, pcm, sr)
        vmap = dict(VOICES.get(name, {}))
        vmap.update(PRIVATE_NON_OWNER.get(name, {}))
        if name in OWNER:
            vmap[OWNER[name]] = "OWNER"
        for label, voice in vmap.items():
            if label not in pools:
                continue
            pooled.append((name, label, voice, "pooled"))
            pooled_v.append(C.embed_pooled(pools[label], sr))
            chunks = C.speaker_windows(pools[label], sr)
            if chunks:
                vecs = C.embed_many(chunks, sr)
                for v in vecs:
                    windows.append((name, label, voice, "window"))
                    windows_v.append(v)
            print(f"  {name:14s} {label:10s} -> {voice:14s} pooled {pools[label].size / sr:5.1f}s, {len(chunks)} windows", flush=True)
    out = {"pooled_meta": np.array(pooled, dtype=object), "pooled": np.stack(pooled_v),
           "window_meta": np.array(windows, dtype=object), "windows": np.stack(windows_v)}
    np.savez(f, **out)
    return out


def l2n(v):
    return speaker_id.l2_normalize(np.asarray(v, dtype=np.float32))


def asnorm(s: float, e: np.ndarray, t: np.ndarray, cohort: np.ndarray, excl: np.ndarray, n: int | None) -> float:
    ce, ct = cohort @ e, cohort @ t
    ce, ct = ce[~excl], ct[~excl]
    if n is not None:
        ce, ct = np.sort(ce)[-n:], np.sort(ct)[-n:]
    return float(0.5 * ((s - ce.mean()) / max(ce.std(), 1e-6) + (s - ct.mean()) / max(ct.std(), 1e-6)))


def build_cohort(E: dict, with_libri: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    pm, pv, wm, wv = E["pooled_meta"], E["pooled"], E["window_meta"], E["windows"]

    def is_cohort(m):
        return m[2] != "OWNER" and m[0] in VOICES
    meta, vec = [], []
    for m, v in zip(pm, pv):
        if is_cohort(m):
            meta.append(tuple(m)); vec.append(v)
    per_voice: dict[str, list] = {}
    for m, v in zip(wm, wv):
        if is_cohort(m):
            per_voice.setdefault(m[2], []).append((tuple(m), v))
    for voice, items in per_voice.items():
        step = max(1, int(np.ceil(len(items) / CAP_PER_VOICE)))
        for m, v in items[::step][:CAP_PER_VOICE]:
            meta.append(m); vec.append(v)
    if with_libri:
        z = np.load(C.CACHE / "libri_emb.npz", allow_pickle=True)
        for m, v in zip(z["pooled_meta"], z["pooled"]):
            meta.append(tuple(m)); vec.append(v)
        for m, v in zip(z["window_meta"], z["windows"]):
            meta.append(tuple(m)); vec.append(v)
    cohort = np.stack(vec)
    voice = np.array([m[2] for m in meta])
    n_pooled = sum(1 for m in meta if m[3] == "pooled")
    info = {"n_vectors": int(len(meta)), "n_pooled": n_pooled, "n_windows": int(len(meta) - n_pooled),
            "n_voices": int(len(set(voice))), "voices": sorted(set(voice.tolist())), "cap_per_voice": CAP_PER_VOICE}
    return cohort, voice, info


def evaluate(E: dict, cohort: np.ndarray, coh_voice: np.ndarray, prints: dict, settings: dict) -> dict:
    pm, pv, wm, wv = E["pooled_meta"], E["pooled"], E["window_meta"], E["windows"]
    probes = [(tuple(m), v) for m, v in zip(pm, pv)]                       # pooled speakers
    wprobes = [(tuple(m), v) for m, v in zip(wm, wv)]                       # 1.5 s windows
    norms = {"raw": None, **{f"asnorm_N{n if n else 'all'}": n for n in NS}}

    def sc(e, t, voice, n):
        s = float(e @ t)
        if n == "raw":
            return s
        return asnorm(s, e, t, cohort, coh_voice == voice, norms[n])

    out = {}
    hdr = f"{'print':48s} {'norm':12s} {'owner OOS min':>14s} {'non-owner max':>14s} {'gap':>7s} {'gap/sd':>7s} {'thr':>7s} | win p10/p90 gap"
    print(hdr)
    for pname, (e, built_from) in prints.items():
        entry = {"settings": settings[pname], "built_from": sorted(built_from), "norms": {}}
        for nname in norms:
            table = {}
            for (fx, label, voice, _), t in probes:
                table[f"{fx}/{label}"] = {"score": round(sc(e, t, voice, nname), 3), "owner": voice == "OWNER",
                                          "in_sample": voice == "OWNER" and fx in built_from}
            owner_oos = [v["score"] for v in table.values() if v["owner"] and not v["in_sample"]]
            non = [v["score"] for v in table.values() if not v["owner"]]
            o_min, n_max = min(owner_oos), max(non)
            ow = [sc(e, t, voice, nname) for (fx, label, voice, _), t in wprobes if voice == "OWNER" and fx not in built_from]
            nw = [sc(e, t, voice, nname) for (fx, label, voice, _), t in wprobes if voice != "OWNER"]
            contrast = {}
            for rec in OWNER:
                sp = {k.split("/")[1]: v["score"] for k, v in table.items() if k.startswith(rec + "/")}
                win = max(sp, key=sp.get)
                others = [v for k, v in sp.items() if k != win]
                contrast[rec] = {"winner": win, "winner_is_owner": win == OWNER[rec], "top": round(sp[win], 3),
                                 "margin": round(sp[win] - max(others), 3), "in_sample": rec in built_from}
            sd_non = float(np.std(non))
            r = {
                "owner_oos_min": round(o_min, 3), "nonowner_max": round(n_max, 3), "gap": round(o_min - n_max, 3),
                "gap_over_nonowner_sd": round((o_min - n_max) / max(sd_non, 1e-6), 2),
                "threshold": round((o_min + n_max) / 2, 3),
                "owner_window_p10": round(float(np.percentile(ow, 10)), 3) if ow else None,
                "owner_window_median": round(float(np.median(ow)), 3) if ow else None,
                "nonowner_window_p90": round(float(np.percentile(nw, 90)), 3),
                "nonowner_window_max": round(float(np.max(nw)), 3),
                "window_frac_separable": round(float(np.mean([1.0 if o > np.percentile(nw, 99) else 0.0 for o in ow])), 3) if ow else None,
                "nonowner_argmax": max((k for k, v in table.items() if not v["owner"]), key=lambda k: table[k]["score"]),
                "owner_argmin_oos": min((k for k, v in table.items() if v["owner"] and not v["in_sample"]), key=lambda k: table[k]["score"]),
                "scores": table, "contrast": contrast,
            }
            entry["norms"][nname] = r
            print(f"{pname:48s} {nname:12s} {o_min:14.3f} {n_max:14.3f} {o_min - n_max:7.3f} {r['gap_over_nonowner_sd']:7.2f} {r['threshold']:7.3f} | "
                  f"{r['owner_window_p10']} / {r['nonowner_window_p90']}  {round(r['owner_window_p10'] - r['nonowner_window_p90'], 3)}")
        out[pname] = entry
    return out


def main() -> None:
    E = build_embeddings()
    pm, pv = E["pooled_meta"], E["pooled"]

    def pooled_of(fx, label):
        return next(v for m, v in zip(pm, pv) if m[0] == fx and m[1] == label)
    prof = json.loads(OWNER_PROFILE.read_text())
    samples = prof["samples"]
    magg_id = "cae798c0-18a9-47ec-8d51-e5111b620e3a"
    stored_magg = [l2n(s["embedding"]) for s in samples if s.get("recording_id") == magg_id]
    guided = [l2n(s["embedding"]) for s in samples if s.get("recording_id") is None]
    prints = {
        "maggiano_only (rubric dad pooled)": (l2n(pooled_of("maggiano3", "dad")), {"maggiano3"}),
        "maggiano_only (3 stored app samples)": (l2n(np.mean(stored_magg, 0)), {"maggiano3"}),
        "guided_only (stored guided sample)": (guided[0], set()),
        "family_only (Sage pooled)": (l2n(pooled_of("family_real", "Sage")), {"family_real"}),
        "poker_only (Player6 pooled)": (l2n(pooled_of("poker6", "Player6")), {"poker6"}),
        "blend maggiano+family (2 settings, this script)": (
            l2n(np.mean([l2n(pooled_of("maggiano3", "dad")), l2n(pooled_of("family_real", "Sage"))], 0)),
            {"maggiano3", "family_real"}),
        "stored blend (5 samples / 3 settings, app)": (l2n(speaker_id.blend_samples(samples)),
                                                       {"maggiano3", "family_real"}),
    }
    settings = {k: (1 if "only" in k else (2 if "2 settings" in k else 3)) for k in prints}
    res = {"cohorts": {}}
    for cname, with_libri in (("fixtures", False), ("fixtures+libri", True)):
        cohort, coh_voice, info = build_cohort(E, with_libri)
        print(f"\n=== cohort {cname}: {info['n_vectors']} vectors ({info['n_pooled']} pooled + {info['n_windows']} windows), {info['n_voices']} voices")
        res["cohorts"][cname] = {"cohort": info, "prints": evaluate(E, cohort, coh_voice, prints, settings)}
    C.merge_results("exp1", res)
    print("wrote results.json[exp1]")


if __name__ == "__main__":
    main()
