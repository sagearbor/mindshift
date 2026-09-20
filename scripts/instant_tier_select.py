"""INSTANT TIER — pick the compact acoustic subset the PHONE can compute itself.

The shipped instant signal is loudness over the speaker's own baseline. It
cannot tell an angry shout from a delighted one: cross-corpus angry-vs-happy
AUC 0.733. The full 88-descriptor eGeMAPSv02 set with a linear probe gets
0.829 — but openSMILE is a native dependency we will not ship, and 88
functionals is not a thing anyone hand-writes in TypeScript.

This script answers: how few of those 88 do we actually need, restricted to
descriptors that are cheap and well-defined to re-implement from scratch on a
2 s window of 16 kHz PCM?

Protocol (deliberately does NOT select on the number it reports):

  selection   greedy forward selection maximising angry-vs-HAPPY AUC under
              speaker-grouped 5-fold CV on CREMA-D alone
  reporting   cross-corpus CREMA-D -> RAVDESS (different actors, rooms,
              scripts, label protocol) — never seen during selection

Subcommands:

  select      greedy selection over the implementable pool; prints candidate
              subsets of several sizes with both numbers; writes
              tmp/instant-tier/selection.json
  dump-pcm    write the 2 s "loudest window" of each clip as raw int16 PCM
              under tmp/instant-tier/pcm/ (gitignored) so the TypeScript
              extractor can be run over real audio from node, plus openSMILE
              features recomputed on exactly those 2 s windows
  fit-ts      fit the shipped model on the features the TYPESCRIPT extractor
              produced (node scripts/instant_tier_eval.mjs --features ...),
              so the coefficients are calibrated to what the phone measures,
              and write apps/mobile/src/live/instantTier.model.json
  fixture     build apps/mobile/__tests__/fixtures/instantTier.parity.json

    python scripts/instant_tier_select.py select
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/Users/sagearbor/projects/githubs/mindshift")
BANK = MAIN_REPO / "tmp/feature-bank"
WORK = MAIN_REPO / "tmp/instant-tier"
MODEL_OUT = REPO / "apps/mobile/src/live/instantTier.model.json"
FIXTURE_OUT = REPO / "apps/mobile/__tests__/fixtures/instantTier.parity.json"

WINDOW_SEC = 2.0
SR = 16000

# ---------------------------------------------------------------------------
# The pool: eGeMAPS functionals that a hand-written TypeScript extractor can
# honestly reproduce on a 2 s window — energy, F0, band slopes, voicing
# ratios, MFCC means. Excluded on purpose: formant frequencies/bandwidths/
# amplitudes (needs LPC root-solving), H1-H2 / H1-A3 (needs harmonic peak
# picking against a reliable F0). Those are the descriptors we are NOT
# willing to write twice and keep in sync.
# ---------------------------------------------------------------------------
POOL = [
    # --- loudness / energy -------------------------------------------------
    "egemaps_loudness_sma3_amean",
    "egemaps_loudness_sma3_stddevNorm",
    "egemaps_loudness_sma3_percentile20.0",
    "egemaps_loudness_sma3_percentile50.0",
    "egemaps_loudness_sma3_percentile80.0",
    "egemaps_loudness_sma3_pctlrange0-2",
    "egemaps_loudness_sma3_meanRisingSlope",
    "egemaps_loudness_sma3_stddevRisingSlope",
    "egemaps_loudness_sma3_meanFallingSlope",
    "egemaps_loudness_sma3_stddevFallingSlope",
    "egemaps_equivalentSoundLevel_dBp",
    "egemaps_loudnessPeaksPerSec",
    # --- pitch -------------------------------------------------------------
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_amean",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
    "egemaps_F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
    # --- spectral shape (voiced) ------------------------------------------
    "egemaps_alphaRatioV_sma3nz_amean",
    "egemaps_hammarbergIndexV_sma3nz_amean",
    "egemaps_slopeV0-500_sma3nz_amean",
    "egemaps_slopeV500-1500_sma3nz_amean",
    "egemaps_spectralFluxV_sma3nz_amean",
    "egemaps_mfcc1V_sma3nz_amean",
    "egemaps_mfcc2V_sma3nz_amean",
    "egemaps_mfcc3V_sma3nz_amean",
    "egemaps_mfcc4V_sma3nz_amean",
    # --- spectral shape (unvoiced) ----------------------------------------
    "egemaps_alphaRatioUV_sma3nz_amean",
    "egemaps_hammarbergIndexUV_sma3nz_amean",
    "egemaps_slopeUV0-500_sma3nz_amean",
    "egemaps_slopeUV500-1500_sma3nz_amean",
    "egemaps_spectralFluxUV_sma3nz_amean",
    # --- spectral shape (all frames) --------------------------------------
    "egemaps_spectralFlux_sma3_amean",
    "egemaps_spectralFlux_sma3_stddevNorm",
    "egemaps_mfcc1_sma3_amean",
    "egemaps_mfcc2_sma3_amean",
    "egemaps_mfcc3_sma3_amean",
    "egemaps_mfcc4_sma3_amean",
    # --- voice quality -----------------------------------------------------
    "egemaps_jitterLocal_sma3nz_amean",
    "egemaps_shimmerLocaldB_sma3nz_amean",
    "egemaps_HNRdBACF_sma3nz_amean",
    # --- rhythm / voicing --------------------------------------------------
    "egemaps_VoicedSegmentsPerSec",
    "egemaps_MeanVoicedSegmentLengthSec",
    "egemaps_MeanUnvoicedSegmentLength",
]

LOUDNESS_ONLY = ["egemaps_loudness_sma3_amean"]

# What each TypeScript descriptor is TRYING to be, for the fidelity table the
# fit prints. The TS side is a hand-written approximation, never a port —
# different windows, filterbank and pitch tracker — so this maps intent, and
# the Spearman column says how close the intent landed.
TS_TO_EGEMAPS = {
    "loudMean": "egemaps_loudness_sma3_amean",
    "loudP20": "egemaps_loudness_sma3_percentile20.0",
    "loudPctlRange": "egemaps_loudness_sma3_pctlrange0-2",
    "loudStddevNorm": "egemaps_loudness_sma3_stddevNorm",
    "loudRiseSlopeMean": "egemaps_loudness_sma3_meanRisingSlope",
    "loudRiseSlopeStd": "egemaps_loudness_sma3_stddevRisingSlope",
    "loudFallSlopeMean": "egemaps_loudness_sma3_meanFallingSlope",
    "loudFallSlopeStd": "egemaps_loudness_sma3_stddevFallingSlope",
    "f0SemitoneP80": "egemaps_F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "spectralFluxMean": "egemaps_spectralFlux_sma3_amean",
    "mfcc2Mean": "egemaps_mfcc2_sma3_amean",
    "mfcc4Mean": "egemaps_mfcc4_sma3_amean",
    "alphaRatioV": "egemaps_alphaRatioV_sma3nz_amean",
    "alphaRatioUV": "egemaps_alphaRatioUV_sma3nz_amean",
    "hammarbergUV": "egemaps_hammarbergIndexUV_sma3nz_amean",
    "slopeV0to500": "egemaps_slopeV0-500_sma3nz_amean",
}


def model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def recall_at_fa(y: np.ndarray, s: np.ndarray, fa: float = 0.05) -> float:
    thr = np.quantile(s[y == 0], 1 - fa)
    return float((s[y == 1] >= thr).mean())


def load_bank() -> pd.DataFrame:
    idx = pd.read_parquet(BANK / "index.parquet")
    eg = pd.read_parquet(BANK / "egemaps.parquet")
    return idx.join(eg, how="inner").reset_index()


def cv_auc_happy(d: pd.DataFrame, cols: list[str]) -> float:
    """Angry-vs-happy AUC, speaker-grouped 5-fold, CREMA-D only.
    This is the SELECTION objective; RAVDESS is never touched here."""
    X = d[cols].to_numpy(dtype=float)
    y = d.is_angry.to_numpy().astype(int)
    grp = d.speaker.to_numpy()
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        oof[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    hm = (d.emotion == "happy").to_numpy() | (y == 1)
    return float(roc_auc_score(y[hm], oof[hm]))


def report(df: pd.DataFrame, cols: list[str]) -> dict:
    d = df.dropna(subset=cols)
    X = d[cols].to_numpy(dtype=float)
    y = d.is_angry.to_numpy().astype(int)
    happy = (d.emotion == "happy").to_numpy()
    out: dict[str, float | int] = {"n_features": len(cols), "n": int(len(d))}

    cd = d.corpus == "cremad"
    out["cremad_cv_auc_happy"] = round(cv_auc_happy(d[cd], cols), 3)

    tr, te = cd.to_numpy(), (d.corpus == "ravdess").to_numpy()
    s = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    hm = happy[te] | (y[te] == 1)
    out["xcorp_auc_all"] = round(float(roc_auc_score(y[te], s)), 3)
    out["xcorp_auc_happy"] = round(float(roc_auc_score(y[te][hm], s[hm])), 3)
    out["xcorp_recall@5fa"] = round(recall_at_fa(y[te], s), 3)
    return out


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
def cmd_select(args: argparse.Namespace) -> int:
    df = load_bank()
    pool = [c for c in POOL if c in df.columns]
    d = df.dropna(subset=pool)
    print(f"bank: {len(d)} clips with every pooled feature · pool {len(pool)} of 88")

    cd = d[d.corpus == "cremad"]
    chosen: list[str] = []
    remaining = list(pool)
    trace: list[dict] = []
    target = min(args.max_features, len(pool))
    while len(chosen) < target:
        best, best_auc = None, -1.0
        for c in remaining:
            a = cv_auc_happy(cd, chosen + [c])
            if a > best_auc:
                best, best_auc = c, a
        chosen.append(best)
        remaining.remove(best)
        trace.append({"k": len(chosen), "added": best, "cremad_cv_auc_happy": round(best_auc, 4)})
        print(f"  +{len(chosen):2d}  {best:52}  cv angry-vs-happy {best_auc:.4f}")

    print("\n--- candidates (cross-corpus numbers were never used to select) ---")
    print(f"{'candidate':34} {'k':>3} {'cv a-vs-h':>10} {'xc all':>8} {'xc a-vs-h':>10} {'xc rec@5fa':>11}")
    results: dict[str, dict] = {}

    def row(name: str, cols: list[str]) -> None:
        r = report(d, cols)
        results[name] = {**r, "features": cols}
        print(f"{name:34} {r['n_features']:>3} {r['cremad_cv_auc_happy']:>10} "
              f"{r['xcorp_auc_all']:>8} {r['xcorp_auc_happy']:>10} {r['xcorp_recall@5fa']:>11}")

    row("loudness-only (shipped instant)", LOUDNESS_ONLY)
    row("eGeMAPS-88 (reference ceiling)", [c for c in df.columns if c.startswith("egemaps_")])
    row("implementable pool (all)", pool)
    for k in args.sizes:
        if k <= len(chosen):
            row(f"greedy-{k}", chosen[:k])

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "selection.json").write_text(json.dumps({"trace": trace, "candidates": results}, indent=1))
    print(f"\n-> {WORK / 'selection.json'}")
    return 0


# ---------------------------------------------------------------------------
# dump-pcm
# ---------------------------------------------------------------------------
def loudest_window(y: np.ndarray, sr: int, seconds: float = WINDOW_SEC) -> np.ndarray:
    """The `seconds`-long slice with the most energy — what a live rolling
    window lands on while somebody is actually talking. Short clips are
    returned whole (the extractor handles < 2 s)."""
    n = int(round(seconds * sr))
    if len(y) <= n:
        return y
    e = np.cumsum(np.concatenate([[0.0], y.astype(np.float64) ** 2]))
    best = int(np.argmax(e[n:] - e[:-n]))
    return y[best:best + n]


def cmd_dump_pcm(args: argparse.Namespace) -> int:
    import librosa
    import opensmile

    df = load_bank()
    rng = np.random.default_rng(20260919)
    # Every RAVDESS clip (the cross-corpus test set) plus a CREMA-D training
    # sample: all angry, and an equal-ish share of the other emotions.
    rav = df[df.corpus == "ravdess"]
    cd = df[df.corpus == "cremad"]
    ang = cd[cd.is_angry]
    oth = cd[~cd.is_angry]
    keep = oth.sample(n=min(args.cremad_other, len(oth)), random_state=7)
    sel = pd.concat([rav, ang, keep]).reset_index(drop=True)
    print(f"dumping {len(sel)} clips ({len(rav)} ravdess, {len(ang)} cremad angry, {len(keep)} cremad other)")

    pcm_dir = WORK / "pcm"
    pcm_dir.mkdir(parents=True, exist_ok=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    rows = []
    feats = []
    for i, r in enumerate(sel.itertuples()):
        y, _ = librosa.load(r.path, sr=SR, mono=True)
        w = loudest_window(y, SR)
        key = f"{r.corpus}_{Path(r.path).stem}"
        (pcm_dir / f"{key}.pcm").write_bytes(
            np.clip(w * 32767.0, -32768, 32767).astype("<i2").tobytes()
        )
        f = smile.process_signal(w, SR).iloc[0]
        feats.append({f"egemaps_{k}": float(v) for k, v in f.items()})
        rows.append({
            "key": key, "corpus": r.corpus, "speaker": str(r.speaker),
            "emotion": r.emotion, "is_angry": bool(r.is_angry),
            "samples": int(len(w)), "path": r.path,
        })
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(sel)}")
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(feats)], axis=1)
    out.to_parquet(WORK / "windows.parquet")
    # the TypeScript side reads plain JSON, not parquet
    (WORK / "windows.json").write_text(json.dumps(rows))
    print(f"-> {pcm_dir} ({len(rows)} files) and {WORK / 'windows.parquet'}")
    _ = rng  # reserved
    return 0


# ---------------------------------------------------------------------------
# fit-ts  — calibrate the shipped coefficients to the TypeScript extractor
# ---------------------------------------------------------------------------
def cmd_fit_ts(args: argparse.Namespace) -> int:
    """Put the model the corpus work CHOSE onto the phone's own measurements.

    Two ways to get coefficients for the TypeScript features existed, and both
    were measured on the same 2 s windows before one was picked:

      A  fit the anger labels directly
         cross-corpus angry-vs-happy 0.795, rank agreement with the openSMILE
         reference 0.958
      B  fit the openSMILE REFERENCE model's logit — a ridge regression that
         reproduces its decision function from what the phone can measure
         cross-corpus angry-vs-happy 0.795, angry-vs-all 0.857 (vs 0.854),
         rank agreement 0.974

    B ships. It is the same accuracy and a far more faithful reproduction, and
    it says the honest thing about what the phone is doing: not learning anger
    a second time off a hand-rolled feature set, but carrying the model the
    bench selected as far as a pure-TypeScript extractor can carry it. The
    accuracy claim never rests on this choice — it rests on the cross-corpus
    AUC below, measured on RAVDESS, which neither fit ever trains on.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    ts = pd.read_json(WORK / "ts_features.json")
    meta = pd.read_parquet(WORK / "windows.parquet")
    d = meta.merge(ts, on="key", how="inner")
    cols = [c for c in ts.columns if c != "key"]
    print(f"{len(d)} windows · {len(cols)} TypeScript features")

    # --- how faithfully does the TS extractor track openSMILE? -------------
    print("\nTS vs openSMILE, per feature (Spearman over the same 2 s windows):")
    fidelity = {}
    for ts_name, sm_name in TS_TO_EGEMAPS.items():
        if ts_name in d.columns and sm_name in d.columns:
            rho = float(spearmanr(d[ts_name], d[sm_name], nan_policy="omit").statistic)
            fidelity[ts_name] = round(rho, 3)
            print(f"  {ts_name:20} vs {sm_name:52} {rho:+.3f}")

    y = d.is_angry.to_numpy().astype(int)
    tr = (d.corpus == "cremad").to_numpy()
    te = (d.corpus == "ravdess").to_numpy()
    happy = (d.emotion == "happy").to_numpy()

    # --- the reference: the chosen eGeMAPS subset, on these same windows ---
    chosen = json.loads((WORK / "selection.json").read_text())["candidates"][args.candidate]["features"]
    R = d[chosen].to_numpy(dtype=float)
    R = np.where(np.isnan(R), np.nanmean(R, axis=0), R)
    ref = model().fit(R[tr], y[tr])
    ref_p = ref.predict_proba(R)[:, 1]
    ref_logit = np.log(ref_p / (1 - ref_p))

    # --- the phone's features, standardised on the training corpus ---------
    X = d[cols].to_numpy(dtype=float)
    # The extractor reports `null` for anything it could not measure (a window
    # with no voiced frame has no F0 percentile). The runtime substitutes the
    # training mean — i.e. z = 0, the feature contributes nothing — so the fit
    # has to see exactly the same substitution, or the coefficients would be
    # calibrated against a population the phone never produces.
    missing = np.isnan(X)
    if missing.any():
        print("\nunmeasurable features (null -> training mean at runtime):")
        for j, c in enumerate(cols):
            if missing[:, j].any():
                print(f"  {c:20} {int(missing[:, j].sum())} of {len(d)} windows")
    mu = np.nanmean(X[tr], axis=0)
    sd = np.nanstd(X[tr], axis=0)
    sd[sd == 0] = 1.0
    X = np.where(missing, mu, X)
    Z = (X - mu) / sd

    fit = Ridge(alpha=args.ridge).fit(Z[tr], ref_logit[tr])
    s = 1 / (1 + np.exp(-fit.predict(Z)))

    hm = happy[te] | (y[te] == 1)
    auc_all = roc_auc_score(y[te], s[te])
    auc_happy = roc_auc_score(y[te][hm], s[te][hm])
    rec = recall_at_fa(y[te], s[te])
    emo = d.emotion.isin(["angry", "happy", "neutral"]).to_numpy()
    rank = float(spearmanr(s[emo], ref_p[emo]).statistic)
    print(f"\nTS features, CREMA-D -> RAVDESS: auc_all {auc_all:.3f}  "
          f"auc_angry_vs_happy {auc_happy:.3f}  recall@5%FA {rec:.3f}")
    print(f"rank agreement with the openSMILE reference (angry/happy/neutral): {rank:.3f}")

    MODEL_OUT.write_text(json.dumps({
        "_comment": (
            "Instant tier: a plain standardised linear model over acoustic features "
            "the phone computes itself from a 2 s window of 16 kHz PCM "
            "(apps/mobile/src/live/instantTier.ts). The coefficients reproduce the "
            "openSMILE eGeMAPS reference model chosen in the bench, recalibrated onto "
            "the phone's own measurements; the accuracy claim is the held-out "
            "cross-corpus AUC below. Regenerate with scripts/instant_tier_select.py fit-ts."
        ),
        "fit_date": str(date.today()),
        "trained_on": (
            "CREMA-D (loudest 2 s window per clip): ridge regression onto the logit of "
            "the openSMILE " + args.candidate + " reference model, itself fitted on "
            "CREMA-D angry vs every other emotion"
        ),
        "evaluated_on": "RAVDESS (cross-corpus, unseen actors/rooms/scripts)",
        "metrics": {
            "xcorp_auc_angry_vs_all": round(float(auc_all), 3),
            "xcorp_auc_angry_vs_happy": round(float(auc_happy), 3),
            "xcorp_recall_at_5pct_fa": round(float(rec), 3),
            "rank_agreement_with_reference": round(rank, 3),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
        },
        "reference_features": chosen,
        "feature_fidelity_spearman_vs_opensmile": fidelity,
        "features": cols,
        "mean": [round(float(v), 6) for v in mu],
        "sd": [round(float(v), 6) for v in sd],
        "coef": [round(float(v), 6) for v in fit.coef_],
        "intercept": round(float(fit.intercept_), 6),
    }, indent=1) + "\n")
    print(f"-> {MODEL_OUT}")
    return 0



# ---------------------------------------------------------------------------
# fixture — parity vectors for jest
# ---------------------------------------------------------------------------
def cmd_fixture(args: argparse.Namespace) -> int:
    """Rank-parity vectors: for ~40 clips spread over both corpora, three
    emotions and as many speakers as possible, what the openSMILE REFERENCE
    model says about the clip's 2 s window, and what the TypeScript extractor
    measured on exactly the same samples.

    Both sides see the same audio — the reference model is fitted on the
    openSMILE functionals of the 2 s windows, not of the whole clips — so any
    disagreement the parity test catches is extraction, not framing.

    A handful of clips also carry the raw PCM (base64 int16) so jest can run
    the extractor end to end rather than trusting the stored feature vectors.
    """
    import base64

    chosen = json.loads((WORK / "selection.json").read_text())["candidates"][args.candidate]["features"]
    meta = pd.read_parquet(WORK / "windows.parquet")
    ts = pd.read_json(WORK / "ts_features.json")
    d = meta.merge(ts, on="key", how="inner")
    ts_cols = [c for c in ts.columns if c != "key"]

    # --- the openSMILE reference, fitted on CREMA-D's 2 s windows ----------
    X = d[chosen].to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    y = d.is_angry.to_numpy().astype(int)
    tr = (d.corpus == "cremad").to_numpy()
    ref = model().fit(X[tr], y[tr])
    d = d.assign(python_score=ref.predict_proba(X)[:, 1])

    # --- pick the clips: every speaker at most once per emotion ------------
    rng = np.random.default_rng(11)
    picks = []
    for corpus in ("cremad", "ravdess"):
        for emo in ("angry", "happy", "neutral"):
            g = d[(d.corpus == corpus) & (d.emotion == emo)]
            if g.empty:
                continue
            one_each = g.sample(frac=1, random_state=3).drop_duplicates("speaker")
            picks.append(one_each.sample(n=min(args.n_per, len(one_each)), random_state=int(rng.integers(1 << 30))))
    sub = pd.concat(picks).reset_index(drop=True)

    # the clips that also carry audio: spread across the emotions
    audio_keys: list[str] = []
    per_emotion = max(1, args.n_audio // 3)
    for emo in ("angry", "happy", "neutral"):
        audio_keys += list(sub[sub.emotion == emo].key.head(per_emotion))
    audio_keys = audio_keys[: args.n_audio]

    clips = []
    for r in sub.itertuples():
        entry: dict = {
            "key": r.key,
            "corpus": r.corpus,
            "speaker": str(r.speaker),
            "emotion": r.emotion,
            "isAngry": bool(r.is_angry),
            "pythonScore": round(float(r.python_score), 6),
            "tsFeatures": {
                c: (None if pd.isna(getattr(r, c)) else round(float(getattr(r, c)), 6)) for c in ts_cols
            },
        }
        if r.key in audio_keys:
            entry["pcmBase64"] = base64.b64encode((WORK / "pcm" / f"{r.key}.pcm").read_bytes()).decode()
            entry["sampleRate"] = SR
        clips.append(entry)

    FIXTURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_OUT.write_text(json.dumps({
        "_comment": (
            "Parity vectors for the instant tier (__tests__/instantTier.parity.test.ts). "
            "`pythonScore` is the openSMILE eGeMAPS reference model's P(angry) for the "
            "clip's loudest 2 s window; `tsFeatures` is what apps/mobile/src/live/"
            "instantTier.ts measured on exactly those samples. The bar is RANK agreement, "
            "not numeric equality — different windows, filterbank and pitch tracker. "
            "Clips carrying `pcmBase64` hold that 2 s window as little-endian int16 at "
            "16 kHz so the extractor can be re-run end to end; the rest carry features "
            "only, because audio does not belong in git. "
            "Regenerate: python scripts/instant_tier_select.py fixture."
        ),
        "generated": str(date.today()),
        "referenceModel": {"features": chosen, "trainedOn": "cremad 2 s windows"},
        "clips": clips,
    }, indent=1) + "\n")
    kb = FIXTURE_OUT.stat().st_size / 1024
    print(f"-> {FIXTURE_OUT}  ({len(clips)} clips, {len(audio_keys)} with audio, {kb:.0f} KB)")
    return 0



def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--max-features", type=int, default=24)
    s.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20, 24])
    s.set_defaults(fn=cmd_select)

    s = sub.add_parser("dump-pcm")
    s.add_argument("--cremad-other", type=int, default=2200)
    s.set_defaults(fn=cmd_dump_pcm)

    s = sub.add_parser("fit-ts")
    s.add_argument("--candidate", default="greedy-16")
    s.add_argument("--ridge", type=float, default=1.0)
    s.set_defaults(fn=cmd_fit_ts)

    s = sub.add_parser("fixture")
    s.add_argument("--candidate", default="greedy-16")
    s.add_argument("--n-per", type=int, default=25)
    s.add_argument("--n-audio", type=int, default=8)
    s.set_defaults(fn=cmd_fixture)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
