"""HOW HEATED IS EACH RECORDING, AND HOW WELL DO WE MEASURE IT? — one dot per recording.

The owner's ask: a graph where he can see, for many real recordings, how
volatile each conversation is on one axis and how well the app measures it on
the other — so it is visible at a glance whether we do fine on calm
conversations and badly on heated ones (or the reverse).

Every recording in every corpus we hold becomes one point:

  x  VOLATILITY — how much the loudness moves around the speaker's own baseline,
     from the SAME window/baseline machinery the watch runs (a faithful port,
     conversation_audit.py). Two candidates are recorded; the primary is the
     standard deviation of dB-over-baseline across voiced windows, the other is
     the fraction of windows that clear the +6 dB first rung.
  y  HOW WELL WE MEASURE — several metrics, each only where the ground truth for
     it exists, never fabricated:
       dose_per_hour     buzzes/hour the shipped chain would deliver (all recordings)
       attribution_acc   share of turns attributed to the right speaker
                         (recordings with speaker ground truth + a scored run)
       nudge_hit_rate    expected heated moments the coach caught
       nudge_false_rate  nudges fired where nothing heated was labelled
                         (recordings with heat labels — today only the scenes)
  colour  the CORPUS, and the SETTING it came from (meeting / family / game /
          scripted argument / ...). Diversity is the point of the exercise.

Corpora plug in through manifests, so the three researchers' finds drop
straight in: any tmp/corpora/*/heatmap_manifest.json is merged, one entry per
recording:

  {"id": "...", "corpus": "...", "setting": "...", "audio": "abs-or-relative.wav",
   "speakers": {"S0": [[a,b],...]} | null, "self": "S0" | null,
   "heated_spans": [[a,b],...] | null, "notes": "..."}

    python scripts/heat_map.py        # writes tmp/heat-map/heat_map.{json,html}
"""
from __future__ import annotations

import glob
import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import conversation_audit as ca  # noqa: E402

OUT = REPO / "tmp/heat-map"
FIXTURES = REPO / "server/tests/fixtures/audio"

FIXTURE_SETTINGS = {
    "family_real": ("owner recordings", "family, father & son"),
    "poker6_real": ("owner recordings", "poker game, six friends"),
    "scene_couple_escalation": ("TTS scenes", "scripted couple argument"),
    "scene_family3": ("TTS scenes", "scripted family, teen shouts"),
    "scene_meeting4": ("TTS scenes", "scripted meeting, one flare"),
    "gptaudio": ("TTS scenes", "scripted, older generator"),
    "openai": ("TTS scenes", "scripted, older generator"),
}


# ------------------------------------------------------------- manifests --

def ami_entries() -> list[dict]:
    mp = REPO / "tmp/ami-corpus/manifest.json"
    if not mp.exists():
        return []
    m = json.loads(mp.read_text())
    return [{
        "id": x["meeting"], "corpus": "AMI", "setting": "office meeting, 4 people",
        "audio": str(REPO / "tmp/ami-corpus" / x["mix"]),
        "speakers": x["speakers"], "self": "S0", "heated_spans": None,
        "duration_s": x["duration_s"],
    } for x in m["meetings"]]


def _fixture_turns(meta: dict) -> list[dict]:
    """The fixtures store turn timing three different ways:
    real recordings carry start_time/end_time from the pipeline; poker6 carries
    approx_start/approx_end; the TTS scenes carry only each turn's duration_sec
    plus a fixed silence_gap_sec between turns, so their boundaries are
    cumulative — the same reconstruction the replay's parseSceneMeta does."""
    if meta.get("turns") and "start_time" in meta["turns"][0]:
        return [{"speaker": t["speaker"], "start_time": float(t["start_time"]), "end_time": float(t["end_time"])}
                for t in meta["turns"]]
    if meta.get("approx_turns"):
        return [{"speaker": t["speaker"], "start_time": float(t["approx_start"]), "end_time": float(t["approx_end"])}
                for t in meta["approx_turns"]]
    if meta.get("turns") and "duration_sec" in meta["turns"][0]:
        gap = float(meta.get("silence_gap_sec", 0.0))
        out, t0 = [], 0.0
        for t in meta["turns"]:
            d = float(t["duration_sec"])
            out.append({"speaker": t["speaker"], "start_time": round(t0, 3), "end_time": round(t0 + d, 3)})
            t0 += d + gap
        return out
    return []


def fixture_entries() -> list[dict]:
    out = []
    for meta_path in sorted(FIXTURES.glob("test_recording_*_meta.json")):
        name = meta_path.name[len("test_recording_"):-len("_meta.json")]
        wav = FIXTURES / f"test_recording_{name}.wav"
        if not wav.exists():
            continue
        meta = json.loads(meta_path.read_text())
        turns = _fixture_turns(meta)
        speakers: dict[str, list[list[float]]] = {}
        for t in turns:
            speakers.setdefault(t["speaker"], []).append([t["start_time"], t["end_time"]])
        heated = None
        if meta.get("expected_nudges"):
            heated = []
            for n in meta["expected_nudges"]:
                i = n["after_turn_index"]
                if 0 <= i < len(turns):
                    heated.append([turns[i]["start_time"], turns[i]["end_time"]])
        corpus, setting = FIXTURE_SETTINGS.get(name, ("fixtures", name))
        with wave.open(str(wav)) as w:
            dur = w.getnframes() / w.getframerate()
        out.append({
            "id": name, "corpus": corpus, "setting": setting, "audio": str(wav),
            "speakers": speakers or None, "self": meta.get("self_speaker") or (turns[0]["speaker"] if turns else None),
            "heated_spans": heated, "duration_s": round(dur, 1),
        })
    return out


def extra_entries() -> list[dict]:
    out = []
    for mp in glob.glob(str(REPO / "tmp/corpora/*/heatmap_manifest.json")):
        base = Path(mp).parent
        for e in json.loads(Path(mp).read_text()):
            a = Path(e["audio"])
            e["audio"] = str(a if a.is_absolute() else base / a)
            if Path(e["audio"]).exists():
                out.append(e)
    return out


# --------------------------------------------------------------- metrics --

def load_pcm(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype("<i2")
    return x, sr


def volatility(over: np.ndarray) -> dict:
    voiced = over[over != 0.0]
    if len(voiced) < 3:
        return {"db_sd": 0.0, "frac_over_6db": 0.0, "db_p95": 0.0}
    return {
        "db_sd": round(float(np.std(voiced)), 2),
        "frac_over_6db": round(float((voiced >= 6.0).mean()), 3),
        "db_p95": round(float(np.percentile(voiced, 95)), 2),
    }


def nudge_metrics(felt: list[tuple[float, str, int]], heated: list[list[float]] | None, dur_s: float) -> dict:
    """Escalations vs labelled heated spans, with a 3 s grace after a span
    (the loop acts when a turn CLOSES, so a hit lands just after the span)."""
    if not heated:
        return {}
    esc = [t for t, kind, _ in felt if kind == "escalation"]
    hit = sum(1 for a, b in heated if any(a <= t <= b + 3.0 for t in esc))
    false = sum(1 for t in esc if not any(a <= t <= b + 3.0 for a, b in heated))
    return {
        "nudge_hit_rate": round(hit / len(heated), 3),
        "nudge_false_rate": round(false / max(len(esc), 1), 3),
        "nudge_expected": len(heated), "nudge_hits": hit, "nudge_false": false,
    }


def attribution_from_identity(entry: dict) -> float | None:
    """AMI attribution at the SHIPPED threshold, from identity_audit's output."""
    p = REPO / "tmp/ami-corpus/identity.json"
    if not p.exists() or entry["corpus"] != "AMI":
        return None
    sys.path.insert(0, str(REPO / "server"))
    try:
        import speaker_id as sid
        thr = sid.MATCH_THRESHOLD
    except Exception:
        thr = 0.60
    rows = [r for r in json.loads(p.read_text()) if r["meeting"] == entry["id"] and r["is_self"]]
    if not rows:
        return None
    return round(float(np.mean([r["score"] >= thr for r in rows])), 3)


def attribution_from_nudge_report(entry: dict) -> float | None:
    """Scene attribution from the newest replay report (the loop's real verdicts)."""
    reports = sorted(glob.glob(str(REPO / "tmp/nudge-report-*.json")))
    if not reports:
        return None
    d = json.loads(Path(reports[-1]).read_text())
    for s in d.get("scenes", []):
        if s["scene"] == entry["id"]:
            turns = s.get("turns", [])
            ok = [t.get("attributionOk") for t in turns if t.get("attributionOk") is not None]
            return round(float(np.mean(ok)), 3) if ok else None
    return None


def measure(entry: dict) -> dict:
    pcm, sr = load_pcm(entry["audio"])
    dbfs = ca.windows_dbfs(pcm, sr)
    over = ca.db_over_baseline(dbfs)
    felt = ca.replay(over, pulse_on=False, reminder_backoff=True)
    dur = entry.get("duration_s") or len(pcm) / sr
    hours = dur / 3600.0
    row = {
        "id": entry["id"], "corpus": entry["corpus"], "setting": entry["setting"],
        "duration_min": round(dur / 60, 1),
        **volatility(over),
        "dose_per_hour": round(len(felt) / hours, 1) if hours else None,
        "buzzes": len(felt),
        **nudge_metrics(felt, entry.get("heated_spans"), dur),
        "has_speaker_truth": bool(entry.get("speakers")),
        "has_heat_labels": bool(entry.get("heated_spans")),
    }
    att = attribution_from_identity(entry)
    if att is None:
        att = attribution_from_nudge_report(entry)
    if att is not None:
        row["attribution_acc"] = att
    return row


# ---------------------------------------------------------------- render --

def render_html(rows: list[dict], out: Path) -> None:
    data = json.dumps(rows)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>How heated, how well measured</title>
<style>
:root{{--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0b0f17;--fg:#e5e7eb;--muted:#9ca3af;--line:#1f2937;--card:#111827}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;padding:16px;max-width:1000px;margin-inline:auto}}
h1{{font-size:18px;margin:0 0 4px}} p.sub{{color:var(--muted);margin:0 0 12px;font-size:13.5px}}
.controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:10px 0;font-size:13.5px}}
select{{font:inherit;padding:4px 6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}}
svg{{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
.axis text{{fill:var(--muted);font-size:11px}} .axis line,.axis path{{stroke:var(--line)}}
.dot{{cursor:pointer;stroke:var(--bg);stroke-width:1.2}} .dot:hover{{stroke:var(--fg);stroke-width:2}}
.legend{{display:flex;gap:12px;flex-wrap:wrap;font-size:12.5px;margin:8px 0}} .legend span{{display:inline-flex;align-items:center;gap:5px}}
.sw{{width:11px;height:11px;border-radius:50%;display:inline-block}}
#tip{{position:fixed;pointer-events:none;background:var(--fg);color:var(--bg);padding:8px 10px;border-radius:8px;font-size:12.5px;display:none;max-width:320px;line-height:1.4}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:14px}} th,td{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}} th{{color:var(--muted)}}
.note{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13px;color:var(--muted);margin:10px 0}}
</style></head><body>
<h1>How heated is each recording — and how well do we measure it?</h1>
<p class="sub">One dot per recording. Left–right: how much the loudness moves around the speaker's own baseline. Up–down: pick a metric. Hover a dot.</p>
<div class="controls">
  <label>y-axis <select id="ymetric"></select></label>
  <label>x-axis <select id="xmetric">
    <option value="db_sd">volatility — sd of dB over baseline</option>
    <option value="frac_over_6db">share of windows over the +6 dB rung</option>
    <option value="db_p95">95th-percentile dB over baseline</option>
    <option value="ref_arousal_sd">volatility by the tone model — sd of arousal (independent of loudness)</option>
    <option value="ref_arousal_mean">how aroused the tone model hears the whole recording</option>
  </select></label>
  <label><input type="checkbox" id="size"> dot size = duration</label>
</div>
<div class="legend" id="legend"></div>
<svg id="plot" viewBox="0 0 900 520"></svg>
<div id="tip"></div>
<div class="note" id="note"></div>
<table id="tbl"></table>
<script>
const rows = {data};
const METRICS = {{
  agreement_loudness_vs_arousal: {{label:"how well our loudness-heat agrees with the tone model's arousal (Spearman, per recording)", good:"high"}},
  loud_but_pleasant: {{label:"share of LOUD windows the tone model calls PLEASANT — anger mistaken for joy", good:"low"}},
  dose_per_hour:   {{label:"buzzes per hour the wrist would deliver (lower is better on calm audio)", good:"low"}},
  attribution_acc: {{label:"share of turns attributed to the right speaker", good:"high"}},
  nudge_hit_rate:  {{label:"labelled heated moments the coach caught", good:"high"}},
  nudge_false_rate:{{label:"share of nudges that fired on nothing heated", good:"low"}},
}};
const PALETTE = ["#2563eb","#dc2626","#16a34a","#d97706","#7c3aed","#0891b2","#be185d","#4b5563"];
const corpora = [...new Set(rows.map(r=>r.corpus))];
const color = c => PALETTE[corpora.indexOf(c) % PALETTE.length];
const ysel = document.getElementById("ymetric"), xsel = document.getElementById("xmetric");
for (const [k,v] of Object.entries(METRICS)) {{
  const n = rows.filter(r=>r[k]!=null).length;
  const o = document.createElement("option"); o.value=k; o.textContent=`${{v.label}}  (${{n}} recordings)`; ysel.appendChild(o);
}}
document.getElementById("legend").innerHTML = corpora.map(c=>`<span><i class="sw" style="background:${{color(c)}}"></i>${{c}} (${{rows.filter(r=>r.corpus===c).length}})</span>`).join("");
const svg = document.getElementById("plot"), tip = document.getElementById("tip");
function draw(){{
  const yk = ysel.value, xk = xsel.value, sized = document.getElementById("size").checked;
  const pts = rows.filter(r=>r[yk]!=null && r[xk]!=null);
  const W=900,H=520,L=64,R=20,T=18,B=54;
  const xs = pts.map(p=>p[xk]), ys = pts.map(p=>p[yk]);
  const xmax = Math.max(1e-6, ...xs)*1.08;
  // Correlations run negative — and the negative dots ARE the finding on the
  // synthetic scenes — so the y-range must include them rather than clip them.
  const yhi = Math.max(1e-6, ...ys)*1.12, ylo = Math.min(0, ...ys)*1.12;
  const X = v => L + v/xmax*(W-L-R), Y = v => T + (yhi - v)/(yhi - ylo)*(H-T-B);
  const yfmt = v => (yhi - ylo) <= 2.5 ? v.toFixed(2) : v.toFixed(0);
  const xfmt = v => xk==="frac_over_6db" ? v.toFixed(2) : (xmax <= 2.5 ? v.toFixed(2) : v.toFixed(1));
  let s = `<g class="axis">`;
  for (let i=0;i<=5;i++) {{ const v=xmax*i/5, x=X(v); s+=`<line x1="${{x}}" y1="${{T}}" x2="${{x}}" y2="${{H-B}}"/><text x="${{x}}" y="${{H-B+16}}" text-anchor="middle">${{xfmt(v)}}</text>`; }}
  for (let i=0;i<=5;i++) {{ const v=ylo + (yhi-ylo)*i/5, y=Y(v); s+=`<line x1="${{L}}" y1="${{y}}" x2="${{W-R}}" y2="${{y}}"/><text x="${{L-6}}" y="${{y+4}}" text-anchor="end">${{yfmt(v)}}</text>`; }}
  if (ylo < 0) {{ const y0=Y(0); s+=`<line x1="${{L}}" y1="${{y0}}" x2="${{W-R}}" y2="${{y0}}" stroke="currentColor" stroke-opacity="0.45" stroke-dasharray="4 3"/><text x="${{W-R-4}}" y="${{y0-4}}" text-anchor="end">0 — no agreement</text>`; }}
  s += `<text x="${{(L+W-R)/2}}" y="${{H-8}}" text-anchor="middle">${{xsel.selectedOptions[0].textContent}}</text>`;
  s += `<text transform="translate(14,${{(T+H-B)/2}}) rotate(-90)" text-anchor="middle">${{METRICS[yk].label}}</text></g>`;
  for (const p of pts) {{
    const r = sized ? 4 + Math.sqrt(p.duration_min||1)*1.6 : 7;
    s += `<circle class="dot" data-id="${{p.id}}" cx="${{X(p[xk])}}" cy="${{Y(p[yk])}}" r="${{r}}" fill="${{color(p.corpus)}}" fill-opacity="0.85"/>`;
  }}
  svg.innerHTML = s;
  const missing = rows.length - pts.length;
  document.getElementById("note").textContent = missing
    ? `${{missing}} recording(s) not plotted for this metric — no ground truth of that kind exists for them, and nothing is estimated.`
    : `All ${{rows.length}} recordings plotted.`;
  svg.querySelectorAll(".dot").forEach(d=>{{
    d.onmousemove = e => {{
      const p = rows.find(r=>r.id===d.dataset.id);
      tip.style.display="block"; tip.style.left=(e.clientX+12)+"px"; tip.style.top=(e.clientY+12)+"px";
      tip.innerHTML = `<b>${{p.id}}</b> · ${{p.corpus}}<br>${{p.setting}} · ${{p.duration_min}} min<br>` +
        `volatility sd ${{p.db_sd}} dB · ${{(p.frac_over_6db*100).toFixed(0)}}% of windows over +6<br>` +
        (p.agreement_loudness_vs_arousal!=null?`tone-model agreement ${{p.agreement_loudness_vs_arousal}} · loud-but-pleasant ${{(p.loud_but_pleasant*100).toFixed(0)}}%<br>`:"") +
        `dose ${{p.dose_per_hour}}/h` + (p.attribution_acc!=null?` · attribution ${{(p.attribution_acc*100).toFixed(0)}}%`:"") +
        (p.nudge_hit_rate!=null?`<br>nudges: caught ${{p.nudge_hits}}/${{p.nudge_expected}}, ${{p.nudge_false}} false`:"");
    }};
    d.onmouseleave = () => tip.style.display="none";
  }});
  const cols = ["id","corpus","setting","duration_min","db_sd","ref_arousal_sd","agreement_loudness_vs_arousal","loud_but_pleasant","dose_per_hour","attribution_acc","nudge_hit_rate","nudge_false_rate"];
  document.getElementById("tbl").innerHTML = `<tr>${{cols.map(c=>`<th>${{c}}</th>`).join("")}}</tr>` +
    rows.slice().sort((a,b)=>b.db_sd-a.db_sd).map(r=>`<tr>${{cols.map(c=>`<td>${{r[c]==null?"—":r[c]}}</td>`).join("")}}</tr>`).join("");
}}
ysel.onchange = xsel.onchange = document.getElementById("size").onchange = draw;
draw();
</script></body></html>"""
    out.write_text(html)


def merge_reference(rows: list[dict]) -> int:
    """Fold in heat_reference.py's second opinion where it has been computed.
    Absent for a recording => the reference metrics are simply missing for it,
    and the plot says so rather than filling anything in."""
    p = OUT / "heat_reference.json"
    if not p.exists():
        return 0
    ref = {r["id"]: r for r in json.loads(p.read_text())}
    n = 0
    for row in rows:
        r = ref.get(row["id"])
        if not r:
            continue
        n += 1
        row["agreement_loudness_vs_arousal"] = (
            None if r["agreement_loudness_vs_arousal"] is None else round(r["agreement_loudness_vs_arousal"], 3)
        )
        row["ref_arousal_mean"] = r["ref_arousal_mean"]
        row["ref_arousal_sd"] = r["ref_arousal_sd"]
        row["ref_valence_mean"] = r["ref_valence_mean"]
        row["loud_but_pleasant"] = r["loud_but_pleasant"]
        row["ref_windows"] = r["windows"]
    return n


def main() -> int:
    entries = ami_entries() + fixture_entries() + extra_entries()
    if not entries:
        raise SystemExit("no recordings found")
    rows = []
    for e in entries:
        try:
            rows.append(measure(e))
            print(f"  {e['corpus']:16} {e['id']:26} sd {rows[-1]['db_sd']:5.2f} dB  dose {rows[-1]['dose_per_hour']:6.1f}/h", flush=True)
        except Exception as exc:
            print(f"  {e['id']}: FAILED {exc}")
    OUT.mkdir(parents=True, exist_ok=True)
    n_ref = merge_reference(rows)
    print(f"  reference (tone model) merged for {n_ref}/{len(rows)} recordings")
    (OUT / "heat_map.json").write_text(json.dumps(rows, indent=1))
    render_html(rows, OUT / "heat_map.html")
    print(f"\n{len(rows)} recordings -> {OUT/'heat_map.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
