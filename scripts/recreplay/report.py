"""The HTML report for one run bundle: mobile-first, light + dark, no
external assets (it is opened from tmp/ on a laptop or a phone)."""

from __future__ import annotations

import html
import json
from pathlib import Path

PALETTE = ["#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2", "#65a30d"]


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt_t(t) -> str:
    if t is None:
        return "–"
    t = float(t)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _ms(v) -> str:
    return "–" if v is None else (f"{v / 1000:.1f} s" if abs(v) >= 1000 else f"{v:.0f} ms")


def _score_txt(v) -> str:
    return "" if not isinstance(v, (int, float)) else f"{v:.2f}"


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _synthetic_flags(b: dict) -> list[str]:
    flags = []
    for a in b.get("annotations") or []:
        if "SYNTHETIC" in str(a.get("model") or "").upper():
            flags.append(f"annotation “{a['label']}” is SYNTHETIC ({a.get('model')})")
    raw_notes = (b.get("notes") or {}).get("setting") or ""
    if "SYNTHETIC" in raw_notes.upper() or any("SYNTHETIC" in (m.get("text") or "").upper() for m in (b.get("notes") or {}).get("moments") or []):
        flags.append("the notes are SYNTHETIC (not written by the owner)")
    return flags


def _speaker_colors(b: dict, segs: list[dict]) -> dict[str, str]:
    labels = []
    for s in segs:
        if s["label"] not in labels:
            labels.append(s["label"])
    wearer = next((s["label"] for s in segs if s["wearer"]), None)
    out = {}
    if wearer:
        out[wearer] = PALETTE[0]
    i = 1
    for lab in labels:
        if lab not in out:
            out[lab] = PALETTE[i % len(PALETTE)]
            i += 1
    return out


def _speaker_names(b: dict, segs: list[dict]) -> dict[str, str]:
    names = {}
    primary = b.get("primary_annotation")
    descs = {}
    for a in b.get("annotations") or []:
        if a.get("label") == primary and a.get("aligned"):
            descs = {s["id"]: s.get("voice_description") for s in a["aligned"].get("speakers", [])}
    for s in segs:
        lab = s["label"]
        if lab in names:
            continue
        d = descs.get(lab)
        names[lab] = ("Owner · " if s["wearer"] else "") + lab + (f" ({d})" if d else "")
    return names


# ---------------------------------------------------------------------------
# Overview strip (SVG in a viewBox — scales to the screen, never scrolls)
# ---------------------------------------------------------------------------

def overview_svg(b: dict, sc: dict, segs: list[dict], colors: dict[str, str]) -> str:
    """Every lane is ROW px tall in a fixed-height SVG that only stretches
    horizontally (preserveAspectRatio=none), so the HTML lane labels line up
    at any width; no text inside the SVG (it would stretch) — the time axis
    is HTML underneath."""
    dur = max(float((b.get("audio") or {}).get("duration_s") or 1.0), 1.0)
    W = 1000.0
    ROW = 28
    lanes = []
    for s in segs:
        if s["label"] not in lanes:
            lanes.append(s["label"])
    wearer = next((s["label"] for s in segs if s["wearer"]), None)
    if wearer in lanes:
        lanes.remove(wearer)
        lanes.insert(0, wearer)
    n_rows = len(lanes) + 4
    H = n_rows * ROW
    x = lambda t: (max(0.0, min(float(t), dur)) / dur) * W  # noqa: E731
    parts = []
    step = 5 if dur <= 60 else 15 if dur <= 300 else 60 if dur <= 1200 else 300
    for t in range(0, int(dur) + 1, step):
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{H}" class="tick"/>')
    for i, lab in enumerate(lanes):
        y = i * ROW + 3
        for s in segs:
            if s["label"] == lab:
                parts.append(f'<rect x="{x(s["start"]):.1f}" y="{y}" width="{max(2.0, x(s["end"]) - x(s["start"])):.1f}" height="{ROW - 6}" rx="2" fill="{colors.get(lab)}"><title>{esc(lab)} {_fmt_t(s["start"])}: {esc(s.get("text"))}</title></rect>')
    r = len(lanes)
    for h in sc.get("heat") or []:
        lvl = h.get("intensity")
        if lvl is None:
            continue
        op = [0.08, 0.35, 0.65, 0.95][max(0, min(3, int(lvl)))]
        parts.append(f'<rect x="{x(h["start"]):.1f}" y="{r * ROW + 7}" width="{max(2.0, x(h["end"]) - x(h["start"])):.1f}" height="{ROW - 14}" fill="var(--heat)" opacity="{op}"><title>heat {lvl} · {esc(h.get("emotion"))} ({esc(h.get("label"))})</title></rect>')
    r += 1
    for ln in sc.get("lines") or []:
        cls = {"nudge": "f-nudge", "response": "f-resp", "haptic": "f-hap", "positive": "f-pos"}.get(ln["kind"], "f-room")
        w = 7 if ln["fires"] else 4
        parts.append(f'<rect x="{x(ln["at_s"]) - w / 2:.1f}" y="{r * ROW + 5}" width="{w}" height="{ROW - 10}" rx="2" class="{cls}{"" if ln["fires"] else " quiet"}"><title>{esc(ln["kind"])} @ {_fmt_t(ln["at_s"])} ({_ms(ln.get("latency_ms"))}): {esc(ln["text"])}</title></rect>')
    r += 1
    for m in (sc.get("moments") or {}).get("items") or []:
        cls = "m-hit" if m["hit"] else "m-miss"
        top = r * ROW + (4 if m["source"] == "owner" else ROW // 2)
        parts.append(f'<rect x="{x(m["t"]) - 4:.1f}" y="{top}" width="8" height="{ROW // 2 - 4}" class="{cls}"><title>{esc(m["source"])} moment {_fmt_t(m["t"])}: {esc(m["text"])} — {"caught" if m["hit"] else "missed"}</title></rect>')
    r += 1
    for e in (sc.get("identity") or {}).get("timeline") or []:
        if not e["is_self"]:
            continue
        cls = "id-ok" if e.get("ok") else ("id-bad" if e.get("ok") is False else "id-unk")
        top = r * ROW + (4 if e["who"] == "phone" else ROW // 2)
        parts.append(f'<rect x="{x(e["t"]) - 3:.1f}" y="{top}" width="6" height="{ROW // 2 - 4}" class="{cls}"><title>{esc(e["who"])} says {esc(e["label"])} is the owner @ {_fmt_t(e["t"])} ({"right" if e.get("ok") else "WRONG" if e.get("ok") is False else "?"})</title></rect>')
    labels = [*(f"{lab} (you)" if lab == wearer else lab for lab in lanes), "heat", "coach", "moments ★top ◆bottom", "“that's you” 📱top 🖥bottom"]
    legend_rows = "".join(f'<li title="{esc(lb)}">{esc(lb)}</li>' for lb in labels)
    axis = "".join(f'<span style="left:{100 * t / dur:.2f}%">{_fmt_t(t)[:-2]}</span>' for t in range(0, int(dur) + 1, step)
                   if t < dur * 0.97)
    return (f'<div class="strip"><ul class="lanes">{legend_rows}</ul><div>'
            f'<svg viewBox="0 0 {W:.0f} {H}" width="100%" height="{H}" preserveAspectRatio="none" role="img" aria-label="timeline overview">{"".join(parts)}</svg>'
            f'<div class="axis">{axis}</div></div></div>')


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1c1c1e;--muted:#6b6b70;--line:#e3e3e0;--accent:#2563eb;--good:#15803d;--bad:#b91c1c;--warn:#b45309;--heat:#dc2626;--chip:#f0f0ec}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#111214;--card:#1b1c1f;--ink:#ececf0;--muted:#a0a0a8;--line:#2c2d31;--accent:#60a5fa;--good:#4ade80;--bad:#f87171;--warn:#fbbf24;--heat:#f87171;--chip:#26272b}}
:root[data-theme="dark"]{--bg:#111214;--card:#1b1c1f;--ink:#ececf0;--muted:#a0a0a8;--line:#2c2d31;--accent:#60a5fa;--good:#4ade80;--bad:#f87171;--warn:#fbbf24;--heat:#f87171;--chip:#26272b}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:1.35rem;margin:.2rem 0}h2{font-size:1.05rem;margin:0 0 .5rem}
.sub{color:var(--muted);font-size:.85rem}
.banner{background:color-mix(in srgb,var(--warn) 16%,var(--card));border:1px solid var(--warn);border-radius:10px;padding:10px 12px;margin:12px 0;font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.tile b{display:block;font-size:1.35rem;font-variant-numeric:tabular-nums}
.tile span{color:var(--muted);font-size:.8rem}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}.muted{color:var(--muted)}
.strip{display:grid;grid-template-columns:auto 1fr;gap:8px;align-items:start}
.strip svg{display:block}
.lanes{list-style:none;margin:0;padding:0;font-size:.72rem;color:var(--muted)}
.lanes li{height:28px;line-height:28px;white-space:nowrap;max-width:7rem;overflow:hidden;text-overflow:ellipsis}
.axis{position:relative;height:16px;font-size:.68rem;color:var(--muted)}
.axis span{position:absolute;top:2px;transform:translateX(-2px)}
.tick{stroke:var(--line);stroke-width:1}.ax{fill:var(--muted);font-size:10px}
.f-nudge{fill:var(--accent)}.f-resp{fill:#059669}.f-hap{fill:var(--warn)}.f-pos{fill:var(--good)}.f-room{fill:#7c3aed}.quiet{opacity:.35}
.m-hit{fill:var(--good)}.m-miss{fill:var(--bad)}
.id-ok{fill:var(--good)}.id-bad{fill:var(--bad)}.id-unk{fill:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{text-align:left;padding:6px 4px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tl{list-style:none;margin:0;padding:0}
.tl li{display:grid;grid-template-columns:3.4rem 1fr;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)}
.tl .t{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.8rem;padding-top:2px}
.say{border-left:4px solid var(--c,#888);padding-left:8px}
.say .who{font-size:.75rem;color:var(--muted)}
.coach{background:color-mix(in srgb,var(--accent) 10%,var(--card));border:1px solid color-mix(in srgb,var(--accent) 35%,var(--line));border-radius:10px;padding:6px 8px}
.coach.phone{background:color-mix(in srgb,var(--warn) 10%,var(--card));border-color:color-mix(in srgb,var(--warn) 35%,var(--line))}
.coach.quiet{opacity:.6}
.pill{display:inline-block;border-radius:999px;padding:0 7px;font-size:.72rem;background:var(--chip);color:var(--muted);margin-right:4px;white-space:nowrap}
.pill.ok{color:var(--good)}.pill.bad{color:var(--bad)}
.mom{border:1px dashed var(--muted);border-radius:10px;padding:4px 8px;font-size:.85rem}
details>summary{cursor:pointer;font-weight:600}
code{font-size:.8rem;word-break:break-word}
ul.plain{margin:.3rem 0;padding-left:1.1rem}
.legend{font-size:.78rem;color:var(--muted)}
@media (max-width:520px){.lanes li{max-width:4.2rem}.tl li{grid-template-columns:2.9rem 1fr}}
"""


def render(b: dict) -> str:
    from .score import truth_segments  # local import: report has no other score dependency

    sc = b.get("score") or {}
    segs, truth_source = truth_segments(b)
    colors = _speaker_colors(b, segs)
    names = _speaker_names(b, segs)
    lat = sc.get("latency") or {}
    mom = sc.get("moments") or {}
    ident = sc.get("identity") or {}
    ph = ident.get("phone") or {}
    sv = ident.get("server") or {}
    viol = sc.get("violations") or []
    settings = b.get("settings") or {}
    flags = _synthetic_flags(b)
    srv = b.get("server") or {}

    def tile(value, label, cls=""):
        return f'<div class="tile"><b class="{cls}">{value}</b><span>{label}</span></div>'

    hits_cls = "good" if mom.get("total") and mom.get("hits") == mom.get("total") else ("warn" if mom.get("hits") else "bad")
    tiles = "".join([
        tile(f"{mom.get('hits', 0)}/{mom.get('total', 0)}", f"moments the coach caught (±{mom.get('window_s', 0):g} s)", hits_cls if mom.get("total") else "muted"),
        tile(str(len(mom.get("unmatched_lines") or [])), "lines with no moment nearby", "warn" if mom.get("unmatched_lines") else "good"),
        tile(_pct(ph.get("accuracy")), f"phone knew who spoke ({ph.get('correct', 0)}/{ph.get('decided', 0)} decided turns)",
             "good" if (ph.get("accuracy") or 0) >= 0.8 else "warn"),
        tile(_fmt_t(ph.get("first_confirmed_s")) if ph.get("first_confirmed_s") is not None else "never",
             "phone first said “that's you” (correctly)", "good" if ph.get("first_confirmed_s") is not None else "bad"),
        tile(_ms(lat.get("p50_ms")), f"coach latency p50 (p90 {_ms(lat.get('p90_ms'))})"),
        tile(_ms(lat.get("first_words_p50_ms")), "first words on screen p50"),
        tile(str(len(viol)), "possible invented facts / scripting", "bad" if viol else "good"),
        tile(str(len(sc.get("errors") or [])), "coach errors (no line produced)", "bad" if sc.get("errors") else "good"),
    ])

    # timeline rows
    rows: list[tuple[float, int, str]] = []
    for s in segs:
        col = colors.get(s["label"], "#888")
        emo = f'<span class="pill">{esc(s["emotion"])}{" · heat " + str(s["intensity"]) if s.get("intensity") is not None else ""}</span>' if s.get("emotion") else ""
        rows.append((s["start"], 0, f'<div class="say" style="--c:{col}"><div class="who">{esc(names.get(s["label"], s["label"]))} {emo}</div>{esc(s.get("text"))}</div>'))
    for r in ph.get("rows") or []:
        pred = r["pred"]
        lab = "owner" if pred else ("someone else" if pred is False else "unknown voice")
        okc = "ok" if r["ok"] else ("bad" if r["ok"] is False else "")
        sc_txt = f" · match {r['score']:.2f}" if isinstance(r.get("score"), (int, float)) else ""
        rows.append((r["sent_at"], 1, f'<div class="legend">📱 phone sent turn {_fmt_t(r["start"])}–{_fmt_t(r["end"])} as <span class="pill {okc}">{esc(r["speaker"])} = {lab}{sc_txt}</span>{"" if r["truth"] is None else ("✓" if r["ok"] else "✗" if r["ok"] is False else "")}</div>'))
    for r in sv.get("rows") or []:
        okc = "ok" if r["ok"] else ("bad" if r["ok"] is False else "")
        rows.append((r["t"], 1, f'<div class="legend">🖥 server voiceprint: <span class="pill {okc}">{esc(r["label"])} {"IS" if r["is_self"] else "is not"} the owner · {r["score"]}</span></div>'))
    for ln in sc.get("lines") or []:
        icon = {"nudge": "🎧 nudge", "response": "🎧 say", "haptic": "📳 buzz", "positive": "📳 positive", "room_card": "🗂 card", "room_answer": "🔊 answer"}.get(ln["kind"], ln["kind"])
        extra = "".join(f"<li>{esc(x)}</li>" for x in (ln.get("all") or [])[1:])
        lat_txt = f'{_ms(ln.get("latency_ms"))} after the turn ended' if ln.get("latency_ms") is not None else ""
        fw = f' · first words {_ms(ln.get("first_words_ms"))}' if ln.get("first_words_ms") not in (None, ln.get("latency_ms")) else ""
        quiet = "" if ln["fires"] else " quiet"
        rows.append((ln["at_s"], 2, f'<div class="coach {ln["source"]}{quiet}"><span class="pill">{icon}</span>'
                     f'{"" if ln["fires"] else "<span class=pill>silent (low importance)</span>"}<b>{esc(ln["text"])}</b>'
                     f'<div class="legend">{lat_txt}{fw}{" · on: “" + esc((ln.get("utterance_text") or "")[:60]) + "”" if ln.get("utterance_text") else ""}</div>'
                     f'{"<ul class=plain legend>" + extra + "</ul>" if extra else ""}</div>'))
    for m in mom.get("items") or []:
        icon = "★ owner" if m["source"] == "owner" else "◆ annotation"
        rows.append((m["t"], 3, f'<div class="mom"><span class="pill {"ok" if m["hit"] else "bad"}">{icon} moment · {"caught" if m["hit"] else "missed"}</span>{esc(m["text"])}</div>'))
    rows.sort(key=lambda r: (r[0], r[1]))
    timeline = "".join(f'<li><div class="t">{_fmt_t(t)}</div><div>{body}</div></li>' for t, _, body in rows)

    mom_rows = "".join(
        f'<tr><td class="num">{_fmt_t(m["t"])}</td><td>{"★ owner" if m["source"] == "owner" else "◆ " + esc(m.get("kind") or "annotation")}</td>'
        f'<td>{esc(m["text"])}</td><td class="{"good" if m["hit"] else "bad"}">{"caught" if m["hit"] else "missed"}</td>'
        f'<td>{"" if not m.get("nearest") else esc(m["nearest"]["kind"]) + " " + ("+" if m["nearest"]["delta_s"] >= 0 else "") + str(m["nearest"]["delta_s"]) + " s: " + esc(m["nearest"]["text"][:50])}</td></tr>'
        for m in mom.get("items") or [])
    line_rows = "".join(
        f'<tr><td class="num">{_fmt_t(ln["at_s"])}</td><td>{esc(ln["source"])} {esc(ln["kind"])}{"" if ln["fires"] else " (silent)"}</td>'
        f'<td>{esc(ln["text"])}</td><td class="num">{_ms(ln.get("latency_ms"))}</td><td class="num">{_ms(ln.get("first_words_ms"))}</td>'
        f'<td class="num">{_ms(ln.get("from_sent_ms"))}</td></tr>'
        for ln in sc.get("lines") or [])
    unmatched = "".join(f"<li>{_fmt_t(u['at_s'])} {esc(u['kind'])}: {esc(u['text'])}</li>" for u in mom.get("unmatched_lines") or [])
    viol_rows = "".join(f'<li><b>{esc(v["kind"])}</b> @ {_fmt_t(v["at_s"])} ({esc(v["line_kind"])}): “{esc(v["text"])}” — <span class="muted">{esc(v["evidence"])}</span></li>' for v in viol) or "<li class=good>none detected</li>"
    err_rows = "".join(f"<li>{_fmt_t(e['at_s'])}: {esc(e['reason'])} on “{esc((e.get('utterance_text') or '')[:60])}”</li>" for e in sc.get("errors") or []) or "<li class=good>none</li>"

    idn = b.get("identity") or {}
    votes = idn.get("votes") or {}
    ceiling = (b.get("phone_ceiling") or {})
    ceil_att = ceiling.get("attribution") or {}
    phone_att = ((b.get("phone") or {}).get("attribution")) or {}
    enroll = settings.get("enroll")
    id_turn_rows = "".join(
        f'<tr><td class="num">{_fmt_t(r["start"])}</td><td>{esc(r["speaker"])}</td><td>{"owner" if r["pred"] else "other" if r["pred"] is False else "?"}</td>'
        f'<td>{"owner" if r["truth"] else "other" if r["truth"] is False else "?"}</td><td class="{"good" if r["ok"] else "bad" if r["ok"] is False else "muted"}">{"✓" if r["ok"] else "✗" if r["ok"] is False else "–"}</td>'
        f'<td class="num">{_score_txt(r.get("score"))}</td></tr>'
        for r in ph.get("rows") or [])
    direction = ident.get("coaching_direction") or {}

    ann_rows = ""
    for a in b.get("annotations") or []:
        q = a.get("quality") or {}
        probs = "".join(f"<li>{esc(p)}</li>" for p in (a.get("problems") or [])[:30])
        reps = "".join(f"<li>{esc(p)}</li>" for p in (a.get("repairs") or [])[:30])
        ann_rows += (f'<div class="card"><b>{esc(a["label"])}</b> <span class="muted">{esc(a.get("model"))}</span>'
                     f'{" · <b>primary (truth)</b>" if a["label"] == b.get("primary_annotation") else ""}<br>'
                     f'parsed: {"yes" if a.get("ok") else "<span class=bad>no</span>"} · words matched to Deepgram {q.get("words_matched_pct", "–")}% · '
                     f'segments re-timed {q.get("segments_aligned", "–")}/{q.get("segments", "–")} · median shift {q.get("median_shift_s", "–")} s (max {q.get("max_abs_shift_s", "–")} s)'
                     f'{"<details><summary>problems (" + str(len(a.get("problems") or [])) + ")</summary><ul class=plain>" + probs + "</ul></details>" if probs else ""}'
                     f'{"<details><summary>repairs (" + str(len(a.get("repairs") or [])) + ")</summary><ul class=plain>" + reps + "</ul></details>" if reps else ""}</div>')
    if not ann_rows:
        ann_rows = "<p class=muted>No annotation file — identity truth and heat come from Deepgram alone.</p>"

    notes = b.get("notes") or {}
    llm = lat.get("llm_cache") or {}
    problems = "".join(f"<li>{esc(p)}</li>" for p in b.get("problems") or []) or "<li class=good>none</li>"
    stage = lat.get("server_stage_summary") or {}
    stage_rows = "".join(f"<tr><td>{esc(k)}</td><td class=num>{v.get('p50')}</td><td class=num>{v.get('p95')}</td><td class=num>{v.get('n')}</td></tr>" for k, v in stage.items() if isinstance(v, dict))

    banner = "".join(f'<div class="banner">⚠ {esc(f)}</div>' for f in flags)
    warn_list = "".join(f'<div class="banner">{esc(w)}</div>' for w in idn.get("warnings") or [])
    title = f"Coach replay: {b.get('name')}"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head><body><main>
<h1>{esc(title)}</h1>
<div class="sub">{_fmt_t((b.get("audio") or {}).get("duration_s"))} of audio · mode {esc(settings.get("mode"))} · server {esc(srv.get("target") or "not run")} · generated {esc((b.get("generated_at") or "")[:19].replace("T", " "))} UTC{" · OFFLINE re-run" if b.get("offline") else ""}</div>
{banner}{warn_list}
<section class="tiles" style="margin-top:12px">{tiles}</section>

<section class="card"><h2>Timeline at a glance</h2>
<p class="legend">Speech bars by voice (owner on top) · heat = the annotation's vocal intensity · coach marks at their real arrival time
(<span style="color:var(--accent)">●</span> nudge about your own turn, <span style="color:#059669">●</span> what to say, <span style="color:var(--warn)">●</span> phone buzz; faded = silent) ·
★ your moments, ◆ annotation moments (green caught, red missed) · bottom lane = “that's you” verdicts (green right, red wrong).</p>
{overview_svg(b, sc, segs, colors)}</section>

<section class="card"><h2>Moments</h2>
<p class="legend">Caught = something the wearer would notice (a spoken line or a buzz) landed from {mom.get("pre_s", 2):g} s before to {mom.get("window_s", 10):g} s after the moment.</p>
<table><tr><th class="num">time</th><th>from</th><th>what should happen</th><th>result</th><th>nearest coach line</th></tr>{mom_rows or "<tr><td colspan=5 class=muted>No moments: add mm:ss lines to the notes, or an annotation with coach_moments.</td></tr>"}</table>
{"<p class=legend><b>Lines with no moment nearby</b> (not necessarily wrong — your notes may not list every moment):</p><ul class=plain>" + unmatched + "</ul>" if unmatched else ""}
</section>

<section class="card"><h2>Everything, in order</h2>
<p class="legend">Who said what (ground truth: {esc(truth_source)}), what the phone decided about each turn, and every coach line at the moment it would have reached you.</p>
<ul class="tl">{timeline}</ul></section>

<section class="card"><h2>Coach lines and latency</h2>
<p class="legend">“After turn” = from the moment that person stopped talking to the line arriving (real time, ±0.1 s). “From send” excludes the phone's own turn-closing time.
Server p50 {_ms(lat.get("p50_ms"))} / p90 {_ms(lat.get("p90_ms"))} over {lat.get("n", 0)} lines; first words p50 {_ms(lat.get("first_words_p50_ms"))}; phone buzz p50 {_ms(lat.get("phone_haptic_p50_ms"))}.</p>
<table><tr><th class="num">at</th><th>kind</th><th>line</th><th class="num">after turn</th><th class="num">first words</th><th class="num">from send</th></tr>{line_rows or "<tr><td colspan=6 class=muted>No coach lines.</td></tr>"}</table>
<h2 style="margin-top:12px">Possible invented facts / words in someone's mouth</h2><ul class="plain">{viol_rows}</ul>
<h2>Coach errors</h2><ul class="plain">{err_rows}</ul>
{"<details><summary>server per-stage latency (ms)</summary><table><tr><th>stage</th><th class=num>p50</th><th class=num>p95</th><th class=num>n</th></tr>" + stage_rows + "</table></details>" if stage_rows else ""}
</section>

<section class="card"><h2>Who is the owner? (identity over time)</h2>
<p>The owner's voice in this recording: <b>{esc(idn.get("wearer_label"))}</b> / annotation <b>{esc(idn.get("wearer_ann_id"))}</b>, decided by <b>{esc(idn.get("method"))}</b>.
Votes: notes description {esc(json.dumps(votes.get("description")))} · voiceprint {esc(json.dumps(votes.get("voiceprint")))} → {esc(votes.get("voiceprint_label"))}
{"· <span class=good>votes agree</span>" if idn.get("agree") else "· <span class=bad>votes DISAGREE</span>" if idn.get("agree") is False else ""}</p>
<p>Phone (enrolled with <b>{esc(enroll)}</b>): right on {ph.get("correct", 0)}/{ph.get("decided", 0)} turns it decided ({_pct(ph.get("coverage"))} of {ph.get("turns", 0)} turns decided);
recognised the owner on {_pct(ph.get("wearer_recall"))} of his turns; called someone else the owner {ph.get("false_self", 0)}×; first correct “that's you” at {_fmt_t(ph.get("first_confirmed_s"))}.
Server voiceprint: {sv.get("correct", 0)}/{sv.get("decided", 0)} right, first confirmation {_fmt_t(sv.get("first_confirmed_s"))}.
Coaching direction (nudge for the owner's own turns, “say this” for others'): {direction.get("correct", 0)}/{direction.get("total", 0)}.</p>
<p class="legend">Ceiling — the same loop enrolled from this very recording: self {ceil_att.get("selfCorrect", "–")}/{ceil_att.get("selfTotal", "–")}, attribution {ceil_att.get("correct", "–")}/{ceil_att.get("total", "–")} (vs {phone_att.get("selfCorrect", "–")}/{phone_att.get("selfTotal", "–")}, {phone_att.get("correct", "–")}/{phone_att.get("total", "–")} with {esc(enroll)}).</p>
<details><summary>per-turn verdicts</summary><table><tr><th class="num">turn</th><th>phone label</th><th>phone said</th><th>truth</th><th></th><th class="num">match</th></tr>{id_turn_rows}</table></details>
</section>

<section class="card"><h2>Inputs</h2>
<p><b>Notes</b>: who “{esc(notes.get("who"))}” · setting “{esc(notes.get("setting"))}” · phone “{esc(notes.get("phone"))}” · relationship {esc(notes.get("relationship"))} · {len(notes.get("moments") or [])} moments</p>
{"<details><summary>notes problems</summary><ul class=plain>" + "".join(f"<li>{esc(p)}</li>" for p in notes.get("problems") or []) + "</ul></details>" if notes.get("problems") else ""}
<p><b>Deepgram</b> ({esc((b.get("stt") or {}).get("source"))}): {len((b.get("stt") or {}).get("words") or [])} words, {len((b.get("stt") or {}).get("turns") or [])} turns.
<b>Phone script</b>: {esc((b.get("phone_meta") or {}).get("segmentation"))} segmentation, tone stand-in = {esc(settings.get("phone_tone"))}.
<b>LLM</b>: {esc(llm.get("model"))} · cache hits {llm.get("hits", "–")}, misses {llm.get("misses", "–")}{", offline misses " + str(llm.get("offline_misses")) if llm.get("offline_misses") else ""}.
<b>Session context sent</b>: “{esc(settings.get("session_context"))}”.</p>
{ann_rows}
<details><summary>server notes</summary><ul class="plain">{"".join(f"<li>{esc(n)}</li>" for n in srv.get("notes") or [])}</ul></details>
<details><summary>run problems</summary><ul class="plain">{problems}</ul></details>
</section>

<section class="card"><h2>What is and isn't measured</h2><ul class="plain">
<li><b>Real:</b> the phone's own TypeScript loop (Silero VAD, segmenter, ECAPA speaker-ID, nudge policy) on this audio; the server's real coaching code and LLM (or its recorded replies, with the recorded timing), its real voiceprint matcher, streamed in real time over a real WebSocket.</li>
<li><b>Stand-ins:</b> the phone's speech-to-text is Deepgram's words (Android's recognizer can't run on a laptop); the phone's on-device tone read is the annotation's vocal emotion ({esc(settings.get("phone_tone"))}); the phone's own spoken suggestion (Gemini Nano) is not replayed — only the server's lines are shown.</li>
<li><b>Not measured:</b> what the earpiece audio sounds like; network on a phone; the server-side Deepgram path (this session is local-first like the app's); library/room answers need a real account (local server has no library).</li>
<li><b>Heuristics:</b> “invented fact” = a name/number nobody said; “words in mouth” = a line telling someone else what to say. They flag candidates for a human look, not verdicts.</li>
</ul></section>
</main></body></html>"""


def write_report(bundle: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(bundle))
    return path
