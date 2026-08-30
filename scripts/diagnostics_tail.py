#!/usr/bin/env python3
"""diagnostics_tail.py — print the phone app's latest "Send diagnostics"
records for an account (uid or email) or a diagnostics id, straight from
the server's telemetry channel.

The app (Settings → "Send diagnostics", and automatically when a live
session had errors) POSTs one ``client_diagnostics`` event to
``/telemetry`` with a structured ``data`` payload: the capability probe,
the last session's latency summary / provider outcomes / STT restarts /
WebSocket reconnects / call outcome, app + OTA + device facts. The screen
shows the diagnostics id (``dx-XXXX-XXXX``) so the owner can read it out.

Usage
-----
  python scripts/diagnostics_tail.py --email sagearbor@gmail.com
  python scripts/diagnostics_tail.py --uid <firebase uid> --limit 3
  python scripts/diagnostics_tail.py --id dx-7K3M-P9QA
  python scripts/diagnostics_tail.py --email ... --raw      # full JSON
  python scripts/diagnostics_tail.py --id dx-... \\
      --score-rubric tmp/private_fixtures/maggiano3/rubric.json --owner dad
      # + frame accuracy of the record's on-phone voice separation
      #   (``device_diarization``, the replay screen's "Separate voices on
      #   this phone" row) against a per-second rubric, scored exactly as
      #   the 2026-08-29 bake-off (docs/research/.../score.py)

``--base-url`` (or ``MINDSHIFT_API_URL``) defaults to the production
Cloud Run URL. ``GET /telemetry`` is unauthenticated (an inherited owner
decision — server/watch/routers/telemetry.py); ``--device`` narrows the
server-side query to one ``phone:<platform>:<uid>`` device id when you
know it, otherwise the newest 1000 events are scanned client-side.

Exit status: 0 when at least one record printed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = os.environ.get(
    "MINDSHIFT_API_URL", "https://mindshift-api-e5zja2g5bq-uc.a.run.app"
)
TAG = "client_diagnostics"


def fetch_events(base_url: str, device: str | None, limit: int) -> list[dict]:
    params = {"limit": str(limit)}
    if device:
        params["device"] = device
    url = f"{base_url.rstrip('/')}/telemetry?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (owner-supplied URL)
        return json.load(resp)


def matches(event: dict, *, uid: str | None, email: str | None, diag_id: str | None) -> bool:
    if event.get("tag") != TAG:
        return False
    data = event.get("data") or {}
    if diag_id:
        return str(data.get("diagnostics_id", "")).upper() == diag_id.upper()
    if uid and data.get("uid") != uid and not str(event.get("device", "")).endswith(f":{uid}"):
        return False
    if email and str(data.get("email") or "").lower() != email.lower():
        return False
    return True


def _ms(v) -> str:
    return "-" if v is None else f"{v} ms"


def summarize(event: dict) -> str:
    data = event.get("data") or {}
    app = data.get("app") or {}
    dev = data.get("device") or {}
    lines = [
        f"== {data.get('diagnostics_id', '?')}  {event.get('received_at', '?')}  "
        f"trigger={data.get('trigger', '?')}  uid={data.get('uid')}  email={data.get('email')}",
        f"   app {app.get('version')} build {app.get('build')} runtime {app.get('runtimeVersion')} "
        f"ota {app.get('updateId') or '-'} ({app.get('channel') or '-'})  "
        f"{dev.get('platform')} {dev.get('osVersion') or ''} {dev.get('model') or ''}"
        + (f"  ua={dev.get('userAgent')}" if dev.get("userAgent") else ""),
    ]
    cap = data.get("capability")
    if cap:
        sid = cap.get("speakerId") or {}
        lines.append(
            f"   capability: vad={cap.get('vad')} speakerId="
            f"{'on' if sid.get('active') else 'off'} ({sid.get('reason') or sid.get('enrolled')}) "
            f"llm={' -> '.join(cap.get('llm') or [])}"
        )
    elif data.get("capability_reason"):
        lines.append(f"   capability: {data['capability_reason']}")
    s = data.get("last_session")
    if s:
        lat = s.get("latency") or {}
        lines.append(
            f"   session {s.get('sessionId')} mode={s.get('mode')} turns={s.get('turns')} "
            f"onDevice={s.get('onDevice')} post={s.get('postStatus')}"
        )
        lines.append(
            f"   latency: median to-speak {_ms(lat.get('medianToSpeakMs'))} p90 {_ms(lat.get('p90ToSpeakMs'))} "
            f"llm {_ms(lat.get('medianLlmMs'))} stt-wait {_ms(lat.get('medianSttWaitMs'))} "
            f"spoken {lat.get('spoken')}/{lat.get('turns')} held {lat.get('held')} "
            f"providers={json.dumps(lat.get('byProvider') or {})}"
        )
        if lat.get("byOutcome"):
            # WHY each local rung did/didn't answer, e.g. {"os:refused": 12}.
            lines.append(f"   attempt outcomes={json.dumps(lat.get('byOutcome'))}")
        for key, sample in (lat.get("outcomeSamples") or {}).items():
            # The actual error/refusal text (e.g. the Gemini Nano exception).
            lines.append(f"     {key}: {sample}")
        lines.append(
            f"   stt restarts={s.get('sttRestarts')} failure={s.get('sttFailure') or '-'} | "
            f"ws reconnects={s.get('wsReconnects')} | mic={s.get('micError') or '-'}"
        )
        if s.get("liveStatus"):
            lines.append(f"   live: {s['liveStatus']}")
        if s.get("call"):
            c = s["call"]
            lines.append(
                f"   call: {c.get('status')} iceRestarts={c.get('iceRestarts')} "
                f"connected={c.get('connectedSeconds')} s error={c.get('error') or '-'}"
            )
        errs = s.get("errors") or []
        lines.append("   errors: " + (" | ".join(errs) if errs else "none"))
    else:
        lines.append("   (no session in this record)")
    dd = data.get("device_diarization")
    if dd:
        lines.extend(summarize_device_diarization(dd))
    return "\n".join(lines)


def _label(index: int) -> str:
    """The app's cluster naming (speakerId.unknownLabel): 0 → Speaker A."""
    return f"Speaker {chr(65 + index % 26)}"


def summarize_device_diarization(dd: dict, max_segments: int = 40) -> list[str]:
    """The on-phone voice-separation run (apps/mobile/src/live/
    deviceDiarization.ts) carried in ``data.device_diarization``."""
    dev = dd.get("device") or {}
    segs = dd.get("segments") or []
    lines = [
        f"   device_diarization: recording {dd.get('recording_id')} engine {dd.get('engine')} "
        f"k={dd.get('k')} (eigengap {dd.get('k_eigengap')}) runs={len(segs)} "
        f"windows={dd.get('windows')}/{dd.get('windows_total')} @ {dd.get('hop_s')} s hop "
        f"gate={dd.get('gate_rms')} speech={dd.get('speech_s')} s of {dd.get('duration_s')} s",
        f"     timings: download {_ms(dd.get('download_ms'))} ({dd.get('download_bytes')} B) · "
        f"embed {dd.get('embed_ms_mean')} ms/window (p90 {dd.get('embed_ms_p90')}) · "
        f"cluster {_ms(dd.get('cluster_ms'))} · total {_ms(dd.get('total_ms'))}",
        f"     model {dd.get('model_rev')} ({dd.get('model_source')})  device {dev.get('platform')} "
        f"{dev.get('osVersion') or ''} {dev.get('model') or ''}  at {dd.get('created_at')}",
        f"     eigenvalues {dd.get('eigenvalues')}",
    ]
    shown = segs[:max_segments]
    lines.append("     segments: " + " | ".join(f"{s:.2f}-{e:.2f} {_label(int(lab))}" for s, e, lab in shown)
                 + (f" … (+{len(segs) - len(shown)} more)" if len(segs) > len(shown) else ""))
    return lines


def _bakeoff_score_module():
    """``score`` from docs/research/2026-08-29-voice-separation (the shared
    bake-off scorer), imported by path so this script needs no package."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "docs", "research", "2026-08-29-voice-separation", "score.py")
    spec = importlib.util.spec_from_file_location("bakeoff_score", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rubric(path: str) -> list:
    """A per-second listen-through rubric ``{"segments": [{"start", "end",
    "speaker"}]}`` (speaker may be a list for overlap — either is credited)
    → the scorer's ``[(start, end, label | (labels...)), ...]``."""
    with open(path, encoding="utf-8") as fh:
        rub = json.load(fh)
    segs = rub["segments"] if isinstance(rub, dict) else rub
    out = []
    for seg in segs:
        sp = seg["speaker"]
        out.append((float(seg["start"]), float(seg["end"]), tuple(sp) if isinstance(sp, list) else sp))
    return out


def score_against_rubric(dd: dict, rubric_path: str, owner: str | None) -> str:
    """Frame accuracy of the phone's segments against the rubric, exactly as
    the bake-off scored every approach (score.py: 10 ms frames, best
    one-to-one label mapping, rubric speech only)."""
    score = _bakeoff_score_module()
    gt = load_rubric(rubric_path)
    pred = [(float(s), float(e), _label(int(lab))) for s, e, lab in (dd.get("segments") or [])]
    res = score.score_segments(gt, pred, owner)
    recall = " ".join(f"{k}={v}" for k, v in (res.get("per_gt_recall") or {}).items())
    return (
        f"   rubric {os.path.basename(rubric_path)}: frame accuracy {res['frame_accuracy']} "
        f"k {res['k_pred']}/{res['k_true']} owner purity {res['owner_purity']} "
        f"unlabelled {res['unlabelled_frac']}\n"
        f"     mapping {json.dumps(res['mapping'])}\n"
        f"     recall {recall}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--uid")
    ap.add_argument("--email")
    ap.add_argument("--id", dest="diag_id", help="a diagnostics id read off the screen (dx-XXXX-XXXX)")
    ap.add_argument("--device", help="exact device id (phone:<platform>:<uid>) to filter server-side")
    ap.add_argument("--limit", type=int, default=1, help="how many records to print (newest first)")
    ap.add_argument("--scan", type=int, default=1000, help="how many telemetry events to fetch (max 1000)")
    ap.add_argument("--raw", action="store_true", help="print the full JSON payloads")
    ap.add_argument(
        "--score-rubric", metavar="RUBRIC_JSON",
        help="score each record's device_diarization segments against a per-second rubric "
             "({segments: [{start, end, speaker}]}) with the bake-off scorer",
    )
    ap.add_argument("--owner", help="the rubric label that is the phone's owner (for owner purity)")
    args = ap.parse_args(argv)
    if not (args.uid or args.email or args.diag_id or args.device):
        ap.error("give --uid, --email, --id or --device")
    try:
        events = fetch_events(args.base_url, args.device, min(max(args.scan, 1), 1000))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as err:
        print(f"couldn't read {args.base_url}/telemetry: {err}", file=sys.stderr)
        return 1
    hits = [e for e in events if matches(e, uid=args.uid, email=args.email, diag_id=args.diag_id)]
    hits.sort(key=lambda e: (e.get("received_at", ""), e.get("ts", "")), reverse=True)
    if not hits:
        print("no client_diagnostics records match", file=sys.stderr)
        return 1
    for e in hits[: max(args.limit, 1)]:
        if args.raw:
            print(json.dumps(e, indent=2, sort_keys=True))
        else:
            print(summarize(e))
        if args.score_rubric:
            dd = (e.get("data") or {}).get("device_diarization")
            if dd:
                print(score_against_rubric(dd, args.score_rubric, args.owner))
            else:
                print("   (no device_diarization in this record — nothing to score)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
