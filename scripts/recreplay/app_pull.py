""""Use my latest app recording": copy the newest recording the owner made in
the MindShift app (the server stores them in GCS under
``recordings/{uid}/{recording_id}/`` — server/recordings_store.py) into the
drop folder, with a notes stub to fill in.

Needs Google Application Default Credentials that can read the bucket
(``gcloud auth application-default login``) and, to go from an email to a
uid, Firebase Admin on the same credentials (server/auth.resolve_uid_by_email).
Read-only: nothing in the bucket is changed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

DEFAULT_BUCKET = "arborfam-hub-mindshift-recordings"   # scripts/deploy_cloudrun.sh's default
AUDIO_NAMES = ("audio.m4a", "audio.wav", "audio.webm", "audio.mp3", "audio.ogg", "audio.aac")


class AppPullError(RuntimeError):
    pass


def bucket_name() -> str:
    from .stt import env_value
    return (os.getenv("MINDSHIFT_RECORDINGS_BUCKET") or env_value("MINDSHIFT_RECORDINGS_BUCKET") or DEFAULT_BUCKET).strip()


def resolve_uid(email: str) -> str:
    try:
        import auth
        uid = auth.resolve_uid_by_email(email)
    except Exception as exc:  # noqa: BLE001 — credentials/SDK problems are reported, not guessed around
        raise AppPullError(f"could not look up {email} in Firebase ({type(exc).__name__}: {exc}); "
                           "pass --uid instead, or run `gcloud auth application-default login`") from exc
    if not uid:
        raise AppPullError(f"no MindShift account has the email {email}")
    return uid


def list_recordings(bucket, uid: str) -> list[dict]:
    """[{id, meta, files:{name: blob}}] newest first (meta.created_at)."""
    prefix = f"recordings/{uid}/"
    by_id: dict[str, dict] = {}
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        rid, _, fname = rel.partition("/")
        if rid and fname:
            by_id.setdefault(rid, {})[fname] = blob
    out = []
    for rid, files in by_id.items():
        if "meta.json" not in files:
            continue
        meta = json.loads(files["meta.json"].download_as_bytes())
        out.append({"id": rid, "meta": meta, "files": files})
    out.sort(key=lambda r: r["meta"].get("created_at", ""), reverse=True)
    return out


def pick_latest_with_audio(recs: list[dict], recording_id: str | None = None) -> dict:
    for r in recs:
        if recording_id and r["id"] != recording_id:
            continue
        if any(n in r["files"] for n in AUDIO_NAMES):
            return r
    raise AppPullError("no app recording with stored audio found"
                       + (f" for id {recording_id}" if recording_id else "")
                       + " (live sessions keep audio only when 'keep audio' is on)")


def slug_for(rec: dict) -> str:
    created = rec["meta"].get("created_at") or ""
    try:
        stamp = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y%m%d-%H%M")
    except ValueError:
        stamp = "unknown"
    return f"app-{stamp}-{rec['id'][:8]}"


def notes_stub(rec: dict) -> str:
    meta = rec["meta"]
    title = meta.get("title") or meta.get("original_filename") or "app recording"
    return (
        "who: I'm the owner — FILL IN which voice is you (e.g. the man with the low voice; other voice is my son)\n"
        f"setting: {title}\n"
        "phone: FILL IN (e.g. in my pocket, earbuds in)\n"
        "moments (optional, one per line, mm:ss — what a good coach would have said):\n"
    )


def pull_latest(inbox: Path, *, email: str | None = None, uid: str | None = None,
                recording_id: str | None = None, client=None) -> Path:
    if not uid:
        if not email:
            raise AppPullError("need --email or --uid")
        uid = resolve_uid(email)
    if client is None:
        try:
            from google.cloud import storage
            client = storage.Client()
        except Exception as exc:  # noqa: BLE001
            raise AppPullError(f"Google Cloud Storage unavailable ({type(exc).__name__}: {exc}); "
                               "run `gcloud auth application-default login`") from exc
    bucket = client.bucket(bucket_name())
    rec = pick_latest_with_audio(list_recordings(bucket, uid), recording_id)
    name = slug_for(rec)
    dest = Path(inbox) / name
    dest.mkdir(parents=True, exist_ok=True)
    audio_name = next(n for n in AUDIO_NAMES if n in rec["files"])
    rec["files"][audio_name].download_to_filename(str(dest / f"{name}{Path(audio_name).suffix}"))
    (dest / f"{name}.app_meta.json").write_text(json.dumps({**rec["meta"], "recording_id": rec["id"], "uid": uid}, indent=1))
    if "turns.json" in rec["files"]:
        (dest / f"{name}.app_turns.json").write_bytes(rec["files"]["turns.json"].download_as_bytes())
    notes = dest / f"{name}.notes.txt"
    if not notes.exists():
        notes.write_text(notes_stub(rec))
    return dest
