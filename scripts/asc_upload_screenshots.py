#!/usr/bin/env /usr/bin/python3
"""Upload App Store screenshots to an appStoreVersionLocalization.

Full reserve -> upload -> commit -> verify cycle for the App Store Connect
appScreenshots API, which is fiddly enough that doing it by hand is a good way
to leave half-uploaded assets in a FAILED delivery state.

Auth: ES256-signed JWT from the ASC API key. Reads EXPO_ASC_KEY_ID,
EXPO_ASC_ISSUER_ID and EXPO_ASC_API_KEY_PATH from the environment
(`source ~/.config/asc/asc.env`). Needs the SYSTEM python (/usr/bin/python3),
which is where PyJWT and requests live on this machine.

The display type is NOT guessed. `--display-type` must be one of the values the
API itself accepts; `--list-display-types` asks the API for that enum by
sending a deliberately invalid value and reading back the error, so this script
never encodes a stale assumption about which size Apple currently requires.
As of 2026-09 Apple requires the 6.9" iPhone size, which the API models as
APP_IPHONE_67 and which accepts 1320x2868 (iPhone 17 Pro Max, 16 Pro Max).

Usage:
  source ~/.config/asc/asc.env
  /usr/bin/python3 scripts/asc_upload_screenshots.py --list-display-types
  /usr/bin/python3 scripts/asc_upload_screenshots.py \
      --app-id 6814688797 --display-type APP_IPHONE_67 \
      shot1.png shot2.png shot3.png

Uploading assets is deliberately all this does. It never creates a
submission -- shipping the version stays a human decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import time

import jwt
import requests

BASE = "https://api.appstoreconnect.apple.com"
TIMEOUT = 120


def make_headers() -> dict:
    """A fresh short-lived JWT per call, so long uploads can't outlive a token."""
    kid = os.environ["EXPO_ASC_KEY_ID"]
    iss = os.environ["EXPO_ASC_ISSUER_ID"]
    key = pathlib.Path(os.path.expandvars(os.environ["EXPO_ASC_API_KEY_PATH"])).read_text()
    token = jwt.encode(
        {"iss": iss, "iat": int(time.time()), "exp": int(time.time()) + 900,
         "aud": "appstoreconnect-v1"},
        key, algorithm="ES256", headers={"kid": kid, "typ": "JWT"},
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def png_size(path: pathlib.Path) -> tuple[int, int]:
    """Width/height straight out of the IHDR, so we can reject a wrong-size
    file locally instead of after a round trip."""
    head = path.open("rb").read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def list_display_types() -> list[str]:
    """Ask the API which screenshotDisplayType values it accepts.

    There is no GET endpoint for the enum, so we POST an invalid one against a
    nonexistent parent and parse the enumeration out of the validation error.
    """
    body = {"data": {"type": "appScreenshotSets",
                     "attributes": {"screenshotDisplayType": "ZZ_INVALID_PROBE"},
                     "relationships": {"appStoreVersionLocalization": {
                         "data": {"type": "appStoreVersionLocalizations", "id": "0"}}}}}
    r = requests.post(f"{BASE}/v1/appScreenshotSets", headers=make_headers(),
                      json=body, timeout=TIMEOUT)
    for err in r.json().get("errors", []):
        detail = err.get("detail", "")
        if "Expected one of" in detail:
            raw = detail.split("Expected one of", 1)[1]
            # The enum arrives as a quoted, comma-joined list; pull the quoted
            # tokens out directly rather than trying to strip punctuation off
            # the split pieces (the first and last carry extra characters).
            return sorted(set(re.findall(r"'([A-Z0-9_]+)'", raw)))
    die(f"could not read the display-type enum from the API: {r.status_code} {r.text[:400]}")
    return []


def find_localization(app_id: str, locale_prefix: str) -> tuple[str, str]:
    """Resolve the editable version's localization id for the given locale."""
    H = make_headers()
    versions = requests.get(f"{BASE}/v1/apps/{app_id}/appStoreVersions?limit=10",
                            headers=H, timeout=TIMEOUT).json().get("data", [])
    # Only a version still in an editable state can take new assets.
    editable = {"PREPARE_FOR_SUBMISSION", "DEVELOPER_REJECTED", "REJECTED",
                "METADATA_REJECTED", "INVALID_BINARY", "WAITING_FOR_REVIEW"}
    for v in versions:
        attrs = v["attributes"]
        if attrs.get("appStoreState") not in editable:
            continue
        locs = requests.get(
            f"{BASE}/v1/appStoreVersions/{v['id']}/appStoreVersionLocalizations",
            headers=H, timeout=TIMEOUT).json().get("data", [])
        for loc in locs:
            if loc["attributes"].get("locale", "").startswith(locale_prefix):
                print(f"version {attrs.get('versionString')} "
                      f"({attrs.get('appStoreState')}) "
                      f"locale {loc['attributes']['locale']}")
                return v["id"], loc["id"]
    die(f"no editable appStoreVersion with a {locale_prefix}* localization on app {app_id}")
    return "", ""


def ensure_set(loc_id: str, display_type: str) -> str:
    """Reuse the existing set for this display type, else create one.

    Creating a second set for a display type that already has one is an error,
    and re-running this script should be safe.
    """
    H = make_headers()
    sets = requests.get(
        f"{BASE}/v1/appStoreVersionLocalizations/{loc_id}/appScreenshotSets",
        headers=H, timeout=TIMEOUT).json().get("data", [])
    for s in sets:
        if s["attributes"].get("screenshotDisplayType") == display_type:
            print(f"reusing existing {display_type} set {s['id']}")
            return s["id"]
    body = {"data": {"type": "appScreenshotSets",
                     "attributes": {"screenshotDisplayType": display_type},
                     "relationships": {"appStoreVersionLocalization": {
                         "data": {"type": "appStoreVersionLocalizations", "id": loc_id}}}}}
    r = requests.post(f"{BASE}/v1/appScreenshotSets", headers=H, json=body, timeout=TIMEOUT)
    if r.status_code not in (200, 201):
        die(f"creating {display_type} set failed: {r.status_code} {r.text[:600]}")
    set_id = r.json()["data"]["id"]
    print(f"created {display_type} set {set_id}")
    return set_id


def upload_one(set_id: str, path: pathlib.Path) -> str:
    """Reserve, PUT the bytes to every upload operation, then commit."""
    data = path.read_bytes()
    body = {"data": {"type": "appScreenshots",
                     "attributes": {"fileSize": len(data), "fileName": path.name},
                     "relationships": {"appScreenshotSet": {
                         "data": {"type": "appScreenshotSets", "id": set_id}}}}}
    r = requests.post(f"{BASE}/v1/appScreenshots", headers=make_headers(),
                      json=body, timeout=TIMEOUT)
    if r.status_code not in (200, 201):
        die(f"reserving {path.name} failed: {r.status_code} {r.text[:600]}")
    reserved = r.json()["data"]
    shot_id = reserved["id"]
    ops = reserved["attributes"].get("uploadOperations") or []
    if not ops:
        die(f"{path.name}: reservation returned no uploadOperations")

    for i, op in enumerate(ops, 1):
        chunk = data[op["offset"]:op["offset"] + op["length"]]
        headers = {h["name"]: h["value"] for h in (op.get("requestHeaders") or [])}
        # Raw bytes to Apple's blob store -- deliberately NOT make_headers(),
        # which would attach the ASC bearer token to a third-party URL.
        resp = requests.request(op["method"], op["url"], headers=headers,
                                data=chunk, timeout=TIMEOUT)
        if resp.status_code not in (200, 201, 204):
            die(f"{path.name}: upload op {i}/{len(ops)} failed: "
                f"{resp.status_code} {resp.text[:300]}")
        print(f"  uploaded chunk {i}/{len(ops)} ({len(chunk)} bytes)")

    checksum = hashlib.md5(data).hexdigest()
    patch = {"data": {"type": "appScreenshots", "id": shot_id,
                      "attributes": {"uploaded": True, "sourceFileChecksum": checksum}}}
    r = requests.patch(f"{BASE}/v1/appScreenshots/{shot_id}", headers=make_headers(),
                       json=patch, timeout=TIMEOUT)
    if r.status_code != 200:
        die(f"{path.name}: commit failed: {r.status_code} {r.text[:600]}")
    print(f"  committed {path.name} as {shot_id} (md5 {checksum})")
    return shot_id


def wait_delivery(shot_ids: list[str], tries: int = 30, delay: int = 6) -> dict:
    """Poll until nothing is still processing. UPLOAD_COMPLETE/AWAITING_UPLOAD
    mean Apple hasn't finished validating yet -- only VALID is success."""
    states: dict[str, str] = {}
    for _ in range(tries):
        H = make_headers()
        pending = False
        for sid in shot_ids:
            r = requests.get(f"{BASE}/v1/appScreenshots/{sid}", headers=H, timeout=TIMEOUT)
            ads = (r.json().get("data", {}).get("attributes", {})
                   .get("assetDeliveryState") or {})
            state = ads.get("state", "UNKNOWN")
            states[sid] = state
            if ads.get("warnings") or ads.get("errors"):
                states[sid] = f"{state} {json.dumps(ads.get('errors') or ads.get('warnings'))}"
            if state in ("UPLOAD_COMPLETE", "AWAITING_UPLOAD"):
                pending = True
        if not pending:
            return states
        time.sleep(delay)
    return states


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screenshots", nargs="*", type=pathlib.Path)
    ap.add_argument("--app-id", default="6814688797")
    ap.add_argument("--display-type", help="e.g. APP_IPHONE_67 (see --list-display-types)")
    ap.add_argument("--locale", default="en", help="locale prefix to match (default en)")
    ap.add_argument("--expect-size", metavar="WxH",
                    help="reject any PNG that is not exactly this size, e.g. 1320x2868")
    ap.add_argument("--list-display-types", action="store_true",
                    help="print the screenshotDisplayType enum the API accepts and exit")
    args = ap.parse_args()

    if args.list_display_types:
        for v in list_display_types():
            print(v)
        return 0

    if not args.screenshots:
        die("no screenshots given")
    if not args.display_type:
        die("--display-type is required (see --list-display-types)")

    valid = list_display_types()
    if args.display_type not in valid:
        die(f"{args.display_type} is not accepted by the API. Valid: {', '.join(valid)}")

    want = None
    if args.expect_size:
        w, h = args.expect_size.lower().split("x")
        want = (int(w), int(h))
    for p in args.screenshots:
        if not p.is_file():
            die(f"{p} does not exist")
        size = png_size(p)
        if want and size != want:
            die(f"{p} is {size[0]}x{size[1]}, expected {want[0]}x{want[1]}")
        print(f"{p.name}: {size[0]}x{size[1]}, {p.stat().st_size} bytes")

    _, loc_id = find_localization(args.app_id, args.locale)
    set_id = ensure_set(loc_id, args.display_type)

    shot_ids = []
    for p in args.screenshots:
        print(f"uploading {p.name} ...")
        shot_ids.append(upload_one(set_id, p))

    print("\nwaiting for assetDeliveryState ...")
    states = wait_delivery(shot_ids)
    failed = False
    for p, sid in zip(args.screenshots, shot_ids):
        state = states.get(sid, "UNKNOWN")
        print(f"  {p.name}  {sid}  {state}")
        if not state.startswith("COMPLETE") and state != "VALID":
            failed = True
    if failed:
        print("\nat least one screenshot is not VALID", file=sys.stderr)
        return 1
    print("\nall screenshots VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
