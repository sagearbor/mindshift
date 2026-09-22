#!/usr/bin/env python3
"""Create every Apple signing artefact an EAS store build needs, without a single
interactive prompt — reusable across every repo on this machine.

WHY THIS EXISTS
  `eas build -p ios --non-interactive` refuses to mint a distribution certificate
  ("Distribution Certificate is not validated for non-interactive builds"), and
  `eas credentials` has no non-interactive mode. So we mint the certificate and
  the App Store provisioning profile ourselves through the App Store Connect API
  and hand EAS a local credentials.json instead.

WHAT IT DOES (all idempotent — existing artefacts are reused, never duplicated)
  1. registers the bundle identifier on the Apple Developer portal
  2. generates an RSA key + CSR, POSTs it for an iOS Distribution certificate
  3. builds a .p12 from that certificate and key, with a random password
  4. creates an IOS_APP_STORE provisioning profile bound to both
  5. repeats 1 and 4 for every extra Xcode target (--watch-target /
     --extra-target): an App Extension, App Clip or watchOS companion has its
     own bundle identifier and so needs its own profile — but it shares the one
     team certificate
  6. writes <project>/credentials.json pointing at the files, in EAS's
     single-target form when there is one target and its MULTI-TARGET form
     (keyed by Xcode target name) as soon as there is more than one

WHERE THE ARTEFACTS LIVE — and why the split matters
  ~/.config/ios-credentials/_team/          dist.key, dist.cer, dist.p12,
                                            p12_password.txt, cert_id.txt
  ~/.config/ios-credentials/<bundle id>/    AppStore.mobileprovision

  ONE distribution certificate signs every app on the team, and Apple caps
  them at TWO PER TEAM. An earlier version of this script kept the certificate
  under the per-bundle-id directory, so the "reuse what we already minted"
  check missed on every new app and minted a fresh certificate — the second
  app took the last slot and the third could not sign at all. The certificate
  is therefore team-level; only the provisioning profile is per app.
  Pre-existing per-app certificates are migrated into _team/ automatically on
  the next run.

CREDENTIALS IT READS (machine-level, shared by every repo — see docs/play/README.md)
  ~/.config/asc/asc.env  ->  EXPO_ASC_KEY_ID, EXPO_ASC_ISSUER_ID,
                             EXPO_ASC_API_KEY_PATH, APPLE_TEAM_ID
  Source that file first:  source ~/.config/asc/asc.env

USAGE
  source ~/.config/asc/asc.env
  python3 scripts/ios_credentials_bootstrap.py \
      --bundle-id com.sagearbor.mindshift.app \
      --name MindShift \
      --project-dir apps/mobile

  # …with a watchOS companion target (Xcode target name MindShiftWatch):
  python3 scripts/ios_credentials_bootstrap.py \
      --bundle-id com.sagearbor.mindshift.app \
      --name MindShift \
      --project-dir apps/mobile \
      --xcode-target MindShift \
      --watch-target MindShiftWatch

MULTI-TARGET credentials.json — and why the names matter
  A project with more than one Xcode target needs credentials for EVERY target,
  because each one is identified by its own bundle identifier
  (docs.expo.dev/app-signing/local-credentials/#multi-target-project). The
  multi-target form REPLACES the single-target form — `ios.provisioningProfilePath`
  at the top level stops being read — and its keys are XCODE TARGET NAMES:

    {"ios": {"MindShift":      {"provisioningProfilePath": …, "distributionCertificate": …},
             "MindShiftWatch": {"provisioningProfilePath": …, "distributionCertificate": …}}}

  Get a name wrong and the cloud build fails at signing with
  "No profiles for 'com.x.app.watchkitapp' were found" after ~20 minutes. Read
  the names out of the generated project rather than guessing them:

    grep productName apps/mobile/ios/*.xcodeproj/project.pbxproj

  For an Expo CNG project the same names must also appear in
  extra.eas.build.experimental.ios.appExtensions — that config key, not a
  pbxproj that does not exist yet, is how eas-cli enumerates targets. Config
  plugins such as @bacons/apple-targets write it for you; check it with
  `npx expo config --type introspect`.

AFTER IT RUNS
  Add "credentialsSource": "local" to the ios block of the eas.json build
  profile, then `eas build -p ios --profile production --non-interactive` works
  unattended.  Certificates last one year; re-run this to roll them.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys
import time

try:
    import jwt
    import requests
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    # PyJWT and requests live in the SYSTEM python on this machine, not in
    # Homebrew's. Running this with /opt/homebrew/bin on PATH ahead of /usr/bin
    # fails here, and the traceback looks like a bug in the script rather than
    # the wrong interpreter — so say which it is.
    raise SystemExit(
        f"{exc.name} is not available to {sys.executable}.\n"
        "This script needs PyJWT and requests. On this machine they are in the "
        "system interpreter, so run it as /usr/bin/python3, or install them "
        "into whichever python you are using:\n"
        f"  /usr/bin/python3 {' '.join(sys.argv)}"
    ) from exc

API = "https://api.appstoreconnect.apple.com"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


class ASC:
    """Thin App Store Connect client. Tokens last 20 min, so mint one per call."""

    def __init__(self) -> None:
        try:
            self.kid = os.environ["EXPO_ASC_KEY_ID"]
            self.iss = os.environ["EXPO_ASC_ISSUER_ID"]
            key_path = os.path.expandvars(os.environ["EXPO_ASC_API_KEY_PATH"])
        except KeyError as exc:
            die(f"{exc.args[0]} is not set — run: source ~/.config/asc/asc.env")
        self.key = pathlib.Path(key_path).expanduser().read_text()

    def _headers(self) -> dict:
        now = int(time.time())
        token = jwt.encode(
            {"iss": self.iss, "iat": now, "exp": now + 900, "aud": "appstoreconnect-v1"},
            self.key,
            algorithm="ES256",
            headers={"kid": self.kid, "typ": "JWT"},
        )
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def get(self, path: str) -> dict:
        r = requests.get(f"{API}{path}", headers=self._headers(), timeout=30)
        if r.status_code != 200:
            die(f"GET {path} -> {r.status_code}: {r.text[:400]}")
        return r.json()

    def delete(self, path: str) -> None:
        r = requests.delete(f"{API}{path}", headers=self._headers(), timeout=30)
        if r.status_code not in (200, 204):
            die(f"DELETE {path} -> {r.status_code}: {r.text[:400]}")

    def post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{API}{path}", headers=self._headers(),
                          data=json.dumps(body), timeout=30)
        if r.status_code not in (200, 201):
            die(f"POST {path} -> {r.status_code}: {r.text[:600]}")
        return r.json()


def sh(*args: str) -> None:
    subprocess.run(args, check=True, capture_output=True)


def ensure_bundle_id(asc: ASC, bundle_id: str, name: str, team: str) -> str:
    # Apple's filter[identifier] is a PREFIX match, not an exact one, and the
    # order is not the order you would hope for. Querying
    # com.sagearbor.mindshift.app returns BOTH that record and
    # com.sagearbor.mindshift.app.watchkitapp, with the watch one first — so
    # taking data[0] silently resolves the parent app to its own watch target.
    # Nothing failed while the parent profile already existed and was merely
    # reused, which is exactly why this was worth catching before a reissue
    # bound a profile to the wrong identifier. Always re-check exactly.
    found = [b for b in asc.get(
        f"/v1/bundleIds?filter[identifier]={bundle_id}&limit=200")["data"]
        if b["attributes"]["identifier"] == bundle_id]
    if len(found) > 1:
        die(f"{len(found)} bundle id records claim {bundle_id} — resolve this at "
            "developer.apple.com before continuing")
    if found:
        print(f"  bundle id {bundle_id} already registered (id={found[0]['id']})")
        return found[0]["id"]
    created = asc.post("/v1/bundleIds", {"data": {"type": "bundleIds", "attributes": {
        "identifier": bundle_id, "name": name, "platform": "IOS", "seedId": team}}})
    print(f"  registered bundle id {bundle_id} (id={created['data']['id']})")
    return created["data"]["id"]


def adopt_legacy_certificate(team_out: pathlib.Path) -> None:
    """Migrate a certificate minted by the older per-bundle-id layout.

    Versions of this script before 2026-09-21 stored the certificate under
    ~/.config/ios-credentials/<bundle id>/. Copy the first complete set we find
    into _team/ so existing apps keep their certificate and new apps reuse it
    instead of burning one of Apple's two slots. Non-destructive: the old files
    are left where they are, just no longer used.
    """
    if (team_out / "cert_id.txt").exists() and (team_out / "dist.key").exists():
        return
    root = team_out.parent
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d == team_out:
            continue
        if not ((d / "cert_id.txt").exists() and (d / "dist.key").exists()):
            continue
        for f in ("cert_id.txt", "dist.key", "dist.cer", "dist.pem",
                  "dist.p12", "p12_password.txt"):
            src = d / f
            if src.exists():
                dst = team_out / f
                dst.write_bytes(src.read_bytes())
                dst.chmod(0o600)
        print(f"  migrated the certificate from {d.name} into _team/ "
              f"(it is shared by every app now)")
        return


def ensure_certificate(asc: ASC, out: pathlib.Path, email: str, name: str) -> str:
    """Reuse the team certificate; only mint when the private key we hold has
    no matching certificate on the account.

    Apple caps distribution certificates at two per team and one certificate
    signs every app, so minting per app is always wrong — see the module
    docstring.
    """
    adopt_legacy_certificate(out)
    cert_id_file = out / "cert_id.txt"
    if cert_id_file.exists() and (out / "dist.key").exists():
        cid = cert_id_file.read_text().strip()
        live = [c["id"] for c in asc.get("/v1/certificates?limit=200")["data"]]
        if cid in live:
            print(f"  reusing distribution certificate {cid}")
            return cid
        print(f"  recorded certificate {cid} is gone from the account — minting a new one")

    # Fail loudly rather than letting Apple reject the POST with a cryptic 409.
    # Match the two types that share the iOS distribution cap exactly, NOT any
    # type containing "DISTRIBUTION": MAC_APP_DISTRIBUTION and
    # MAC_INSTALLER_DISTRIBUTION have their own separate limits, and counting
    # them here would refuse to mint an iOS certificate while a slot was free.
    existing = [c for c in asc.get("/v1/certificates?limit=200")["data"]
                if c["attributes"]["certificateType"]
                in ("IOS_DISTRIBUTION", "DISTRIBUTION")]
    if len(existing) >= 2:
        die("Apple allows 2 iOS distribution certificates per team and this "
            f"account already has {len(existing)}:\n  " +
            "\n  ".join(f"{c['id']}  {c['attributes'].get('name')}  "
                        f"expires {c['attributes'].get('expirationDate')}"
                        for c in existing) +
            "\n\nOne certificate signs every app, so you almost certainly want to "
            "reuse one rather than mint a third. If you still hold its private key, "
            "put dist.key/dist.cer/cert_id.txt in "
            f"{out} and re-run. Otherwise revoke an unused one at "
            "developer.apple.com -> Certificates first.")

    key = out / "dist.key"
    if not key.exists():
        sh("openssl", "genrsa", "-out", str(key), "2048")
        key.chmod(0o600)
    csr = out / "dist.csr"
    sh("openssl", "req", "-new", "-key", str(key), "-out", str(csr),
       # CN is team-level: this one certificate signs every app.
       "-subj", f"/emailAddress={email}/CN={name}/C=US")

    created = asc.post("/v1/certificates", {"data": {"type": "certificates", "attributes": {
        "certificateType": "IOS_DISTRIBUTION", "csrContent": csr.read_text()}}})
    attrs = created["data"]["attributes"]
    (out / "dist.cer").write_bytes(base64.b64decode(attrs["certificateContent"]))
    cert_id_file.write_text(created["data"]["id"])
    print(f"  minted {attrs['certificateType']} '{attrs.get('name')}' "
          f"expiring {attrs.get('expirationDate')}")
    return created["data"]["id"]


def ensure_p12(out: pathlib.Path) -> str:
    p12, pw_file = out / "dist.p12", out / "p12_password.txt"
    if p12.exists() and pw_file.exists():
        print("  reusing dist.p12")
        return pw_file.read_text()
    sh("openssl", "x509", "-inform", "DER", "-in", str(out / "dist.cer"),
       "-out", str(out / "dist.pem"))
    pw = base64.urlsafe_b64encode(os.urandom(15)).decode().rstrip("=")
    pw_file.write_text(pw)
    pw_file.chmod(0o600)
    # -legacy: Apple tooling expects the classic PKCS#12 algorithms, which
    # OpenSSL 3 no longer writes by default.
    sh("openssl", "pkcs12", "-export", "-legacy",
       "-inkey", str(out / "dist.key"), "-in", str(out / "dist.pem"),
       "-out", str(p12), "-passout", f"pass:{pw}")
    p12.chmod(0o600)
    print("  built dist.p12")
    return pw


def ensure_profile(asc: ASC, out: pathlib.Path, bid_id: str, cert_id: str,
                   profile_name: str) -> pathlib.Path:
    """Fetch or create the App Store profile for one bundle id.

    `profile_name` is Apple's globally-unique label for the profile, not the
    app name — callers pass "<App> App Store" for the main app and
    "<XcodeTarget> App Store" for each extra target, so a watch companion never
    collides with its host.

    profileType is IOS_APP_STORE for extra targets too, watchOS included: Apple
    has no watch-specific store profile type and an iOS profile covers "iOS and
    watchOS apps and App Clips". Verified 2026-09-22 — the POST for
    com.sagearbor.mindshift.app.watchkitapp succeeded with IOS_APP_STORE.
    """
    path = out / "AppStore.mobileprovision"
    for p in asc.get("/v1/profiles?limit=200")["data"]:
        a = p["attributes"]
        if a["name"] != profile_name:
            continue
        if a["profileState"] == "ACTIVE":
            if not path.exists():
                full = asc.get(f"/v1/profiles/{p['id']}")["data"]["attributes"]
                path.write_bytes(base64.b64decode(full["profileContent"]))
            print(f"  reusing profile '{profile_name}' (expires {a.get('expirationDate')})")
            return path
        # Adding or removing an App ID capability (Sign in with Apple, push,
        # HealthKit …) flips every profile bound to it to INVALID, and Apple
        # refuses a second profile with the same name — so clear it out and
        # reissue rather than failing with a name conflict.
        print(f"  profile '{profile_name}' is {a['profileState']} "
              f"(a capability changed) — deleting and reissuing")
        asc.delete(f"/v1/profiles/{p['id']}")
        path.unlink(missing_ok=True)
    created = asc.post("/v1/profiles", {"data": {
        "type": "profiles",
        "attributes": {"name": profile_name, "profileType": "IOS_APP_STORE"},
        "relationships": {
            "bundleId": {"data": {"type": "bundleIds", "id": bid_id}},
            "certificates": {"data": [{"type": "certificates", "id": cert_id}]}}}})
    a = created["data"]["attributes"]
    path.write_bytes(base64.b64decode(a["profileContent"]))
    print(f"  created profile '{profile_name}' expiring {a.get('expirationDate')}")
    return path


def remember_extra_targets(out: pathlib.Path, extras: list, clear: bool) -> list:
    """Persist --extra-target choices so a later bare run cannot silently undo them.

    The failure this prevents: once an app has a watch target, running

        ios_credentials_bootstrap.py --bundle-id <id> --name <app>

    without repeating --watch-target would write credentials.json back to the
    single-target form, and the NEXT cloud build would fail at signing the watch
    target with an error pointing nowhere near this script. Different sessions
    and different repos share this tool, so "remember to pass the flag" is not a
    safeguard anybody can rely on. The extras are therefore state, remembered
    next to the profile they belong to, exactly like the certificate id.

    Explicit flags always win and are re-saved; --no-extra-targets is the way to
    deliberately go back to a single target.
    """
    state = out / "extra_targets.json"
    if clear:
        state.unlink(missing_ok=True)
        print("  cleared the remembered extra targets (--no-extra-targets)")
        return []
    if extras:
        state.write_text(json.dumps([list(e) for e in extras], indent=2) + "\n")
        state.chmod(0o600)
        return extras
    if state.exists():
        remembered = [tuple(e) for e in json.loads(state.read_text())]
        if remembered:
            print("  reusing remembered extra target(s): " +
                  ", ".join(f"{t} ({sfx})" for sfx, t in remembered))
            print("  (pass --no-extra-targets to drop them)")
        return remembered
    return []


def parse_extra_targets(args) -> list[tuple[str, str]]:
    """Normalise --watch-target / --extra-target into [(bundle suffix, Xcode target)].

    --watch-target NAME is sugar for --extra-target .watchkitapp=NAME, because
    `.watchkitapp` is the suffix Apple expects on a watchOS companion and
    getting it wrong is only discovered at upload time.
    """
    extras: list[tuple[str, str]] = []
    if args.watch_target:
        extras.append((".watchkitapp", args.watch_target))
    for spec in args.extra_target or []:
        suffix, sep, target = spec.partition("=")
        if not sep or not suffix or not target:
            die(f"--extra-target expects SUFFIX=XCODE_TARGET_NAME, got {spec!r}")
        if not suffix.startswith("."):
            die(f"--extra-target suffix must start with a dot, got {suffix!r}")
        extras.append((suffix, target))
    seen = set()
    for suffix, target in extras:
        if suffix in seen:
            die(f"--extra-target/--watch-target gave the suffix {suffix} twice")
        seen.add(suffix)
    return extras


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-id", required=True)
    ap.add_argument("--name", required=True, help="app name, used for cert/profile labels")
    ap.add_argument("--project-dir", default=".", help="directory holding eas.json")
    ap.add_argument("--email", default="sagearbor@gmail.com")
    ap.add_argument("--cert-name", default="Sage Arbor Distribution",
                    help="Common Name on the shared team certificate")
    ap.add_argument("--store", default=None,
                    help="root for the artefacts "
                         "(default ~/.config/ios-credentials)")
    ap.add_argument("--xcode-target", default=None,
                    help="Xcode target name of the main app — the credentials.json "
                         "key in multi-target mode. Defaults to --name. Read it from "
                         "`grep productName ios/*.xcodeproj/project.pbxproj`; only "
                         "used when there is more than one target.")
    ap.add_argument("--no-extra-targets", action="store_true",
                    help="forget any remembered extra targets and write the "
                         "single-target credentials.json")
    ap.add_argument("--watch-target", default=None, metavar="XCODE_TARGET",
                    help="Xcode target name of a watchOS companion. Mints "
                         "<bundle-id>.watchkitapp and its own App Store profile.")
    ap.add_argument("--extra-target", action="append", metavar="SUFFIX=XCODE_TARGET",
                    help="generic form of --watch-target for app extensions and "
                         "App Clips, e.g. .shareextension=ShareExtension. Repeatable.")
    args = ap.parse_args()

    extras = parse_extra_targets(args)
    main_target = args.xcode_target or args.name

    root = pathlib.Path(args.store or "~/.config/ios-credentials").expanduser()
    # The certificate is shared by every app on the team; only the profile is
    # per bundle id (the whole bundle id, so com.x.app and com.y.app never
    # collide). An extra target is just another bundle id under that rule — it
    # gets its own directory and its own profile, and reuses the certificate.
    team_out = root / "_team"
    out = root / args.bundle_id
    for d in (root, team_out, out):
        d.mkdir(parents=True, exist_ok=True)
        d.chmod(0o700)
    team = os.environ.get("APPLE_TEAM_ID") or die("APPLE_TEAM_ID is not set")

    extras = remember_extra_targets(out, extras, args.no_extra_targets)

    print(f"App Store Connect bootstrap for {args.bundle_id} (team {team})")
    asc = ASC()
    bid_id = ensure_bundle_id(asc, args.bundle_id, args.name, team)
    # ONE certificate for the whole team, minted at most once — see
    # ensure_certificate(). Extra targets must never reach this code again:
    # Apple caps iOS distribution certificates at two per team and a third app
    # or target that minted its own would break signing for everything.
    cert_id = ensure_certificate(asc, team_out, args.email, args.cert_name)
    pw = ensure_p12(team_out)
    profile = ensure_profile(asc, out, bid_id, cert_id, f"{args.name} App Store")

    p12 = {"path": str(team_out / "dist.p12"), "password": pw}
    targets: list[tuple[str, pathlib.Path]] = [(main_target, profile)]

    for suffix, target_name in extras:
        extra_bundle_id = args.bundle_id + suffix
        extra_out = root / extra_bundle_id
        extra_out.mkdir(parents=True, exist_ok=True)
        extra_out.chmod(0o700)
        print(f"  --- extra target {target_name} ({extra_bundle_id})")
        extra_bid = ensure_bundle_id(asc, extra_bundle_id, target_name, team)
        # Profile names are globally unique per team, so key them on the Xcode
        # target name rather than the app name — "MindShift App Store" is
        # already taken by the host app.
        extra_profile = ensure_profile(asc, extra_out, extra_bid, cert_id,
                                       f"{target_name} App Store")
        targets.append((target_name, extra_profile))

    creds = pathlib.Path(args.project_dir) / "credentials.json"
    if len(targets) == 1:
        # Single-target form. Kept for apps with one target so existing repos
        # are not churned; EAS reads either shape.
        payload = {"ios": {"provisioningProfilePath": str(profile),
                           "distributionCertificate": p12}}
    else:
        payload = {"ios": {name: {"provisioningProfilePath": str(path),
                                  "distributionCertificate": p12}
                           for name, path in targets}}
    creds.write_text(json.dumps(payload, indent=2) + "\n")
    creds.chmod(0o600)
    print(f"\nwrote {creds}")
    if len(targets) > 1:
        print("  multi-target form, keyed by Xcode target name: " +
              ", ".join(name for name, _ in targets))
        print("  these must match `productName` in the generated pbxproj AND")
        print("  extra.eas.build.experimental.ios.appExtensions[].targetName.")
    print('Now set "credentialsSource": "local" in the eas.json ios build profile,')
    print("and keep credentials.json out of git.")


if __name__ == "__main__":
    main()
