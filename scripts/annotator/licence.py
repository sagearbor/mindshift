"""Licence gate for open-web audio (scripts/fetch_open_audio.py).

A video is kept ONLY when one of these is true, judged from yt-dlp metadata
BEFORE anything is downloaded:

1. ``license`` names Creative Commons (YouTube's "Creative Commons Attribution
   license (reuse allowed)").
2. It is a US federal government work (17 U.S.C. 105): the uploader is an
   official House or Senate channel. Proven by BOTH an official-looking
   channel name (starts "House"/"Senate"/"U.S. House"/"U.S. Senate", or is a
   "Committee on ..." channel naming the House or Senate) AND a link to a
   house.gov / senate.gov domain on the channel itself. News outlets
   re-uploading hearings, C-SPAN, party/campaign channels and state or city
   governments do not qualify.

Anything else, including an unknown or missing licence, is rejected with a
reason. When in doubt: reject.
"""

from __future__ import annotations

import re

_CC = re.compile(r"creative\s*commons|\bcc[- ]by\b", re.I)
_FED_DOMAIN = re.compile(r"(?<![\w-])(?:[\w-]+\.)*(?:house|senate)\.gov(?![\w.-]*\.\w)", re.I)
_FED_NAME = re.compile(
    r"^(?:the\s+)?(?:u\.?\s?s\.?\s+|united states\s+)?(?:house|senate)\b"
    r"|^(?:house|senate)\s+(?:committee|subcommittee)\b"
    r"|\bcommittee\b.*\b(?:house|senate)\b|\b(?:house|senate)\b.*\bcommittee\b",
    re.I,
)
_NOT_FED = re.compile(r"\b(?:state|assembly|county|city|news|tv|network|c-span|cspan|gop|democrats|republicans|"
                      r"campaign|for congress|for senate|media|clips|channel)\b", re.I)


def is_creative_commons(info: dict) -> bool:
    lic = info.get("license") or ""
    return bool(_CC.search(str(lic)))


def is_federal_work(info: dict) -> tuple[bool, str]:
    name = str(info.get("channel") or info.get("uploader") or "").strip()
    if not name:
        return False, "no channel name"
    if not _FED_NAME.search(name):
        return False, f"channel {name!r} is not an official House/Senate name"
    if _NOT_FED.search(name):
        return False, f"channel {name!r} looks like a news/state/party channel, not a federal body"
    proof = " ".join(str(info.get(k) or "") for k in ("channel_description", "channel_url", "uploader_url"))
    if not _FED_DOMAIN.search(proof):
        return False, f"channel {name!r} has no house.gov/senate.gov link to prove it is official"
    return True, "official House/Senate channel with a .gov link"


def decide(info: dict) -> tuple[bool, str, str]:
    """(keep, licence_kind, reason)."""
    if is_creative_commons(info):
        return True, "CC BY (YouTube Creative Commons)", f"license field: {info.get('license')}"
    fed, why = is_federal_work(info)
    if fed:
        return True, "US federal government work", why
    lic = info.get("license")
    return False, "", f"licence {lic!r} is not Creative Commons and not a US federal work ({why})"
