# TURN for in-app calls — options, prices, and what to paste where (2026-08-25)

## Why this document exists

In-app calls (`server/calls.py`, `docs/plans/2026-08-25-in-app-calls.md`) send audio
**peer-to-peer over WebRTC**; the server only relays signaling. Today the deployment
ships **STUN only** (`stun:stun.l.google.com:19302`). STUN tells each phone its own
public address — which is enough on most home Wi-Fi and useless behind
**carrier-grade NAT (CGNAT)**, which is what both phones sit behind on cellular data.
Two Pixels on mobile data will frequently fail to connect, with no error the owner
can read.

The fix is a **TURN relay**: a server both phones can always reach, which forwards the
media for them. This document picks one.

> **The one-line summary:** use **Cloudflare Realtime TURN** — 1,000 GB/month free,
> `turns:` on port 443, one dashboard click and one `curl`. Set
> `MINDSHIFT_TURN_URLS` + `MINDSHIFT_TURN_USERNAME` + `MINDSHIFT_TURN_CREDENTIAL` on
> Cloud Run. If you'd rather have per-member credentials the server mints itself
> (`MINDSHIFT_TURN_SECRET`), that path needs a TURN server that speaks the standard
> shared-secret scheme — ExpressTURN Premium ($9/mo) or self-hosted coturn.

---

## The credential schemes, because they decide which vendor fits

There are two ways a client gets a TURN username/password.

**1. Static long-term credentials.** One username and password, configured on the
server, handed to every client. Simple; the weakness is that the password is shared
by everyone who ever joins a call and does not expire.

**2. TURN REST ephemeral credentials** (`draft-uberti-behave-turn-rest-00`, which is
what coturn's `use-auth-secret` implements). The app server and the TURN server share
a secret. The app server mints, per user, per handout:

```
username   = "<unix-expiry>:<user-name>"
credential = base64( HMAC-SHA1( shared_secret, username ) )
```

The TURN server recomputes the same HMAC from its own copy of the secret, so nothing
is provisioned per user and a leaked credential dies at `expiry`. The base64 is over
the **raw 20 binary bytes** of the SHA-1 digest (hex-then-base64 is the classic bug).
Sources: [draft-uberti-behave-turn-rest-00](https://datatracker.ietf.org/doc/html/draft-uberti-behave-turn-rest-00),
[coturn README.turnserver](https://github.com/coturn/coturn/blob/master/README.turnserver),
[turnserver(1)](https://manpages.debian.org/testing/coturn/turnserver.1.en.html).

MindShift now implements **scheme 2** (`calls.turn_rest_credentials`, behind
`MINDSHIFT_TURN_SECRET` / `MINDSHIFT_TURN_REALM`) and keeps scheme 1 working
(`MINDSHIFT_TURN_USERNAME` / `MINDSHIFT_TURN_CREDENTIAL`).

**Who supports what** — this is the awkward finding, and it is why the recommendation
below is split:

| Provider | Shared-secret HMAC (server mints offline) | Vendor API required |
|---|---|---|
| Cloudflare Realtime | ❌ | ✅ `POST …/credentials/generate-ice-servers` |
| Twilio NTS | ❌ | ✅ `POST /Tokens.json` |
| Metered | ❌ | ✅ `POST /api/v1/turn/credential` |
| ExpressTURN **Premium** ($9/mo) | ✅ | optional |
| ExpressTURN Free | ❌ (static user/pass only) | n/a |
| Self-hosted coturn | ✅ native (`--use-auth-secret`) | n/a |

Cloudflare's FAQ is explicit that its response is *shaped* like the draft but is not
the HMAC scheme and carries no TTL field — you cannot compute a Cloudflare credential
offline. ([FAQ](https://developers.cloudflare.com/realtime/turn/faq/))

---

## The options

### 1. Cloudflare Realtime TURN — **recommended**

* **Price:** $0.05 / GB, with **1,000 GB/month free** (a single allowance shared with
  their SFU). STUN at `stun.cloudflare.com` is free and unmetered. Only egress from
  Cloudflare's edge to the TURN client is metered, and only after successful auth.
  A relayed *audio* call is roughly 30–60 MB/hour, so the free tier is thousands of
  call-hours. ([pricing](https://developers.cloudflare.com/realtime/sfu/pricing),
  [FAQ](https://developers.cloudflare.com/realtime/turn/faq/))
* **Endpoints** ([docs](https://developers.cloudflare.com/realtime/turn/)):

  | Transport | Host | Ports |
  |---|---|---|
  | STUN/UDP | `stun.cloudflare.com` | 3478, 53 |
  | TURN/UDP | `turn.cloudflare.com` | 3478, 53 |
  | TURN/TCP | `turn.cloudflare.com` | 3478, 80 |
  | TURN/TLS | `turn.cloudflare.com` | 5349, **443** |

  `turns:…:443` is the one that survives hotel/corporate Wi-Fi. Port 53 is blocked by
  many ISPs and by Chrome/Firefox — don't use it. Relay addresses are IPv4-only, and
  Cloudflare does not implement RFC 6062, so the *relayed* leg is always UDP (fine
  for us: both legs client→server may still be TCP/TLS).
* **Credentials:** vendor API. You hold a TURN key server-side and mint short-lived
  creds ([docs](https://developers.cloudflare.com/realtime/turn/generate-credentials/)):

  ```bash
  curl -X POST \
    "https://rtc.live.cloudflare.com/v1/turn/keys/$TURN_KEY_ID/credentials/generate-ice-servers" \
    -H "Authorization: Bearer $TURN_KEY_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"ttl": 86400}'
  ```

  Returns `{"iceServers": {"urls": [...], "username": "...", "credential": "..."}}`.
  Revoke early with `POST …/credentials/$USERNAME/revoke`.
* **Ops:** none.
* **Catch:** because it is API-minted, it plugs into MindShift's **static** env vars,
  not `MINDSHIFT_TURN_SECRET`. See the click-path below for exactly what that means
  in practice.

### 2. ExpressTURN — the shared-secret option

* **Free:** 1,000 GB/month, but **UDP/TCP on port 3478 only** — no TLS, no 443. Static
  username/password only.
* **Premium: $9/month, 5,000 GB** — adds ports 80 and 443 with TLS, 18 locations with
  GeoDNS, and **shared-secret authentication**, i.e. exactly the scheme
  `MINDSHIFT_TURN_SECRET` implements. ([expressturn.com](https://www.expressturn.com/))
* **Catch:** no SLA, an opaque operator; BlogGeek rates it lowest on quality of the
  providers it ranks. Fine for a hobby demo, not for anything you'd be paged about.

### 3. Twilio Network Traversal Service — priced for a business

* $0.40 / GB in US-West and Europe, $0.60 in APAC, $0.80 in Sydney/São Paulo, and
  usage counts **both directions**. No free tier.
  ([pricing](https://www.twilio.com/en-us/stun-turn/pricing))
* Credentials via `POST /2010-04-01/Accounts/$SID/Tokens.json` with `Ttl` (max
  86400). No configurable shared secret.
  ([API](https://www.twilio.com/docs/stun-turn/api))
* Its 443 endpoint is `turn:` over plain TCP, not `turns:` TLS — weaker than
  Cloudflare's exactly where it matters.
* Roughly 16× Cloudflare's rate, before Cloudflare's free terabyte. **Skip.**

### 4. Metered.ca (Open Relay)

* Free tier is **500 MB/month** (ingress+egress) — note that the widely-copied
  "20 GB free" figure is stale. Paid: $99/mo for 150 GB, $199/mo for 500 GB.
  ([pricing](https://www.metered.ca/pricing))
* Credentials are API-minted (`POST /api/v1/turn/credential`); no documented
  shared-secret scheme. The HMAC-SHA256 in their docs is *webhook signature*
  verification — unrelated.
  ([docs](https://www.metered.ca/docs/turn-server-service/creating-turn-credentials/))
* Nice touch: endpoints run on **80 and 443** (`global.relay.metered.ca`), good for
  restrictive networks. But 500 MB free is roughly ten relayed call-hours.

### 5. Xirsys

* No self-serve free tier worth the name; paid plans from **$39/month**, $0.09/GB
  overage on relayed traffic. Pricing page is JS-rendered and could not be verified
  from primary source. Non-competitive against a free terabyte. **Skip.**

### 6. Self-hosting coturn on the existing GCP project

* **Cloud Run cannot host it.** Cloud Run serves HTTP/1.1 and HTTP/2 over TLS only —
  no UDP, no arbitrary TCP ports, no stable public IP per instance. coturn needs raw
  UDP/3478 plus a wide UDP relay port range. It would have to be Compute Engine.
* **Real cost, us-central1:**

  | Item | Cost |
  |---|---|
  | e2-micro under GCP Always Free (1/mo, us-central1) | $0.00 |
  | e2-micro on demand | ~$6.11/mo |
  | **External IPv4 on a running VM — NOT in the free tier** | **$3.65/mo** |
  | Internet egress, Premium tier, first TB | ~$0.12/GB |

  So the floor is **~$3.65/month plus $0.12/GB**, i.e. more than 2× Cloudflare's paid
  rate before you have written a line of config, against Cloudflare's $0.
* **Ops you are signing up for:** reserve a static external IP (an ephemeral one
  changes on stop/start and silently breaks ICE); open 3478/udp+tcp, 5349/udp+tcp,
  and the relay range 49152–65535/udp (narrowable with `--min-port`/`--max-port`);
  set `external-ip=<public>/<private>` because a GCP VM only sees its private address
  through 1:1 NAT — **omitting this is the single most common reason a cloud coturn
  silently fails**; and run Let's Encrypt with renewal for `turns:` on 5349/443.
* **When it wins:** third-party 2026 analysis puts self-hosted coturn ahead of
  Cloudflare only above roughly 50M relayed minutes/month, and then on bare metal
  rather than GCP. We are six orders of magnitude below that.
* **Only reason to do it here:** you want the per-member ephemeral credentials
  (`MINDSHIFT_TURN_SECRET`) *and* full control of where the media goes.

---

## Recommendation

**Use Cloudflare Realtime TURN.** Free at our volume, `turns:` on 443, no servers to
own. Its credentials are API-minted rather than HMAC-derived, so configure them
through MindShift's **static** env vars, with a TTL you refresh — one `curl`,
however often you choose (a 24h TTL means a daily paste; the API accepts longer).

**If you want the per-member ephemeral path** that `MINDSHIFT_TURN_SECRET` gives you
— each member gets their own credential, expiring in 4 hours, nobody sharing a
password — the cheapest server that speaks that scheme is **ExpressTURN Premium at
$9/month** (or coturn on a GCE VM at ~$3.65/month plus egress plus your time).

A reasonable end state is **both**: Cloudflare as the primary relay via the static
vars, and ExpressTURN free as a second, no-cost entry in `MINDSHIFT_TURN_URLS` so one
vendor's incident doesn't kill a call. (One caveat: the ICE-server list carries one
username/credential pair per *entry*, and MindShift currently emits a single TURN
entry, so a second vendor with different credentials needs a small code change.)

---

## Exact click-path — Cloudflare, from zero to a working relay

1. Sign in at **https://dash.cloudflare.com** (a free account is enough; no domain
   needed).
2. In the left sidebar choose **Realtime**. (The product was renamed from "Calls";
   the docs' own deep link is still `https://dash.cloudflare.com/?to=/:account/calls`.)
3. Open the **TURN** tab → **Create TURN key** → name it `mindshift`.
4. Copy the two values it shows you — the **Key ID** and the **API token**. The token
   is shown once.
5. Mint a credential (locally, with those two values):

   ```bash
   export TURN_KEY_ID='<the Key ID>'
   export TURN_KEY_API_TOKEN='<the API token>'
   curl -sX POST \
     "https://rtc.live.cloudflare.com/v1/turn/keys/$TURN_KEY_ID/credentials/generate-ice-servers" \
     -H "Authorization: Bearer $TURN_KEY_API_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"ttl": 86400}'
   ```

   The reply contains `username` and `credential` (96 hex characters each).
6. Set them on Cloud Run (see the env-var table below) and redeploy.
7. Open **Live Coach → Call** on the phone. The pre-flight panel's **Peer connection**
   row must read **"relay ready — a call connects even on carrier-grade NAT"**. If it
   says "TURN is configured but gave no relay candidate", the credential expired or
   was pasted wrong — mint a new one.

### Alternative click-path — ExpressTURN, for the ephemeral (HMAC) path

1. Sign up at **https://www.expressturn.com/** (email + password, no card for free).
2. Upgrade to **Premium ($9/mo)** — the free plan has neither TLS/443 nor shared-secret
   auth.
3. In the dashboard, enable **shared-secret authentication** and copy the secret.
4. Set `MINDSHIFT_TURN_SECRET` to that secret and `MINDSHIFT_TURN_REALM` to the realm
   the dashboard shows. Leave `MINDSHIFT_TURN_USERNAME`/`_CREDENTIAL` unset — the
   secret takes precedence over them anyway.

---

## Env vars to set on Cloud Run

Service `mindshift-api` (see `scripts/deploy_cloudrun.sh`).

| Variable | Meaning | Cloudflare (recommended) | ExpressTURN Premium / coturn |
|---|---|---|---|
| `MINDSHIFT_TURN_URLS` | comma-separated relay URLs | `turn:turn.cloudflare.com:3478?transport=udp,turn:turn.cloudflare.com:3478?transport=tcp,turns:turn.cloudflare.com:5349?transport=tcp` | `turn:relay1.expressturn.com:3478,turns:relay1.expressturn.com:443` |
| `MINDSHIFT_TURN_USERNAME` | static username | the 96-hex `username` from the curl | — leave unset |
| `MINDSHIFT_TURN_CREDENTIAL` | static password | the 96-hex `credential` from the curl | — leave unset |
| `MINDSHIFT_TURN_SECRET` | shared secret for TURN REST creds | — leave unset | the vendor's / coturn's `static-auth-secret` |
| `MINDSHIFT_TURN_REALM` | TURN realm (diagnostics + coturn) | — | e.g. `expressturn.com` |
| `MINDSHIFT_TURN_TTL_SECONDS` | ephemeral credential lifetime | — | optional; default `14400` (4h), floor 60, cap 86400 |

`MINDSHIFT_TURN_SECRET` **takes precedence** over the static pair when both are set.
Because it holds a secret, prefer Secret Manager over a plain `--set-env-vars`:

```bash
# one-time
printf '%s' '<the shared secret>' | \
  gcloud secrets create mindshift-turn-secret --data-file=-

gcloud run services update mindshift-api --region us-central1 \
  --set-env-vars "MINDSHIFT_TURN_URLS=turn:...,turns:...,MINDSHIFT_TURN_REALM=<realm>" \
  --set-secrets  "MINDSHIFT_TURN_SECRET=mindshift-turn-secret:latest"
```

For the Cloudflare static pair the same shape applies with
`--set-secrets "MINDSHIFT_TURN_CREDENTIAL=mindshift-turn-credential:latest"`.

Verify from a shell without touching the phone (any signed-in account):

```bash
curl -s -H "Authorization: Bearer $ID_TOKEN" https://<service-url>/calls/ice | jq
# {"ice_servers":[{"urls":["stun:..."]},{"urls":["turn:..."],"username":"...","credential":"..."}],
#  "turn_configured": true, "turn_credential_mode": "ephemeral", "ttl_seconds": 14400}
```

`turn_credential_mode` tells you which path is live: `none` (STUN only), `static`,
`ephemeral`, or `open` (URLs with no auth).

---

## What the app does with this

* `server/calls.py :: ice_servers(user_key)` builds the list per member. With
  `MINDSHIFT_TURN_SECRET` set it mints `("<expiry>:<uid>", base64(HMAC-SHA1(...)))`
  per handout, so each member has their own expiring credential.
* `GET /calls/ice` (`server/routers/calls.py`) returns the same list without creating
  a call.
* `apps/mobile/src/live/call/iceProbe.ts` gathers candidates against it and turns the
  result into one line in the Call-mode pre-flight panel:
  * *relay ready* — a relay candidate came back; CGNAT is handled.
  * *direct likely — but no TURN is configured…* — STUN reflected us; a cellular call
    can still fail.
  * *relay needed — no TURN configured* — this is the state that kills the demo.
  * *TURN is configured but gave no relay candidate…* — credentials/ports/realm wrong.

---

## Sources

- [Cloudflare Realtime TURN](https://developers.cloudflare.com/realtime/turn/) ·
  [FAQ](https://developers.cloudflare.com/realtime/turn/faq/) ·
  [pricing](https://developers.cloudflare.com/realtime/sfu/pricing) ·
  [generate credentials](https://developers.cloudflare.com/realtime/turn/generate-credentials/) ·
  [create TURN key API](https://developers.cloudflare.com/api/resources/calls/subresources/turn/methods/create)
- [Twilio NTS pricing](https://www.twilio.com/en-us/stun-turn/pricing) ·
  [NTS docs](https://www.twilio.com/docs/stun-turn) ·
  [Token API](https://www.twilio.com/docs/stun-turn/api)
- [Metered pricing](https://www.metered.ca/pricing) ·
  [TURN overview](https://www.metered.ca/docs/turn-server-service/overview/) ·
  [creating credentials](https://www.metered.ca/docs/turn-server-service/creating-turn-credentials/)
- [Xirsys pricing](https://xirsys.com/pricing) · [FAQ](https://xirsys.com/faq/)
- [ExpressTURN](https://www.expressturn.com/)
- [BlogGeek: hosted TURN providers compared (2026)](https://bloggeek.me/webrtc-tools/nat-hosted/)
- [GCP Always Free](https://docs.cloud.google.com/free/docs/free-cloud-features) ·
  [external IP pricing](https://cloud.google.com/vpc/pricing-announce-external-ips) ·
  [network pricing](https://cloud.google.com/vpc/network-pricing)
- [Cloud Run protocol limits](https://github.com/ahmetb/cloud-run-faq/blob/master/README.md)
- [draft-uberti-behave-turn-rest-00](https://datatracker.ietf.org/doc/html/draft-uberti-behave-turn-rest-00) ·
  [coturn README.turnserver](https://github.com/coturn/coturn/blob/master/README.turnserver) ·
  [turnserver(1)](https://manpages.debian.org/testing/coturn/turnserver.1.en.html)
- [coturn vs Cloudflare TURN scaling, 2026](https://callsphere.ai/blog/vw3e-webrtc-turn-scaling-coturn-vs-cloudflare-2026)
