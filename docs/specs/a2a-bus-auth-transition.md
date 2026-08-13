# A2A bus authentication: design and transition

Status: DRAFT for review by @taOS-dev, then Jay's sign-off before implementation is carded.
Owner: @taOSmd-dev. Supersedes the held Phase 2 of #138, which was blocked on the registry
identity layer that landed 2026-08-13.

## The problem in one line

Every identity guarantee the controller now enforces stops at the controller proxy and
evaporates at the real bus: `POST :7900/a2a/send` accepts any `from` from anyone, with no
credential at all.

## What already exists (do not rebuild)

- Registry Ed25519 signing, with `GET /api/agents/registry/pubkey` deliberately left
  unauthenticated so the bus can fetch the key on its own schedule. Built for this and
  never wired in.
- `taosmd/registry_auth.py`: `decode_and_verify`, `authorize_sender`, `RegistryVerifier`
  with pubkey/revoked loaders and refresh timing. The HTTP surface already uses it for
  `a2a_send` when a token is present.
- `a2a_auth_enforce` config flag, today defaulting to verify-and-warn.
- Live revocation: `check_agent_scope` re-reads registry status and grant expiry on every
  request, so suspending an agent takes effect on its next call. Confirmed by @taOS-dev
  2026-08-13. This is why the agent class does not need short token TTLs.

## Target state

1. `POST /a2a/send` requires a valid registry-signed Bearer JWT.
2. `from` is **derived from the token**, never read from the body. The body's `from` is
   ignored if present and rejected if it disagrees, so a caller cannot assert an identity.
3. The revocation feed is honoured, including for human principals (see the open question
   in the PR #235 review: humans currently skip the revocation check, which makes their
   credentials unwithdrawable).

## Handle normalisation is the first hazard, not an afterthought

Deriving `from` means mapping token `sub` to a bus handle, and that mapping is where the
fleet has already been bitten twice today:

- `seed-internal` on the controller looks up `@taOSmd-dev` while the registry stores
  `taosmd-dev`, so an exact match misses and a duplicate identity is created (@taOS-dev is
  fixing this red-first).
- Messages sent through the authenticated proxy land as `taosmd-dev`, while every message
  sent to the raw bus lands as `@taOSmd-dev`. The same agent appears under two spellings in
  one thread, and my own wake gate's self-exclusion broke on exactly this.

So: one normalisation function, applied at every boundary, with the `@` prefix and case
folded, and a test that asserts both spellings resolve to one identity. A bus that
authenticates perfectly but attributes to two handles has not solved attribution.

### The split is already materialised in bus state, not just in attribution

Measured against `GET /a2a/channels` on 2026-08-13. Seven channels carry member rows that
are the same agent under different spellings:

| Channel | Split identity | Member rows |
|---|---|---|
| build | taosmd-dev | `@taOSmd-dev`, `taosmd-dev` |
| build, hermes | hermes | `hermes`, `hermes-20260608-153000`, `hermes-20260727-001415` |
| general, taOS-taOSmd-observability | taos | `@taOS`, `taos` |
| general | taosmd | `@taOSmd`, `taOSmd-20260609` |
| taosmd-progress | taosmd | `@taOSmd`, `taOSmd` |
| taOS-taOSmd-hermes-integration | hermes | `hermes`, `hermes-20260608-153000` |

Three distinct spelling families are in play: the `@`-prefixed display handle, the bare
slugified handle, and the timestamped canonical registry id. Hermes appears under all
three.

This makes reconciliation a migration, not a code path. Normalising only new sends means
the enforce flip attributes a verified agent to one member row while its subscribers watch
another, so a correctly authenticated message can land in a channel membership nobody is
reading. Required before Stage 3:

1. An inventory pass that groups existing member rows by normalised identity and reports
   every collision (the table above is the current state, and it must be regenerated at
   migration time rather than trusted from this document).
2. A merge that keeps the union of channel memberships and preserves history, following
   the archive contract: rename-then-merge, never delete a row and never mutate history.
3. A uniqueness assertion afterwards, so a second spelling cannot be reintroduced by a
   send that bypasses normalisation.

## Fail-safe rules

- **The pubkey fetch must fail closed, never open.** If the registry is unreachable and no
  cached key is held, sends are rejected, not accepted. An unreachable registry must never
  silently downgrade every message to trusted.
- **Cache the key with an explicit age.** A cached key is used while the registry is down,
  but its age is logged on every use so a long outage is visible rather than invisible.
- **A verifier that cannot run is an error, not a pass.** If `pyjwt`/`cryptography` are
  missing, the server must refuse to start in enforce mode rather than skip verification.
  This is the #214 lesson: that test surface silently skipped for months.

## Transition: three stages, each with an exit test

The fleet's only coordination channel is this bus, and all four drivers are posting
unauthenticated right now. A cutover that begins by rejecting unsigned sends black-holes
coordination at the moment we need it most. Therefore no stage advances on a date; each
advances only when its exit test passes.

### Stage 1: accept and annotate

Verify a token when present, accept the send either way, and record on every message:
`auth=verified|unsigned|invalid`, the verified `sub`, and whether the body's `from`
disagreed with the token. Nothing is rejected. The point is data: we learn who is actually
signed before anyone is cut off.

Exit test: for 72 consecutive hours, every message from the four drivers is
`auth=verified`, and the set of `unsigned` senders is empty or contains only senders we
have consciously decided to retire.

### Stage 2: verify and warn

Unchanged from today's default, but now with Stage 1's attribution: an invalid or
mismatched token logs loudly and is still accepted. Any legitimate sender broken by
normalisation surfaces here, not in production rejection.

Exit test: zero `invalid` and zero `from`-mismatch events for 48 hours, and a deliberate
negative probe (see below) produces exactly one warning.

### Stage 3: enforce

`a2a_auth_enforce` flips. Unsigned and invalid sends are rejected with a 401 naming the
reason.

Rollback: the flag flips back with no data migration, and Stage 2 remains correct
behaviour indefinitely. Document the rollback command in the runbook before the flip, not
after.

## Prove the reject path goes red on purpose

A bus that accepts everything and a bus whose verifier silently fails open are
indistinguishable from the outside. Before Stage 3, and as a permanent test:

- an unsigned send is rejected,
- a send signed by the wrong key is rejected,
- a send whose body `from` disagrees with its token `sub` is rejected,
- a send from a revoked identity is rejected,
- a valid send is accepted and attributed to the normalised handle.

Each must be observed failing for the intended reason, not merely "not succeeding".

## Read path: fix before we standardise on it

We are about to tell every new agent to use the authenticated controller path. Measured
2026-08-13 against `GET /api/a2a/bus/messages`:

| Call | Result |
|---|---|
| no `channel` param | HTTP 400 `{"error":"channel required"}` |
| `thread=build` (the raw bus's param name) | HTTP 400, `thread` is not accepted |
| `channel=build&limit=5` / `limit=400` | works, limit honoured |
| `channel=all` | **HTTP 200, zero messages** |
| `channel=doesnotexist` | HTTP 200, zero messages (identical) |
| `since_id=` / `after=` | **silently ignored**, full window returned |
| `since=` | honoured, but takes a **ts**, not an id (see correction below) |

Two of these are the fleet's most expensive failure shape, an error that reads as quiet:

- `channel=all` is the documented idiom for `a2a-watch` on the raw bus, where it means
  every thread. On the proxy it is treated as an unknown channel name and returns success
  with nothing. An agent following our own guide gets a permanently silent bus and a 200.
- Cursor parameters are accepted and ignored, so an agent doing incremental reads believes
  it holds a cursor it does not have.

**Correction, and it was my error.** I first reported `since=` as silently ignored. It is
not. `since=` is honoured and takes a **unix timestamp**; I passed a message id (`2400`),
which as a ts is 1970, so everything matched and the full window came back looking exactly
like a dropped parameter. Measured on the raw bus: `since=<real ts>` returned 2 messages
where the bare read returned 500. `after=` and `since_id=` genuinely are ignored, both
returning the identical 500-message window.

So there are two distinct defects here, not one:

- **Unrecognised params silently dropped** (`after=`, `since_id=`). These want the 400 that
  the proxy now returns.
- **A honoured param whose unit is not what callers reach for** (`since=` takes a ts, every
  agent reaches for an id). A 400 cannot catch this, because the value is valid; only a
  documented unit and an out-of-range sanity check can. A bare integer that is plausibly a
  message id and implausibly a recent timestamp should be rejected with a message saying
  which unit is wanted.

The methodological lesson is mine to carry: my probe never proved it could produce a
filtered result, so I read absence into what was actually a wrong-unit query. That is the
same rule I have applied to others all week, and I did not apply it to myself.

Required before the authenticated read becomes the recommended path: support `channel=all`
or reject it explicitly, honour one documented cursor parameter or reject unknown ones with
a 400, and accept `thread` as an alias for `channel` (or document the difference loudly).
An unknown query parameter must be a 400, never a silent no-op.

## Open questions for review

1. Human principals and revocation (PR #235). Agent credentials are withdrawable today via
   live status re-read; human ones are not. The bus cannot honour a revocation the identity
   layer does not express.
2. Ordering: does the read-path fix land before, with, or after Stage 1? My view is before,
   because Stage 1's value is the attribution data and we will be reading it through that
   path.
