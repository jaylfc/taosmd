# A2A bus authentication: design and transition

Status: **STAGE 1 SIGNED OFF** by @taOS-dev, 2026-08-13, reviewed against master 0f0a9f8.
**Stages 2 and 3 are NOT signed off** and each now carries a binding entry gate, recorded in
its own section below. They change what the bus rejects and therefore what a coordination
failure looks like fleet-wide, so neither advances on a date or on this document alone.

Open and NOT settled by that sign-off: whether human principals get a real principal store or
are withdrawn through the session/auth layer. That is Jay's decision, @taOS-dev is putting it
to him with a recommendation for the session/auth layer, and **nothing in this transition may
depend on human revocation working until it is made.** Do not sequence Stage 2 work that
assumes either shape. See open question 1.

**Read the live risk with it, because it runs counter-intuitively in the SAFE direction and
"unresolved" invites the opposite reading.** Nothing is broken today. With no
`human_principal_ids` configured, #235's guard condition is always true, so revocation
currently applies to **everyone**. The hole opens the first time the feature is used for its
purpose, not before.

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
3. The revocation feed is honoured **for agent identities**, which is what the feed can
   actually express.

   **This clause deliberately does NOT promise human-principal revocation, and the earlier
   version of it did.** That was a blocking defect, caught by @taOS-dev's review: Target
   state is the NORMATIVE section, so promising a guarantee that open question 1 disproves
   would instruct an implementer to build a human-revocation path that silently never fires.
   That is the exact failure shape this whole document exists to prevent, written into the
   one section an implementer is most likely to build from. Human principals are open
   question 1 below, and **nothing in this transition may depend on human revocation working
   until Jay has decided** between a real principal store and withdrawal through the
   session/auth layer. PR #244's docstring needs the same correction, for the same reason:
   "pending" implies data that is on its way, and it is not.

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

### Ruling: ONE promoted helper, and it is a slug match (@taOS-dev, 2026-08-13)

> **SYMBOLS AHEAD DO NOT EXIST ON MASTER YET.** `_normalise_handle`, and the raw-`from_`
> comparison described in the Stage 1 section, both live in **PR #241**, which is open and
> blocked. This document sits on master, and a spec on master reads as describing master, so
> without this marker an implementer would take the Stage 1 prerequisite as already
> satisfied. It is not. Flagged by @taOS-dev's review of this file at master 0f0a9f8.

A third local copy of this rule must not land. `_normalise_handle` (strip `@`, casefold) is
promoted to one shared identity-slug helper, carded as tsk-pgtl4b, landing with #241's
revision with #233 rebased onto it. Two constraints come with it, and both are load-bearing:

1. **It is a slug match, NOT an identity check**, and the docstring must say so. Two
   distinct agents sharing a stem would unify under it, and #241's call site is
   authorisation. Anything that decides *who someone is* needs more than this function.
2. **The mint stamp is an explicit, documented parameter, defaulting to NOT stripping.**

That default matters because of a constraint from the other side of the fleet. The
OS-native agent (controller #2391, merged) mints a per-install, owner-linked identity at
first boot, spelled `@taOS-agent-<install8>`. Normalisation must **not** collapse those:
the registry's partial unique index, quoted exactly because the second half carries the
argument, is

```
WHERE status = 'active' AND handle != ''
```

(`agent_registry_store.py:67`). It rejects the second insert the moment two installs share a
registry, which forecloses the account/cluster model Jay has deliberately kept open. The
`handle != ''` half matters: blank-handle rows do not contend, so the collision being
guarded against is specifically between two **named** installs, which is exactly the
`@taOS-agent-<install8>` case. That agent authenticates as a registry identity with
`a2a_send` + `a2a_receive` scopes only.

So the two rules take **different inputs and are not in conflict**: strip the mint stamp
when comparing against a message *body* or resolving self from `TAOS_AGENT_CANONICAL`; do
not strip it for Stage 1's `from_normalised`.

### What the `from` field actually contains (measured, and it narrowed the fix)

Claim under test, from @taOS-dev: a canonical id never appears in a `from` field. Measured
over the last 400 bus messages:

| Spelling family | `from` count |
|---|---|
| `@handle` | 245 |
| canonical (`slug-YYYYMMDD-HHMMSS`) | 106 |
| bare slug | 49 |

The claim is false as stated: canonicals do appear, from four lane agents. But grouping
every sender by stem, **not one canonical has a bare-or-`@` twin**; the only agent on the
bus writing `from` under two spellings is `@taOSmd-dev`/`taosmd-dev`, me.

That is why mint-stripping stays **out** of Stage 1's `from` normalisation: it would unify
nothing that is actually split, and it would collapse two installs of one lane into a
single identity. `_normalise_handle` as it stands is sufficient for Stage 1 and satisfies
the install-discriminator constraint above.

**Scope, so this is not over-read**: `from` fields only, last 400 messages. The
seven-channel table below is channel *membership* rows, a different field, and it has not
been re-measured under this grouping. Carded as tsk-rf5gwb, and it must be measured before
Stage 2 rather than assumed to carry over.

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

**Prerequisite, found by reviewing Stage 1's own implementation (PR #241).** That exit test
is unreachable today, and no amount of waiting fixes it. `_registry_verifier.authorize`
compares the token `sub` against the **raw** `from_`, so normalisation is applied to the
annotation but skipped in the one place it changes an outcome. Proven on the live server
with a bare-handle control: the same token gives `auth=verified` for `from=taosmd-dev` and
`auth=invalid` for `from=@taosmd-dev`. Since `@handle` is how every raw-bus send appears,
and it is 245 of the last 400 messages, the drivers can never all read `verified` and
Stage 2 is unreachable behind it.

Two consequences for the plan, not just for that PR:

- Normalisation must be applied **at the comparison**, not only at the annotation. This is
  the tsk-pgtl4b helper's real call site.
- **An unverified `sub` must never be written into `verified_sub`.** #241 populates it from
  an unverified `jwt.decode` on the AuthError path, so a token forged with an unknown key
  reads `auth=invalid` while `from_mismatch=False`, a forgery presenting as *consistent*,
  on precisely the rows Stage 2's mismatch gate exists to scrutinise, and on the field the
  membership migration keys on. Stage 1's data is the whole point of Stage 1; if the
  annotation can be attacker-chosen, Stages 2 and 3 are built on it. Use `from_mismatch=None`
  (unknown) on the invalid path and a separate `claimed_sub` for the unverified peek, and
  split bad-signature from sub-vs-`from` mismatch: today they are indistinguishable, and one
  is a misconfigured sender while the other is an attacker.

### Stage 2: verify and warn

Unchanged from today's default, but now with Stage 1's attribution: an invalid or
mismatched token logs loudly and is still accepted. Any legitimate sender broken by
normalisation surfaces here, not in production rejection.

Exit test: zero `invalid` and zero `from`-mismatch events for 48 hours, and a deliberate
negative probe (see below) produces exactly one warning.

**ENTRY GATE, set by @taOS-dev 2026-08-13 and binding. Stage 2 does not begin until BOTH:**

1. **The read-path fix is landed AND DEPLOYED**, not merely merged. This is the
   shipped-versus-deployed distinction measured on #2390 above, applied to ourselves: Stage
   1's entire product is annotation data, and we would be reading that data through the very
   path that currently drops cursors and answers `all` with silence. Deployed means verified
   against the running box, by the markers in the deployed tag rather than the merged branch.
2. **tsk-rf5gwb is closed**, meaning channel *membership* rows have been re-measured under
   stem grouping. The 400-message result above is scoped to `from` fields and must not be
   assumed to carry over to a different field.

### Stage 3: enforce

`a2a_auth_enforce` flips. Unsigned and invalid sends are rejected with a 401 naming the
reason.

Rollback: the flag flips back with no data migration, and Stage 2 remains correct
behaviour indefinitely. Document the rollback command in the runbook before the flip, not
after.

**ENTRY GATE, set by @taOS-dev 2026-08-13 and binding. Stage 3 does not begin until BOTH:**

1. **The membership migration is complete**, with the inventory regenerated **at migration
   time** rather than trusted from the table in this document. That table is a snapshot of
   2026-08-13 and says so.
2. **All five reject proofs below have been observed failing FOR THE INTENDED REASON**, with
   fenced evidence and a live-assertion control, **and fail-closed has been DEMONSTRATED
   going red rather than merely specified.** Their reasoning, which I am recording because it
   is the sharper half: the two fail-safe rules in this document are exactly the kind that
   get written down and then implemented fail-open, and a reject path that has only ever been
   seen passing is unproven where it counts. "Not succeeding" is not the same as "failing for
   the intended reason", and only the second is evidence.

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

### Re-measured 2026-08-13 18:05Z (`channel=*` row corrected 18:20Z), with a discriminating probe

I had recorded upstream #2390 as having fixed `all`/`*` and unknown-param rejection. The
claim was missing one word: #2390 is **shipped to dev but not deployed**. The live box is
`1.0.0-beta.48`, tagged 2026-08-12, and #2390 merged to dev at 15:29Z on 2026-08-13, so none
of that behaviour exists on the server either of us can reach (@taOS-dev, bus 2481, who
checked the deployed tag's markers). Measured against what is actually running:

| Call | `/api/a2a/bus/messages` | Raw bus `:7900/a2a/messages` |
|---|---|---|
| `channel=build` (control) | 200, 3 msgs | 200 |
| `channel=all` | **200, zero**, identical to `channel=doesnotexist` | n/a |
| `channel=*` | **400**, `wildcard channel not supported here; use /api/a2a/bus/stream for all-threads` | n/a |
| `thread=build` | 400 `channel required` | n/a |
| unknown `bogusparam=1` | **200, silently ignored** | **200, silently ignored** |
| `after=0` and `after=2479` | **identical ids to bare** | **identical ids to bare** |
| `since_id=2479` | identical to bare | identical to bare |
| `since=<unix ts>` | **filters: 4 of 5** | **filters: 4 of 5, then 0 at a later ts** |

The earlier version of this probe was ambiguous and I nearly filed it as-is: `after=2400`
with `limit=2` returns the newest two messages, which is what a *honoured* cursor and an
*ignored* one both predict. `after=0` and `after=2479` are the discriminating values: a
honoured cursor must give opposite answers to those two, and they are byte-identical. The
`since=<ts>` row is the positive control that keeps the whole table meaningful: the filter
machinery demonstrably works on both servers, so the ignored params are ignored rather than
untestable.

**Correction to my own first version of this table, caught by @taOS-dev at bus 2481.** I
originally recorded `channel=*` as "200, zero" alongside `all`. It is a **400** with an
explicit, helpful error naming the right endpoint. My probe counted `len(messages)` and
discarded the HTTP status, and a 400 error body has no `messages` key, so `len` returned 0
for it exactly as it does for a genuine empty 200. Verified with the control: the old
status-blind predicate returns `n=0` for both `channel=doesnotexist` (a real 200) and
`channel=*` (a real 400). Two different facts, one output, which is the same defect as the
`after=` row two paragraphs down and the third instance of it today. The endpoint therefore
**does** have a working wildcard guard; only the `all` spelling is the silent one.

**Scope, and it is a real limit**: `/api/a2a/bus/read` returns 401 for my agent token, so I
cannot say #2390 did not land *there*. What I can say is that the endpoint we are about to
recommend to every new agent still silently returns nothing for the documented `all` idiom.
The raw-bus half is carded as tsk-d64alg; it must be coordinated before it lands, because
`a2a_watch.sh` and `lead_bus_watch.sh` both send `after=` today and would begin 400ing.

## Open questions for review

**1. Human principals and revocation: ANSWERED, and the answer is worse than the question.**
Asked at bus 2474, answered from source by @taOS-dev at 2475: the controller does **not**
publish human principals on the revocation feed, and **cannot**. `list_revoked()` selects
from `agent_registry`, whose insert writes the agent's *owner* as `user_id` rather than
creating a principal row; `human_principal` appears nowhere in the controller.

So this is not a check waiting on data to arrive. It is a check whose data source is
structurally incapable of carrying the case, and PR #244's docstring (which frames it as
pending) must be corrected before merge. **Do not record this hole as closed**, and do not
close its card as though human withdrawal works. The controller half is @taOS-dev's.

Live risk is currently small and bounded: with no `human_principal_ids` configured, #235's
guard condition is always true, so revocation applies to everyone. The hole opens only once
humans are actually configured, which is to say it opens the first time the feature is
used for its purpose.

**2. Ordering: does the read-path fix land before, with, or after Stage 1?** My view is
unchanged and the re-measurement above strengthens it: **before**. Stage 1's entire value is
the attribution data, we will be reading that data through this path, and the path currently
answers "no messages" and "cursor accepted" when neither is true. Collecting 72 hours of
Stage 1 annotations through a reader that silently drops its cursor is how we would end up
trusting a window we never actually saw.

**3. Merge sequencing for #244 and #235** (added after review). #244 is cut from master and
is a parallel reimplementation of #235's human-principal code, not a patch on top. Merged in
either order, the docstring describing the revocation fix conflicts while **the guard that
undoes it auto-merges silently**, taking #235's weaker version, so a resolver sees a prose
disagreement and ships an auth hole. #244 must rebase onto `exec/tsk-legqtr` (or onto master
once #235 lands) before either merges. If anyone hand-resolves it instead, #244's three
revocation tests are the acceptance check and must be run *after* resolution.
