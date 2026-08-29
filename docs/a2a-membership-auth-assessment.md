# A2A membership auth assessment

## Finding 1: ownership is self-asserted, not bound to the caller's token

The three membership write endpoints (`POST /a2a/threads`,
`POST /a2a/threads/{thread}/members`, `DELETE /a2a/threads/{thread}/members/{principal}`)
derive the caller's identity from the `agent` field in the request body. Nothing
binds that field to the caller's verified token. A client can therefore claim to
be any principal, including one it does not control.

The sibling read path (`/a2a/mentions`) does bind the `reader` query parameter to
the token's `sub` claim (http_server.py:1832) and returns 403 on mismatch. The
membership write path does not.

Impact: a bearer of a valid token for principal A can add or remove members on
behalf of principal B, or create a thread claiming B as the owner. The audit log
records the claimed `agent`, not the verified token subject, so the tampering is
not surfaced without a separate cross-check.

## Finding 2: a2a_create_thread does not check for existing conversation history

`a2a_create_thread` checks only the `a2a_membership` store for an existing
thread entry. It does not consult the archive or the message index. A principal
who has never posted on a thread can therefore call `a2a_create_thread` on a
thread name that already carries message history, becoming sole owner.

Measured: after one message on thread "build", a stranger created it and became
sole owner, with the duplicate-thread guard refusing a second claim in the same
run as the positive control.

## Recommended binding

Bind the membership write path to the caller's verified token, matching the
read-path pattern:

1. Extract the token subject (`sub`) from the verified token when registry auth
   is configured, or require the `server_token` in standalone mode.
2. Compare the token subject to the `agent` field from the request body. On
   mismatch, return 403 before any membership lookup.
3. In `a2a_create_thread`, before checking the membership store, check whether
   the thread name already appears in the archive as an `EVENT_A2A` message
   thread. If it does, refuse creation unless the caller is already an owner
   (i.e., fall through to the normal add-member path or return 409).

This binding preserves the current behaviour when no auth is configured
(standalone single-user mode), so it is additive and safe to deploy without a
flag.

## What is NOT recommended

- Do not silently coerce the `agent` field to the token subject. The mismatch
  must be a 403 so the client can correct its request.
- Do not gate the existing A2A read endpoints on membership yet. Nothing outside
  the four new service functions reads the membership store, and gating reads
  would change behaviour for existing channels. This is tracked separately.
