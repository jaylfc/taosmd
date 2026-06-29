# Dashboard generator-profile setter

Status: design approved (overnight autonomous; Jay pre-approved Approach A + scope-aware + auto-merge-if-green)
Date: 2026-06-30
Author: jaylfc

## Problem

The task-aware generator profiles feature (PR #177) made the answer generator
selectable by workload, but only via the CLI (`taosmd generator-profile set`).
The dashboard shows the active profile READ-ONLY, because consumer-scope controls
are not settable through the generic controls path (`config.set_control` rejects
any control whose scope is not `runtime`, and it writes to `controls.<id>` rather
than the top-level `generator_profile` key the resolver reads). So a user who
wants to switch their generator has to drop to the CLI. The dashboard is where
they would naturally reach for it.

The generator profile is safe to change live (unlike the embedder, a store-scope
control whose change forces a re-embed): switching it only changes which model
answers the next query, no re-indexing. So it is a good candidate for an
interactive dashboard control, via a dedicated path that writes to the correct
key rather than by loosening the generic controls invariant.

## Goals

- Let a user select the generator profile from the dashboard, not just the CLI.
- Honor the dashboard's existing scope selector: set the global default when the
  scope is global/all, set a specific agent's override when an agent is selected.
- Write to the SAME keys the resolver reads (top-level `generator_profile` for
  global, the agent record `generator_profile_id` for per-agent), so the
  selection actually takes effect.
- Do not weaken the read-only guarantee for genuinely-unsafe controls (embedder,
  binary_quant, etc.). This is a dedicated path for one safe control.

## Non-goals (YAGNI)

- No general "settable consumer control" capability. This is a contained,
  dedicated path for the generator profile only. If more consumer controls later
  want live settability, generalize then.
- No creating or editing profiles from the dashboard. Profiles are code-defined
  (the `generator_profiles` registry). The dashboard only selects among them.
- No remote or cloud generator backends. Separate deferred spec.

## Approach (chosen: dedicated setter endpoint + dashboard selector)

Two new HTTP endpoints write through the correct config/agents functions, and a
new dashboard component reads and sets the profile, honoring the active scope.
The existing `generator_profile` controls entry stays (it provides the schema
metadata); the dashboard replaces its read-only row with the interactive
selector.

## Backend: two endpoints in `taosmd/http_server.py`

Mirror the existing `_handle_*` handlers and the GET/POST routing table.

`GET /generator-profile` (optional `?agent=<name>`):
- Returns `{"profiles": [{"id", "label", "workload", "models"}], "active": "<id>", "scope": "global" | "<agent>"}`.
- `profiles` comes from `generator_profiles.list_profiles()`.
- When `agent` is provided and that agent has a `generator_profile_id`, `active`
  is that id and `scope` is the agent name. Otherwise `active` is the global
  `config.get_generator_profile()` or `generator_profiles.default_profile_id()`,
  and `scope` is `"global"`.

`POST /generator-profile` (JSON body `{"profile_id": "<id>", "agent": "<name>"?}`):
- Validate `profile_id` against `generator_profiles.get_profile`; unknown id ->
  HTTP 400 with `{"error": "unknown profile <id>"}`.
- If `agent` is present and non-empty -> `agents.set_agent_generator_profile(agent, profile_id)`;
  an unregistered agent raises `AgentNotFoundError` -> HTTP 404/400 with the error.
- Else -> `config.set_generator_profile(profile_id)`.
- Return the same shape as `GET` (the new active state), so the client can
  confirm without a second round-trip.

Both handlers reuse the server's existing JSON-response and request-body helpers
(whatever `_handle_controls_post` / `_handle_search_post` use). No change to
`config.set_control` or the controls scope guard.

Optional: add `a2a`-style `RemoteClient` methods (`get_generator_profile`,
`set_generator_profile`) for parity, only if other RemoteClient endpoints follow
that pattern and a test needs them. Not required for the dashboard, which calls
the HTTP endpoints directly. Treat as optional in the plan.

## Frontend: `dashboard/src`

- `src/api.ts`: add `getGeneratorProfile(scope?: string)` and
  `setGeneratorProfile(profileId: string, scope?: string)` using the existing
  fetch wrapper. When `scope` is a specific agent (not the global/all sentinel),
  pass it as the `agent` query/body field; when global/all, omit it.
- `src/types.ts`: add the `GeneratorProfile` and response types.
- `src/components/GeneratorProfileSelector.tsx` (new): a labelled `<select>` of
  profiles (by `label`), showing the active one, with the active profile's
  per-tier `models` map shown as a small read-only preview/caption. On change it
  optimistically updates, calls `setGeneratorProfile`, and on failure reverts and
  surfaces the error via the existing `ErrorBanner` pattern. Accessible: a real
  `<label>`/`aria-label`, keyboard operable (it is a native select).
- `src/views/SettingsView.tsx`: render `GeneratorProfileSelector`, passing the
  current scope from the existing scope state (the same value `ScopeSelector`
  drives). Replace the read-only `generator_profile` row with this selector (or
  place the selector in the consumer-controls area and drop the duplicate
  read-only row so the profile is shown once, interactively).
- Rebuild the bundle: `npm run build` in `dashboard/` (runs `tsc -noEmit` typecheck
  then `vite build`, output to `taosmd/webui`). Commit the regenerated
  `taosmd/webui/assets/*` and `taosmd/webui/index.html` together with the source.

## Data flow

Scope state (global/all or an agent) -> `GeneratorProfileSelector` calls
`GET /generator-profile?agent=...` to show the active profile for that scope ->
user picks a profile -> `POST /generator-profile` writes global or per-agent ->
`generator_profiles.resolve_generator` (per-agent over global) honors it on the
next answer. No re-index, no restart.

## Error handling

- Unknown profile id: backend 400; selector reverts to the prior value and shows
  the error. (Should not happen via the UI since options come from the server,
  but the endpoint is defensive.)
- Per-agent set for an unregistered agent: backend error; selector shows it.
- Network/5xx: selector reverts the optimistic change and shows the error banner.
- Backend writes go through the existing atomic config/agents writers, so a
  failed write does not partially persist.

## Testing

- Backend (pytest, mirror the existing http_server endpoint tests):
  - `GET /generator-profile` with no agent returns all profiles and `active` =
    the global default (`balanced`) on a fresh store, `scope` = `global`.
  - After `config.set_generator_profile("factual-recall")`, `GET` returns
    `active` = `factual-recall`.
  - `GET /generator-profile?agent=<a>` reflects a per-agent override set via
    `agents.set_agent_generator_profile`, with `scope` = the agent.
  - `POST /generator-profile {"profile_id": "factual-recall"}` persists globally
    (assert `config.get_generator_profile` == `factual-recall`) and returns the
    new active state.
  - `POST` with `agent` persists per-agent (assert
    `agents.get_agent_generator_profile`).
  - `POST` with an unknown `profile_id` returns 400 and does not change state.
- Frontend: the dashboard has no test runner (no vitest). Coverage is the
  `tsc -noEmit` typecheck in the build plus a manual build-success gate. The plan
  records this gap honestly: the backend endpoints carry the behavioral tests;
  the frontend is type-checked and build-verified, not unit-tested.
- Build gate: `npm run build` must exit 0 (typecheck passes) and regenerate the
  bundle; the committed `taosmd/webui` artifacts must match the rebuilt output
  (no stale bundle).

## Decision log

- Dedicated setter endpoint + selector (Approach A), over generalizing the
  controls system with a `settable` capability (would have to reconcile the
  controls-store vs top-level-key write path and weakens the read-only guarantee)
  and over reclassifying the control as `runtime` scope (would write to the wrong
  key and mislabel a consumer choice).
- Scope-aware: the selector honors the dashboard scope (global vs per-agent),
  matching the CLI `--agent` capability and the resolver's per-agent precedence.
