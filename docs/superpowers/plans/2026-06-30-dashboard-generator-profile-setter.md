# Dashboard Generator-Profile Setter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a user select the generator profile from the dashboard (global or per-agent), via a dedicated endpoint + selector that write to the keys the resolver reads.

**Architecture:** Two new HTTP endpoints (`GET`/`POST /generator-profile`) in `taosmd/http_server.py` write through `config.set_generator_profile` (global) and `agents.set_agent_generator_profile` (per-agent), leaving the controls scope guard untouched. A new React `GeneratorProfileSelector` (scope-aware, reusing the `stats.top_agents` -> `ScopeSelector` pattern) reads/sets the profile and is rendered in `SettingsView`. The webui bundle is rebuilt.

**Tech Stack:** Python stdlib http.server (the existing nested-handler style), pytest; React + TypeScript + Vite (the `dashboard/` app, built to `taosmd/webui`).

## Global Constraints

- The generator profile feature already exists: `taosmd/generator_profiles.py` (`list_profiles()`, `get_profile(id)`, `default_profile_id()` -> `"balanced"`, each profile has `id`, `label`, `workload`, `models`), `config.get_generator_profile`/`set_generator_profile` (top-level key), `agents.get_agent_generator_profile`/`set_agent_generator_profile` (agent record key `generator_profile_id`).
- Do NOT change `config.set_control` or the controls scope guard. This is a dedicated path.
- The global/all scope sentinel in the frontend is the string `"all"` (matches `getStats`). When scope is `"all"`, set the global profile (no agent); otherwise set per-agent.
- New HTTP handlers must use the existing `self._read_json_body()` and `self._send_json(status, payload)` helpers and capture `data_dir` from the enclosing scope, exactly like `_handle_controls_post`.
- No em dashes in any added Python, TypeScript, or docs. No AI-attribution in commits. Author jaylfc.
- The committed `taosmd/webui` bundle must be regenerated from source via `npm run build` (which runs `tsc -noEmit` then `vite build`); never hand-edit the bundle.
- Run Python tests with `python3 -m pytest <path> -q` (no `.venv`). Run the frontend build with `npm run build` from `dashboard/`.

---

### Task 1: Backend GET and POST /generator-profile

**Files:**
- Modify: `taosmd/http_server.py` (add two nested handlers near `_handle_controls_post`; add two routes in the GET and POST dispatch tables)
- Test: `tests/test_http_generator_profile.py`

**Interfaces:**
- Consumes: `generator_profiles.list_profiles`, `get_profile`, `default_profile_id`; `config.get_generator_profile`, `set_generator_profile`; `agents.get_agent_generator_profile`, `set_agent_generator_profile`; the server's `_read_json_body`, `_send_json`, and the `query` dict passed to GET handlers.
- Produces: `GET /generator-profile?agent=` returning `{"profiles": [{"id","label","workload","models"}], "active": "<id>", "scope": "global"|"<agent>"}`; `POST /generator-profile` with body `{"profile_id", "agent"?}` returning the same shape, or 400 on unknown profile / agent error.

- [ ] **Step 1: Write the failing test**

Mirror the existing http_server endpoint tests (find one that constructs the handler/server with a tmp `data_dir` and calls an endpoint; reuse that harness). The test must cover: GET lists profiles and active=balanced on a fresh store (scope "global"); GET reflects a global set; GET with agent reflects a per-agent override; POST sets global and persists; POST with agent sets per-agent and persists; POST unknown profile_id returns 400 and does not change state. Concretely (adapt the client/harness to the existing http_server tests' style):

```python
# tests/test_http_generator_profile.py
import json
from taosmd import config, agents
# import the same server/test harness the other http_server tests use; if they
# use a helper like `make_test_server(tmp_path)` or call handler methods through
# a fake request, reuse it verbatim. The assertions below are harness-agnostic:

def test_get_lists_profiles_and_global_active(http_client, tmp_path):
    resp = http_client.get("/generator-profile")
    body = resp.json()
    ids = {p["id"] for p in body["profiles"]}
    assert {"balanced", "factual-recall"} <= ids
    assert body["active"] == "balanced"
    assert body["scope"] == "global"

def test_get_reflects_global_set(http_client, tmp_path):
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    assert http_client.get("/generator-profile").json()["active"] == "factual-recall"

def test_get_reflects_per_agent(http_client, tmp_path):
    agents.AgentRegistry(tmp_path).register_agent("alice")
    agents.set_agent_generator_profile("alice", "factual-recall", data_dir=tmp_path)
    body = http_client.get("/generator-profile?agent=alice").json()
    assert body["active"] == "factual-recall"
    assert body["scope"] == "alice"

def test_post_sets_global(http_client, tmp_path):
    resp = http_client.post("/generator-profile", {"profile_id": "factual-recall"})
    assert resp.status == 200
    assert config.get_generator_profile(data_dir=tmp_path) == "factual-recall"
    assert resp.json()["active"] == "factual-recall"

def test_post_sets_per_agent(http_client, tmp_path):
    agents.AgentRegistry(tmp_path).register_agent("bob")
    http_client.post("/generator-profile", {"profile_id": "factual-recall", "agent": "bob"})
    assert agents.get_agent_generator_profile("bob", data_dir=tmp_path) == "factual-recall"

def test_post_unknown_profile_400(http_client, tmp_path):
    resp = http_client.post("/generator-profile", {"profile_id": "nope"})
    assert resp.status == 400
    assert config.get_generator_profile(data_dir=tmp_path) is None
```

IMPORTANT: the `http_client` fixture above is a stand-in. Before writing this, open the existing http_server test module (grep `tests/` for `_handle_controls_post`, `/controls`, or how a server is instantiated with `data_dir`) and use that exact mechanism to issue GET/POST and read status + JSON. Keep the assertions; adapt the transport.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_http_generator_profile.py -q`
Expected: FAIL (routes/handlers not defined -> 404 or AttributeError)

- [ ] **Step 3: Write the handlers + routes**

Add two nested handler methods next to `_handle_controls_post` (same closure level, capturing `data_dir`):

```python
def _handle_generator_profile_get(self, query) -> None:
    from taosmd import generator_profiles as _gp, config as _config, agents as _agents  # noqa: PLC0415
    raw = query.get("agent")
    agent = raw[0] if isinstance(raw, list) else raw
    profiles = [
        {"id": p.id, "label": p.label, "workload": p.workload, "models": p.models}
        for p in _gp.list_profiles()
    ]
    active, scope = None, "global"
    if agent:
        pid = _agents.get_agent_generator_profile(agent, data_dir=data_dir)
        if pid:
            active, scope = pid, agent
    if active is None:
        active = _config.get_generator_profile(data_dir=data_dir) or _gp.default_profile_id()
    self._send_json(200, {"profiles": profiles, "active": active, "scope": scope})

def _handle_generator_profile_post(self) -> None:
    from taosmd import generator_profiles as _gp, config as _config, agents as _agents  # noqa: PLC0415
    body = self._read_json_body()
    pid = body.get("profile_id")
    agent = body.get("agent")
    if not pid or _gp.get_profile(pid) is None:
        self._send_json(400, {"error": f"unknown profile: {pid!r}"})
        return
    try:
        if agent:
            _agents.set_agent_generator_profile(agent, pid, data_dir=data_dir)
        else:
            _config.set_generator_profile(pid, data_dir=data_dir)
    except Exception as exc:  # e.g. AgentNotFoundError
        self._send_json(400, {"error": str(exc)})
        return
    # Return the new active state (reuse the GET shape).
    profiles = [
        {"id": p.id, "label": p.label, "workload": p.workload, "models": p.models}
        for p in _gp.list_profiles()
    ]
    if agent:
        active, scope = pid, agent
    else:
        active, scope = pid, "global"
    self._send_json(200, {"profiles": profiles, "active": active, "scope": scope})
```

Match how `query` is read in a neighbouring GET handler (e.g. `_handle_stats(query)`); if `query` is a plain dict of strings rather than lists, drop the `isinstance(raw, list)` branch to match.

Add routes in the dispatch tables, next to the `/controls` routes:

```python
elif method == "GET" and path == "/generator-profile":
    self._handle_generator_profile_get(query)
elif method == "POST" and path == "/generator-profile":
    self._handle_generator_profile_post()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_http_generator_profile.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the http_server suite for no regressions**

Run: `python3 -m pytest tests/ -q -k "http or controls or server"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add taosmd/http_server.py tests/test_http_generator_profile.py
git commit -m "feat(dashboard): GET/POST /generator-profile endpoints (global + per-agent)"
```

---

### Task 2: Dashboard selector + bundle rebuild

**Files:**
- Modify: `dashboard/src/api.ts` (add `getGeneratorProfile`, `setGeneratorProfile`)
- Modify: `dashboard/src/types.ts` (add `GeneratorProfile` and the response type)
- Create: `dashboard/src/components/GeneratorProfileSelector.tsx`
- Modify: `dashboard/src/views/SettingsView.tsx` (render the selector)
- Regenerate: `taosmd/webui/index.html`, `taosmd/webui/assets/*` (via `npm run build`)

**Interfaces:**
- Consumes: the Task 1 endpoints; the existing `req<T>` fetch wrapper, `getStats` (for the agent list via `stats.top_agents`), `ScopeSelector`, `ErrorBanner`.
- Produces: a self-contained `GeneratorProfileSelector` React component rendered in `SettingsView`.

- [ ] **Step 1: Add the API functions + types**

In `dashboard/src/types.ts`:

```ts
export interface GeneratorProfile {
  id: string;
  label: string;
  workload: string;
  models: Record<string, string>;
}
export interface GeneratorProfileState {
  profiles: GeneratorProfile[];
  active: string;
  scope: string;
}
```

In `dashboard/src/api.ts` (mirror `getStats`'s `scope !== "all"` handling):

```ts
export async function getGeneratorProfile(scope?: string): Promise<GeneratorProfileState> {
  const q = scope && scope !== "all" ? `?agent=${encodeURIComponent(scope)}` : "";
  return req<GeneratorProfileState>(`/generator-profile${q}`);
}

export async function setGeneratorProfile(
  profileId: string,
  scope?: string,
): Promise<GeneratorProfileState> {
  const body: { profile_id: string; agent?: string } = { profile_id: profileId };
  if (scope && scope !== "all") body.agent = scope;
  return req<GeneratorProfileState>("/generator-profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
```

Add `GeneratorProfile`, `GeneratorProfileState` to the `types` import in `api.ts`.

- [ ] **Step 2: Build the selector component**

Create `dashboard/src/components/GeneratorProfileSelector.tsx`. Requirements (match the styling conventions in `OverviewView.tsx` and `SettingsView.tsx` -- read them for the exact class names and CSS variables; do not invent a new visual language):
- Own scope state, default `"all"`. Derive scope options the same way `OverviewView` does: call `getStats()` once, build `[{value:"all", label:"All agents"}, ...top_agents.map(a => ({value:a.name, label:a.name}))]`, and render a `ScopeSelector` with them.
- On mount and whenever scope changes, call `getGeneratorProfile(scope)` and store the returned state (profiles + active).
- Render a labelled `<select>` of `profiles` (option value = `id`, text = `label`), value = `active`. Include an `aria-label` and a visible `<label>`.
- Below it, show the active profile's per-tier `models` as a small read-only caption (for each tier: `tier: model || "retrieval-only"`).
- On change: optimistically set `active`, call `setGeneratorProfile(id, scope)`; on success store the returned state; on failure revert `active` to the prior value and show the message via `ErrorBanner` (same usage as `SettingsView`).
- Accessible and keyboard-operable (native select + label).

The component takes no props (self-contained). It must type-check under `tsc -noEmit` (no `any` leaks that break the build; follow the existing components' typing).

- [ ] **Step 3: Render it in SettingsView**

In `dashboard/src/views/SettingsView.tsx`, import and render `<GeneratorProfileSelector />` in a clearly-titled section (for example "Generator profile") near the consumer controls. If the read-only `generator_profile` control row is rendered separately by the controls loop, leave the controls loop as-is (it shows all controls); the interactive selector is an addition, so the profile appears once interactively and the generic row remains a read-only reference. Do not remove or special-case the controls loop.

- [ ] **Step 4: Type-check + build the bundle**

Run (from `dashboard/`): `npm run build`
Expected: exit 0. `tsc -noEmit` passes (this is the frontend's correctness gate, since there is no component test runner) and `vite build` writes the new bundle to `../taosmd/webui`. If `tsc` reports type errors, fix them before proceeding.

- [ ] **Step 5: Confirm the bundle changed and the Python suite is still green**

Run: `git status --porcelain taosmd/webui/` (expect the bundle js/css/index.html to show as modified or renamed-by-hash)
Run: `python3 -m pytest tests/ -q` (expect all pass; the backend endpoints from Task 1 plus everything else)

- [ ] **Step 6: Commit source + bundle together**

```bash
git add dashboard/src/api.ts dashboard/src/types.ts dashboard/src/components/GeneratorProfileSelector.tsx dashboard/src/views/SettingsView.tsx taosmd/webui/
git commit -m "feat(dashboard): scope-aware generator-profile selector + rebuilt bundle"
```

---

### Task 3: Docs + manual verification

**Files:**
- Modify: `README.md` (update the Generator profiles wording: it is now settable from the dashboard, not only the CLI)

- [ ] **Step 1: Update the README**

The prior feature's README said the dashboard displays the profile read-only and the CLI is the sole setter. Update the `### Generator profiles` subsection so it states the profile can now be set from the dashboard Settings view (global or per agent via the scope selector) as well as the CLI. Keep it accurate and em-dash-free. Do not re-introduce any wrong tier claim (balanced is qwen3.5:9b at 12 and 8 GB, llama3.1:8b at 4 GB).

- [ ] **Step 2: Manual end-to-end smoke (optional but recorded)**

If a server can be started in the environment, start it and confirm `GET /generator-profile` returns JSON and the dashboard Settings view renders the selector. If the environment cannot run the server interactively, record that the gate was the passing endpoint tests + the successful `tsc`/`vite` build, and skip the live smoke.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(dashboard): generator profile is now settable from the dashboard"
```
