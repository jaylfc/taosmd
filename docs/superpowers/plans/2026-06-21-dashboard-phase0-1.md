# Dashboard Phase 0 + 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reskin the standalone dashboard to taOS's macOS dark/light themes (Phase 0) and add a Home/Overview tab on a new `GET /stats` endpoint (Phase 1).

**Architecture:** Phase 0 remaps the dashboard's existing semantic CSS tokens to taOS graphite values and adds a `data-scheme="light"` block plus a toggle, so component code barely changes. Phase 1 adds an `api.stats()` aggregator over the existing stores (archive, claims, agents, projects), a `service.stats` wrapper, a `GET /stats` handler, and a React Home tab with hand-rolled SVG charts.

**Tech Stack:** Python stdlib HTTP server + asyncio service loop; React 18 + Vite + TypeScript + Tailwind v3; hand-rolled SVG (no chart library, offline).

**Branch:** `feat/dashboard-cockpit-phase01` off master.

---

## Task 1: Phase 0 token remap + light scheme

**Files:**
- Modify: `dashboard/src/index.css` (the `:root` token block, lines ~10-58)

- [ ] **Step 1: Replace the dark token values** in `:root` with the taOS-aligned graphite values, and append a light-scheme block. Keep every existing token name. New values:

```css
:root {
  --bg: #171717;
  --surface: #1d1d1f;
  --surface-2: #232325;
  --border: rgba(255,255,255,0.10);
  --border-subtle: rgba(255,255,255,0.06);
  --ink: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.55);
  --muted-bright: rgba(255,255,255,0.70);
  --accent: #5b8def;
  --accent-dim: rgba(91,141,239,0.16);
  --accent-hover: #7aa5f5;
  --error: #ff5f57;
  --warning: #febc2e;
  --success: #28c840;
  --info: #5b8def;
  --sender-0-bg: #14202e; --sender-1-bg: #221426; --sender-2-bg: #112019;
  --sender-3-bg: #221b10; --sender-4-bg: #1c1422; --sender-5-bg: #0f1f22;
  --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  --radius-sm: 4px; --radius: 6px; --radius-lg: 8px; --radius-window: 10px;
  --shadow-card: 0 2px 8px rgba(0,0,0,0.2), 0 0 1px rgba(255,255,255,0.05);
  --t-fast: 150ms ease-out; --t-base: 200ms ease-out; --t-slow: 250ms ease-out;
  --sidebar-w: 220px;
}

:root[data-scheme="light"] {
  --bg: #f5f5f7;
  --surface: #ffffff;
  --surface-2: #f0f0f2;
  --border: rgba(0,0,0,0.12);
  --border-subtle: rgba(0,0,0,0.08);
  --ink: rgba(0,0,0,0.88);
  --muted: rgba(0,0,0,0.55);
  --muted-bright: rgba(0,0,0,0.68);
  --accent: #3b6fd4;
  --accent-dim: rgba(91,141,239,0.12);
  --accent-hover: #5b8def;
  --error: #d83a32; --warning: #c98a00; --success: #1e9e33; --info: #3b6fd4;
  --sender-0-bg: #eef3fb; --sender-1-bg: #f7eefb; --sender-2-bg: #eef7f1;
  --sender-3-bg: #fbf5ee; --sender-4-bg: #f5eefb; --sender-5-bg: #eef9fb;
  --shadow-card: 0 2px 8px rgba(0,0,0,0.08), 0 0 1px rgba(0,0,0,0.06);
}
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/src/index.css
git commit -m "feat(dashboard): taOS graphite tokens + light scheme"
```

## Task 2: Phase 0 theme module + toggle

**Files:**
- Create: `dashboard/src/theme.ts`
- Modify: `dashboard/src/App.tsx` (init on mount; add a toggle button in the header, after `<HealthChip>`)

- [ ] **Step 1: Create `dashboard/src/theme.ts`**

```ts
const KEY = "taosmd-scheme";
export type Scheme = "dark" | "light";

export function getScheme(): Scheme {
  const saved = localStorage.getItem(KEY);
  if (saved === "dark" || saved === "light") return saved;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function applyScheme(s: Scheme): void {
  document.documentElement.dataset.scheme = s;
}

export function setScheme(s: Scheme): void {
  localStorage.setItem(KEY, s);
  applyScheme(s);
}

export function initTheme(): Scheme {
  const s = getScheme();
  applyScheme(s);
  return s;
}
```

- [ ] **Step 2: Wire into `App.tsx`** — import `initTheme`/`setScheme`/`Scheme`, add `const [scheme, setSchemeState] = useState<Scheme>("dark");` and `useEffect(() => setSchemeState(initTheme()), []);`, and add a toggle button in the header `ml-auto` group before `HealthChip`:

```tsx
<button
  onClick={() => { const next = scheme === "dark" ? "light" : "dark"; setScheme(next); setSchemeState(next); }}
  className="rounded p-1.5 transition-colors duration-150"
  style={{ color: "var(--muted)" }}
  aria-label={scheme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  aria-pressed={scheme === "light"}
>
  {scheme === "dark" ? "☀" : "☽"}
</button>
```

- [ ] **Step 3: Build and verify** `npm run build` succeeds (tsc clean). Commit:

```bash
git add dashboard/src/theme.ts dashboard/src/App.tsx
git commit -m "feat(dashboard): dark/light theme toggle (taOS data-scheme)"
```

## Task 3: `archive.daily_counts` range query

**Files:**
- Modify: `taosmd/archive.py` (add after `daily_summary`)
- Test: `tests/test_archive.py`

- [ ] **Step 1: Write the failing test** in `tests/test_archive.py`:

```python
@pytest.mark.asyncio
async def test_daily_counts_returns_per_day_totals(tmp_path):
    from taosmd.archive import Archive
    arc = Archive(archive_dir=str(tmp_path / "a"))
    await arc.record(event_type="message", payload={"text": "x"}, agent="a")
    out = await arc.daily_counts(days=30)
    assert isinstance(out, list)
    assert sum(d["count"] for d in out) >= 1
    assert all(set(d) == {"date", "count"} for d in out)
```

- [ ] **Step 2: Run it, expect FAIL** (`AttributeError: daily_counts`). `python3 -m pytest tests/test_archive.py::test_daily_counts_returns_per_day_totals -q`

- [ ] **Step 3: Implement `daily_counts`** in `taosmd/archive.py`:

```python
async def daily_counts(self, days: int = 30) -> list[dict]:
    """Per-day archived-event counts for the last ``days`` days, oldest first."""
    cutoff = time.time() - days * 86400
    rows = self._conn.execute(
        """SELECT date(timestamp, 'unixepoch') AS d, COUNT(*) AS n
           FROM archive_index WHERE timestamp >= ?
           GROUP BY d ORDER BY d""",
        (cutoff,),
    ).fetchall()
    return [{"date": r["d"], "count": r["n"]} for r in rows]
```

- [ ] **Step 4: Run it, expect PASS.** Commit:

```bash
git add taosmd/archive.py tests/test_archive.py
git commit -m "feat(archive): daily_counts range query for the growth chart"
```

## Task 4: `api.stats` aggregator + `service.stats` wrapper

**Files:**
- Modify: `taosmd/api.py` (add `async def stats`), `taosmd/service.py` (add wrapper)
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test** in `tests/test_api.py`:

```python
def test_stats_shape(isolated_data_dir):
    import asyncio, taosmd
    asyncio.get_event_loop().run_until_complete(
        taosmd.api.ingest("Stats test memory.", agent="s", data_dir=isolated_data_dir))
    out = asyncio.get_event_loop().run_until_complete(
        taosmd.api.stats(data_dir=isolated_data_dir))
    assert set(out) >= {"memories", "agents", "projects", "growth", "verification", "top_agents", "recent_activity"}
    assert out["memories"]["total"] >= 1
    assert isinstance(out["growth"], list)
    assert set(out["verification"]) >= {"supported", "unverified", "flagged", "hallucination_rate"}
```

- [ ] **Step 2: Run it, expect FAIL** (`AttributeError: stats`).

- [ ] **Step 3: Implement `api.stats`** in `taosmd/api.py` (uses `_ensure_stores`, `list_projects`, `agents.list_agents`):

```python
async def stats(*, data_dir=None) -> dict:
    """Aggregate read-only dashboard stats over the existing stores."""
    from taosmd import agents as _agents  # noqa: PLC0415
    stores = await _ensure_stores(data_dir)
    arc = stores["archive"]
    arc_stats = await arc.stats()
    growth = await arc.daily_counts(days=30)
    recent = await arc.recent(limit=10)
    claims = stores.get("claims")
    rate = await claims.rate() if claims is not None else {}
    supported = int(rate.get("supported", 0))
    unverified = int(rate.get("unverified", 0))
    flagged = int(rate.get("partial", 0)) + int(rate.get("unsupported", 0)) + int(rate.get("contradicted", 0))
    projects = await list_projects(data_dir=data_dir)
    try:
        agent_rows = _agents.list_agents()
    except Exception:
        agent_rows = []
    top_agents = sorted(
        ({"name": a.get("name", "?"), "count": int(a.get("memory_count", a.get("facts", 0)) or 0)} for a in agent_rows),
        key=lambda x: -x["count"])[:5]
    top_projects = sorted(
        ({"name": p.get("project_id", "?"), "count": len(p.get("agents", []))} for p in projects),
        key=lambda x: -x["count"])[:5]
    return {
        "memories": {"total": int(arc_stats.get("total_events", 0)), "disk_mb": arc_stats.get("disk_usage_mb", 0)},
        "agents": len(agent_rows),
        "projects": len(projects),
        "growth": growth,
        "verification": {
            "supported": supported, "unverified": unverified, "flagged": flagged,
            "hallucination_rate": float(rate.get("hallucination_rate", 0.0)),
        },
        "top_agents": top_agents,
        "top_projects": top_projects,
        "recent_activity": recent,
    }
```

- [ ] **Step 4: Add `archive.recent`** in `taosmd/archive.py`:

```python
async def recent(self, limit: int = 10) -> list[dict]:
    """The most recent archived events as {kind, label, ts}, newest first."""
    rows = self._conn.execute(
        "SELECT event_type, timestamp, agent FROM archive_index ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"kind": r["event_type"], "label": f"{r['event_type']} by {r['agent'] or 'unknown'}",
             "ts": r["timestamp"]} for r in rows]
```

- [ ] **Step 5: Add `service.stats`** in `taosmd/service.py` (mirror `list_projects`):

```python
async def stats(*, data_dir=None) -> dict:
    """Aggregate dashboard stats. Forwarded to the remote server when configured."""
    remote = _get_remote(data_dir)
    if remote is not None:
        return await remote.stats()
    return await _api.stats(data_dir=data_dir)
```

Add a matching `async def stats(self, **_opts) -> dict` to `taosmd/remote.py` that GETs `/stats` (mirror `list_projects` there).

- [ ] **Step 6: Run the test, expect PASS.** Commit:

```bash
git add taosmd/api.py taosmd/service.py taosmd/remote.py taosmd/archive.py tests/test_api.py
git commit -m "feat(api): stats aggregator + service/remote wrappers"
```

## Task 5: `GET /stats` endpoint

**Files:**
- Modify: `taosmd/http_server.py` (add `_handle_stats` near `_handle_list_projects`; add dispatch after `/controls`)
- Test: `tests/test_http_server.py`

- [ ] **Step 1: Write the failing test** in `tests/test_http_server.py`:

```python
def test_stats_endpoint(live_server):
    _post(f"{live_server}/ingest", {"text": "A stats memory.", "agent": "s"})
    status, body = _get(f"{live_server}/stats")
    assert status == 200, body
    assert body["memories"]["total"] >= 1
    assert "verification" in body and "growth" in body
```

- [ ] **Step 2: Run it, expect FAIL** (404).

- [ ] **Step 3: Add the handler** after `_handle_list_projects`:

```python
def _handle_stats(self) -> None:
    result = runner.run(service.stats(data_dir=data_dir))
    self._send_json(200, result)
```

And the dispatch line after the `/controls` GET line:

```python
elif method == "GET" and path == "/stats":
    self._handle_stats()
```

- [ ] **Step 4: Run it, expect PASS.** Run the controls + http suite to confirm no regression. Commit:

```bash
git add taosmd/http_server.py tests/test_http_server.py
git commit -m "feat(http): GET /stats endpoint"
```

## Task 6: Frontend stats client + types

**Files:**
- Modify: `dashboard/src/types.ts` (add `Stats`), `dashboard/src/api.ts` (add `getStats`)

- [ ] **Step 1: Add the `Stats` type** to `types.ts`:

```ts
export interface Stats {
  memories: { total: number; disk_mb: number };
  agents: number;
  projects: number;
  growth: { date: string; count: number }[];
  verification: { supported: number; unverified: number; flagged: number; hallucination_rate: number };
  top_agents: { name: string; count: number }[];
  top_projects: { name: string; count: number }[];
  recent_activity: { kind: string; label: string; ts: number }[];
}
```

Add `"home"` to the `View` union (make it the first member).

- [ ] **Step 2: Add `getStats`** to `api.ts`:

```ts
export async function getStats(): Promise<Stats> {
  return req<Stats>("/stats");
}
```

(import `Stats` from `./types`).

- [ ] **Step 3: Build to typecheck.** Commit:

```bash
git add dashboard/src/types.ts dashboard/src/api.ts
git commit -m "feat(dashboard): stats client + Stats type"
```

## Task 7: SVG chart components

**Files:**
- Create: `dashboard/src/components/StatCard.tsx`, `GrowthChart.tsx`, `VerificationDonut.tsx`, `TopList.tsx`, `ActivityFeed.tsx`

Each is a small presentational component using the design tokens, hand-rolled SVG, ARIA-labelled. Key math:

- `VerificationDonut`: three arcs (supported = `--success`, unverified = `--muted`, flagged = `--warning`) over a circle of circumference `2*PI*r`. Each arc uses `stroke-dasharray="<seg> <rest>"` and a rotating `stroke-dashoffset` accumulator. Centre text shows the verified share `Math.round(100*supported/total)%`. `role="img"` + `aria-label`. Total 0 renders an empty grey ring.
- `GrowthChart`: a row of `<rect>` bars scaled to `max(count)`, width = container / N, with a 7/30-day toggle (slice the array). `role="img"` + `aria-label` summarising "N memories over D days".
- `StatCard`: a `--surface` card with `--shadow-card`, a big `--ink` number and a `--muted` label.
- `TopList`: horizontal `<rect>` bars scaled to the max count, labelled.
- `ActivityFeed`: a list, each row a small dot + label + relative time via `utils/time`.

- [ ] **Step 1: Write the components** (full code authored during execution, following the patterns in `MemoryView.tsx` and `SettingsView.tsx` for token usage and ARIA).

- [ ] **Step 2: Build to typecheck.** Commit:

```bash
git add dashboard/src/components/StatCard.tsx dashboard/src/components/GrowthChart.tsx dashboard/src/components/VerificationDonut.tsx dashboard/src/components/TopList.tsx dashboard/src/components/ActivityFeed.tsx
git commit -m "feat(dashboard): SVG stat/chart components"
```

## Task 8: Overview view + nav

**Files:**
- Create: `dashboard/src/views/OverviewView.tsx`
- Modify: `dashboard/src/App.tsx` (add `home` as first NAV item, default view `home`, render `OverviewView`)

- [ ] **Step 1: Write `OverviewView.tsx`** — loads `getStats()` (loading = SkeletonCard, error = ErrorBanner with retry, empty install = EmptyState), then lays out: four `StatCard`s (Total Memories, Agents, Projects, Verified facts), the `GrowthChart`, the `VerificationDonut` (hero, with legend + hallucination rate), `TopList` for top agents and projects, and `ActivityFeed`. Section layout mirrors `MemoryView`'s `section` container.

- [ ] **Step 2: Wire `App.tsx`** — add `{ id: "home", label: "Home" }` first in `NAV`, change initial `useState<View>("home")`, and render `{view === "home" && <OverviewView />}` first in `main`.

- [ ] **Step 3: Build, copy assets, verify with Playwright** (server on a temp data dir, navigate, screenshot both themes, confirm the four cards + the donut render). Commit:

```bash
git add dashboard/src/views/OverviewView.tsx dashboard/src/App.tsx taosmd/webui/
git commit -m "feat(dashboard): Home/Overview tab"
```

## Task 9: Full suite + PR

- [ ] **Step 1: Run the full Python suite** `python3 -m pytest -q` (expect all pass).
- [ ] **Step 2: Update STATUS.md** with the Phase 0+1 landing.
- [ ] **Step 3: Push the branch and open a PR** (review, not self-merge).

---

## Self-review notes

- Spec coverage: theming (Task 1-2), light scheme + toggle (Task 1-2), GET /stats over real sources (Task 3-5), Home tab + hand-rolled SVG charts + verification donut hero + top categories as agents/projects (Task 6-8), tests at each layer, offline/no-CDN (no chart lib), accessibility (ARIA on charts + toggle). Covered.
- The `archive.recent`, `daily_counts`, `api.stats`, `service.stats`, `remote.stats`, `getStats`, and `Stats` names are consistent across tasks.
- `agents.list_agents()` row shape (name, memory_count/facts) is read defensively in Task 4; the exact key is confirmed against `agents.py` during execution.
