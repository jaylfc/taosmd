# Dashboard Phase 0 + 1: taOS theming and Home/Overview (design spec)

Scope: the first two phases of the memory-cockpit roadmap (see `2026-06-21-memory-cockpit-vision.md`). Phase 0 reskins the standalone dashboard to taOS's macOS-like dark and light themes. Phase 1 adds a Home/Overview tab on a new aggregate stats endpoint. Both stay offline and self-contained (no runtime CDN), accessible, and respect managed mode.

## Phase 0: theming

### Approach

The dashboard already references colour through semantic CSS variables (`--bg`, `--surface`, `--ink`, `--accent`, ...) set in `dashboard/src/index.css` and consumed via inline styles in the components. So we remap the token values to taOS's graphite palette and add a light scheme, rather than rewrite components. taOS's source of truth is `tinyagentos/desktop/src/theme/tokens.css` (macOS graphite: `--color-shell-bg #1d1d1f`, neutral surfaces as white-overlays, a restrained slate accent, traffic-light state colours, soft card shadows, 10px radius) plus a `data-scheme="light"` inversion.

### Files

- Modify `dashboard/src/index.css`: replace the dark token values with the taOS-aligned values below, and add a `:root[data-scheme="light"]` block with the light values. Add `--radius-window: 10px` and `--shadow-card` from taOS.
- Create `dashboard/src/theme.ts`: a tiny module that reads the saved scheme (localStorage key `taosmd-scheme`) or falls back to `prefers-color-scheme`, applies it as `document.documentElement.dataset.scheme`, and exposes `getScheme()` / `setScheme(s)`.
- Modify `dashboard/src/App.tsx`: call the theme init on mount, and add a dark/light toggle button to the top bar (a sun/moon glyph, `aria-pressed`, keyboard reachable).

### Token mapping (current -> taOS-aligned)

| Token | Dark (data-scheme default) | Light (`data-scheme="light"`) |
|---|---|---|
| `--bg` | `#171717` | `#f5f5f7` |
| `--surface` | `#1d1d1f` | `#ffffff` |
| `--surface-2` | `#232325` | `#f0f0f2` |
| `--border` | `rgba(255,255,255,0.10)` | `rgba(0,0,0,0.12)` |
| `--border-subtle` | `rgba(255,255,255,0.06)` | `rgba(0,0,0,0.08)` |
| `--ink` | `rgba(255,255,255,0.92)` | `rgba(0,0,0,0.88)` |
| `--muted` | `rgba(255,255,255,0.55)` | `rgba(0,0,0,0.55)` |
| `--muted-bright` | `rgba(255,255,255,0.70)` | `rgba(0,0,0,0.68)` |
| `--accent` | `#5b8def` | `#3b6fd4` |
| `--accent-dim` | `rgba(91,141,239,0.16)` | `rgba(91,141,239,0.12)` |
| `--accent-hover` | `#7aa5f5` | `#5b8def` |
| `--success` | `#28c840` | `#1e9e33` |
| `--warning` | `#febc2e` | `#c98a00` |
| `--error` | `#ff5f57` | `#d83a32` |
| `--info` | `#5b8def` | `#3b6fd4` |

The slate accent `#5b8def` is taOS's `--color-unread`, its sanctioned non-neutral accent (never purple). The state colours are taOS's traffic-light values; the Overview's charts reuse them so data colour and chrome colour come from one palette. The sender-tint and skeleton tokens keep their roles, retuned to graphite.

### macOS cues

- Card and panel radius moves to the `--radius-window` 10px feel; keep the existing smaller radii for chips and controls.
- Panels and cards use `--shadow-card` (a soft drop shadow plus a 1px hairline) instead of only a border.
- The top bar becomes a frosted graphite bar (semi-transparent `--surface` with a backdrop blur), matching taOS chrome.

### Theme persistence and default

`theme.ts` applies the saved scheme on load; with nothing saved it follows `prefers-color-scheme`. The toggle writes the choice to localStorage (a real served app, so localStorage is fine here; this is not a claude.ai artifact). The existing `prefers-reduced-motion` handling in `index.css` is preserved.

### Testing (Phase 0)

- A Playwright check that toggling switches `document.documentElement.dataset.scheme` and that a known surface computes a light background in light mode and a dark one in dark mode.
- Visual confirmation via a screenshot in both schemes.

## Phase 1: Home/Overview

### Backend: `GET /stats`

A new read-only endpoint that aggregates data already computed by the stores. Add `_handle_stats` to `taosmd/http_server.py` and a `service.stats(data_dir)` helper that gathers:

- `memories`: `archive.stats()["total_events"]`, plus `disk_usage_mb`.
- `agents`: count from the agents/shelves registry; `projects`: `len(service.list_projects())`.
- `assistants`: distinct agents seen by the access tracker (`access_tracker.stats()` gives totals; distinct assistants come from the access log), with `queries` = `total_accesses`.
- `growth`: a list of `{date, count}` for the last 30 days. Add `archive.daily_counts(days=30)` doing one `GROUP BY date(timestamp)` query rather than 30 `daily_summary` calls.
- `verification`: from the claims store `rate()`, which returns counts per status (`unverified`, `supported`, `partial`, `unsupported`, `contradicted`) and `hallucination_rate`. The donut maps these to verified (`supported`), unverified (`unverified`), and flagged (`partial` + `unsupported` + `contradicted`).
- `top_agents` and `top_projects`: top five by memory count (a `GROUP BY agent` / per-project count).
- `recent_activity`: the last ten events across recent ingests and access-tracker queries, each `{kind, label, ts}`.

Response shape:

```json
{
  "memories": {"total": 2066, "disk_mb": 14.2},
  "agents": 4, "projects": 3,
  "assistants": {"connected": 2, "queries": 227},
  "growth": [{"date": "2026-05-23", "count": 31}, "..."],
  "verification": {"supported": 120, "unverified": 1700, "flagged": 36, "hallucination_rate": 0.188},
  "top_agents": [{"name": "default", "count": 600}, "..."],
  "top_projects": [{"name": "taosmd", "count": 420}, "..."],
  "recent_activity": [{"kind": "query", "label": "Claude queried your data", "ts": 1718900000}, "..."]
}
```

Fields with no data yet (a fresh install) return zeros or empty lists, never an error. Sources and documents counts are omitted in Phase 1 and arrive with the share-to-collect and documents phases.

### Frontend: the Home tab

- Create `dashboard/src/views/OverviewView.tsx`, and add a `home` entry as the first item in `NAV` in `App.tsx` (default view becomes `home`).
- Add `getStats()` to `dashboard/src/api.ts` and the `Stats` type to `types.ts`.
- Components (new, under `dashboard/src/components/`), all hand-rolled SVG so nothing loads from a CDN:
  - `StatCard`: label, big number, optional sublabel. Four across the top (Total Memories, Agents, Projects, Assistant queries).
  - `GrowthChart`: an SVG bar chart of `growth`, with a 7-day / 30-day toggle, `role="img"` and an `aria-label` summarising the trend.
  - `VerificationDonut`: an SVG donut of verified / unverified / flagged, centre label showing the verified share. This is the hero: the number no competitor surfaces. Legend with the three counts and the hallucination rate.
  - `TopList`: horizontal bars for `top_agents` and `top_projects` (the "top categories" stand-in until source-typed collections exist).
  - `ActivityFeed`: the `recent_activity` list with a small icon, label, and relative time (reuse `utils/time`).
- Loading uses the existing `SkeletonCard`; errors use `ErrorBanner` with retry; an empty install shows a friendly `EmptyState` ("No memories yet").

### Testing (Phase 1)

- Backend: a `test_http_server.py` test that seeds a few archived events and a claim, calls `GET /stats`, and asserts the shape and that counts reflect the seeded data; plus an empty-install test that asserts zeros and no error.
- A `service.stats` unit test for the aggregation, and an `archive.daily_counts` test for the range query.
- Frontend: the build typechecks, and a Playwright smoke test that the Home tab renders the four stat cards and the donut against a live server.

## Constraints (both phases)

- Offline and self-contained: charts are hand-rolled SVG; no chart library is loaded at runtime. Any new build dependency is bundled by Vite.
- Accessible: ARIA roles and labels on the charts, keyboard-reachable toggle and tabs, reduced-motion honoured.
- Managed mode: `GET /stats` is a read endpoint and stays available; it does not depend on the dashboard write surfaces.
- After the dashboard build, the Vite output is copied into `taosmd/webui/` and committed, as today.

## Out of scope (later phases)

Memory Explorer (galaxy), Sources and per-assistant permissions, Documents, share-to-collect, and ambient capture. The Overview links to those tabs once they exist; for now it shows what taOSmd already knows.
