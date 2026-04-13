# taOSmd Management App (Memory App Upgrade)

## Goal

Upgrade the existing taOS Memory app into a full management interface for the memory system. Dashboard, session browser (calendar + feed), pipeline control, per-agent memory config with overview table, global settings, and a backend abstraction layer so non-taOSmd backends can plug in with their own capabilities and settings.

## Architecture

The Memory app in the taOS desktop gets 5 tabs. Which tabs are visible depends on the active memory backend's declared capabilities. taOSmd (default) shows all 5. A simpler backend shows fewer.

### Backend Abstraction

```python
class MemoryBackend:
    name: str                      # "taosmd", "mem0", "qmd", etc.
    capabilities: list[str]        # ["kg", "vector", "archive", "catalog", "crystals", "pipeline"]
    
    async def get_stats() -> dict
    async def get_settings_schema() -> dict    # JSON Schema for config fields
    async def get_settings() -> dict           # current values
    async def update_settings(settings) -> dict
    async def get_agent_config(agent_name) -> dict
    async def update_agent_config(agent_name, config) -> dict
```

Any backend implements this interface. The app renders the settings UI dynamically from `get_settings_schema()` — no hardcoded taOSmd fields in the frontend. Third-party frameworks define their own schema and capabilities, the app adapts.

Backend is selected globally (system default for new agents) and per-agent:
```yaml
# Agent YAML
memory:
  backend: taosmd
  strategy: thorough
  layers:
    - vector
    - kg
    - catalog
```

### Tab Visibility by Capability

| Tab | Required capability | Always visible |
|-----|-------------------|---------------|
| Dashboard | — | Yes |
| Session Browser | `catalog` | No |
| Pipeline Control | `pipeline` | No |
| Agent Config | — | Yes |
| Settings | — | Yes |

## Tab Details

### Tab 1: Dashboard

Memory health at a glance.

**Cards:**
- KG: entity count, triple count, active triples
- Vector: document count, embedding model, dimension
- Archive: total events, disk usage, oldest/newest event
- Catalog: sessions cataloged, days indexed, last pipeline run
- Crystals: total crystals, crystals with lessons

**Pipeline status bar:** last run date, result (success/partial/error), next scheduled run, current processing tier.

**API:** `GET /api/memory/stats` (aggregates all store stats)

### Tab 2: Session Browser

Two views, toggled:

**Calendar view:**
- Mini month calendar (left sidebar)
- Click a date → right panel shows timeline strip: horizontal blocks for each session
- Each block: coloured by category, shows time range + topic
- Click a block → expand to detail panel

**Feed view:**
- Reverse-chronological scrolling list
- Each card: date, time range, topic, category badge, turn count, duration
- Click to expand

**Expanded session detail:**
- Full description (from enricher)
- Sub-sessions list
- Crystal narrative (if available)
- "View raw" button → shows archive lines from split file
- Linked KG facts

**API:** `GET /api/memory/catalog/date/{date}`, `GET /api/memory/catalog/session/{id}`, `GET /api/memory/catalog/session/{id}/context`, `GET /api/memory/catalog/search`

### Tab 3: Pipeline Control

**Actions:**
- "Index yesterday" button (one-click, most common action)
- Date picker → "Index this date" / "Index range"
- Tier selector: Auto (default) / Force Tier 1 / Force Tier 2 / Force Tier 3
- Crystallization toggle: on/off
- "Rebuild all" button (with confirmation dialog)

**Pipeline history:**
- Table: date, sessions found, tier used, time taken, status
- Scrollable, most recent first

**API:** `POST /api/memory/catalog/index`, `POST /api/memory/catalog/rebuild`, `GET /api/memory/catalog/stats`

### Tab 4: Agent Config

**Overview table (top):**

| Agent | Backend | Strategy | Active Layers | Status |
|-------|---------|----------|--------------|--------|
| research-agent | taOSmd | thorough | vector, kg, catalog, archive, crystals | Active |
| dev-agent | taOSmd | fast | vector, kg | Active |

Click a row → inline expand or navigate to detail.

**Detail panel (bottom or modal):**
- Backend selector dropdown (taOSmd, mem0, qmd, custom)
- Strategy dropdown (thorough / fast / minimal / custom)
- Layer toggles (checkboxes for each available layer)
- Backend-specific settings rendered from `get_settings_schema()`
- Enable/disable: crystal processing, session cataloging
- Retention settings: tier thresholds, TTL defaults, auto-eviction toggle

**API:** `GET /api/agents`, per-agent `GET /api/agents/{name}/memory-config`, `PUT /api/agents/{name}/memory-config`

### Tab 5: Settings

**Global defaults (rendered from backend schema):**

For taOSmd:
- Default strategy for new agents: dropdown
- Processing schedule: cron expression or simple "daily at" time picker
- Model selection per tier: dropdowns populated from Ollama models
- Archive compression: days before gzip (number input)
- Secret filter mode: redact / reject / warn (radio)
- Retention thresholds: hot/warm/cold sliders
- Backend selector: which backend new agents get by default

For other backends: whatever fields their `get_settings_schema()` declares, rendered as form controls based on JSON Schema types (string → text input, number → number input, boolean → toggle, enum → dropdown).

**API:** `GET /api/memory/settings`, `PUT /api/memory/settings`

## Frontend Components

All React + TypeScript, following existing taOS patterns.

| Component | File | Responsibility |
|-----------|------|---------------|
| MemoryApp | `apps/MemoryApp.tsx` | Tab container, backend capability detection |
| MemoryDashboard | `components/memory/Dashboard.tsx` | Health cards, pipeline status |
| SessionBrowser | `components/memory/SessionBrowser.tsx` | Calendar + feed toggle, session list |
| SessionTimeline | `components/memory/SessionTimeline.tsx` | Calendar view with timeline strip |
| SessionDetail | `components/memory/SessionDetail.tsx` | Expanded session with crystal + raw view |
| PipelineControl | `components/memory/PipelineControl.tsx` | Index buttons, tier selector, history |
| AgentMemoryTable | `components/memory/AgentMemoryTable.tsx` | Overview table with inline expand |
| AgentMemoryDetail | `components/memory/AgentMemoryDetail.tsx` | Per-agent config form |
| MemorySettings | `components/memory/MemorySettings.tsx` | Global settings, dynamic schema renderer |
| SchemaFormRenderer | `components/memory/SchemaFormRenderer.tsx` | Renders JSON Schema as form controls |

## API Routes (Backend)

New routes to add to `tinyagentos/app.py`:

| Route | Method | Handler |
|-------|--------|---------|
| `/api/memory/stats` | GET | Aggregate stats from all stores |
| `/api/memory/settings` | GET/PUT | Global memory settings |
| `/api/memory/backend/capabilities` | GET | Backend name + capabilities list |
| `/api/memory/backend/settings-schema` | GET | JSON Schema for backend settings |
| `/api/agents/{name}/memory-config` | GET/PUT | Per-agent memory config |

Catalog, KG, archive, and retrieval routes already exist from specs 1-2.

## Backend Interface (Python)

New file: `taosmd/backend.py`

```python
class MemoryBackend:
    """Base class for memory backends. taOSmd implements this. 
    Third-party backends subclass it."""
    
    name: str = "unknown"
    capabilities: list[str] = []
    
    async def get_stats(self) -> dict: ...
    async def get_settings_schema(self) -> dict: ...
    async def get_settings(self) -> dict: ...
    async def update_settings(self, settings: dict) -> dict: ...
    async def get_agent_config(self, agent_name: str) -> dict: ...
    async def update_agent_config(self, agent_name: str, config: dict) -> dict: ...
```

`taosmd/taosmd_backend.py` implements this with all capabilities.

## Out of Scope

- Benchmark results display in the app (nice-to-have, not core)
- Memory export/import wizard (separate feature)
- Real-time event streaming (websocket) — the app polls on tab switch
