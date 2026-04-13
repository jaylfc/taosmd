# Memory Management App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the taOS Memory app into a full management interface with dashboard, session browser, pipeline control, per-agent config, and global settings — backed by a pluggable backend abstraction so other memory systems can integrate.

**Architecture:** Backend interface (MemoryBackend base class) in taosmd package, taOSmd implementation, API routes in tinyagentos, React frontend with 5 tabs. Backend declares capabilities and settings schema; frontend renders dynamically. Two repos: taosmd (backend) and tinyagentos (frontend + API).

**Tech Stack:** Python (backend interface), FastAPI (API routes), React + TypeScript (frontend), Tailwind CSS (styling)

---

### Task 1: MemoryBackend Base Class

**Files:**
- Create: `taosmd/backend.py`
- Create: `tests/test_backend.py`

Create the abstract base class that all memory backends implement. Declares capabilities, stats, settings schema, and per-agent config methods. All methods are async with NotImplementedError defaults.

### Task 2: taOSmd Backend Implementation

**Files:**
- Create: `taosmd/taosmd_backend.py`
- Modify: `tests/test_backend.py` (add taosmd backend tests)

Implements MemoryBackend for taOSmd. Wires into all existing stores (KG, vector, archive, catalog, crystals). Returns full capabilities list. get_settings_schema() returns JSON Schema for taOSmd-specific settings (strategy, tiers, retention, secret filter mode, cron schedule). get_stats() aggregates from all stores.

### Task 3: Export Backend Classes

**Files:**
- Modify: `taosmd/__init__.py`

Export MemoryBackend and TaOSmdBackend. Run full test suite.

### Task 4: API Routes

**Files:**
- Modify: `tinyagentos/app.py`

Add routes: GET/PUT /api/memory/settings, GET /api/memory/stats, GET /api/memory/backend/capabilities, GET /api/memory/backend/settings-schema, GET/PUT /api/agents/{name}/memory-config. Wire to TaOSmdBackend instance in lifespan.

### Task 5: Memory API Client (TypeScript)

**Files:**
- Create: `desktop/src/lib/memory.ts`

Typed API client with functions for all memory endpoints: fetchMemoryStats(), fetchMemorySettings(), updateMemorySettings(), fetchBackendCapabilities(), fetchSettingsSchema(), fetchAgentMemoryConfig(), updateAgentMemoryConfig(), fetchCatalogSessions(), triggerCatalogIndex(), etc.

### Task 6: Memory Dashboard Component

**Files:**
- Create: `desktop/src/components/memory/Dashboard.tsx`

Health cards for KG, vector, archive, catalog, crystals. Pipeline status bar. Uses fetchMemoryStats().

### Task 7: Session Browser + Timeline

**Files:**
- Create: `desktop/src/components/memory/SessionBrowser.tsx`
- Create: `desktop/src/components/memory/SessionTimeline.tsx`
- Create: `desktop/src/components/memory/SessionDetail.tsx`

Calendar + feed toggle. Calendar view with mini month picker + timeline strip. Feed view with scrolling cards. Session detail panel with crystal narrative, sub-sessions, raw content.

### Task 8: Pipeline Control Component

**Files:**
- Create: `desktop/src/components/memory/PipelineControl.tsx`

Index yesterday button, date picker, tier selector, crystallization toggle, rebuild button with confirmation, pipeline history table.

### Task 9: Agent Memory Config Components

**Files:**
- Create: `desktop/src/components/memory/AgentMemoryTable.tsx`
- Create: `desktop/src/components/memory/AgentMemoryDetail.tsx`

Overview table of all agents with strategy/layers/status. Click-to-expand detail with backend selector, strategy dropdown, layer toggles, backend-specific settings.

### Task 10: Settings + Schema Form Renderer

**Files:**
- Create: `desktop/src/components/memory/MemorySettings.tsx`
- Create: `desktop/src/components/memory/SchemaFormRenderer.tsx`

SchemaFormRenderer takes a JSON Schema and current values, renders form controls dynamically (string→input, number→number, boolean→toggle, enum→dropdown). MemorySettings uses it for global config.

### Task 11: Upgrade MemoryApp with Tabs

**Files:**
- Modify: `desktop/src/apps/MemoryApp.tsx`

Replace current memory browser with tabbed layout. 5 tabs: Dashboard, Session Browser, Pipeline Control, Agent Config, Settings. Tab visibility based on backend capabilities. Register in app-registry if needed.

---

## Self-Review

**Spec coverage:** Backend abstraction → Tasks 1-3 ✓. API routes → Task 4 ✓. All 5 tabs → Tasks 6-11 ✓. Dynamic schema rendering → Task 10 ✓. Agent overview table + detail → Task 9 ✓. Calendar + feed session browser → Task 7 ✓.

**Placeholder scan:** Clean — all tasks have specific files and responsibilities.

**Type consistency:** MemoryBackend interface used consistently in Task 1 (define), Task 2 (implement), Task 4 (wire). API client in Task 5 consumed by Tasks 6-11.
