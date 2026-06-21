# Dashboard Phase 2: Memory Explorer (knowledge-graph galaxy)

A new Explorer tab that visualises the temporal knowledge graph as a D3 force-directed "galaxy": entities are nodes, triples are edges. Inspired by ArcRift's force graph and the real-time-memory-visualisation patterns surveyed earlier. Offline and self-contained (D3 bundled by Vite, no CDN), accessible.

## Data (already in the store)

`taosmd/knowledge_graph.py` `TemporalKnowledgeGraph`:
- `kg_entities`: `id, name, type, properties_json, created_at`.
- `kg_triples`: `id, subject_id, predicate, object_id, valid_from, valid_to, confidence, source, superseded_by, appeared_count, accessed_count, last_accessed_at, created_at`.

So nodes = entities (sized by degree, coloured by `type`), edges = triples (coloured by `predicate`, faded when superseded or expired). `valid_to is not null` or `superseded_by is not null` means the fact is no longer active, which gives the temporal "what changed" read for free.

## Phase 2a: the static galaxy (this phase)

### Backend
- Add `kg.graph(limit=300)` returning `{ "nodes": [{id, name, type, degree}], "edges": [{source, target, predicate, confidence, active}] }`. `degree` is the count of triples touching an entity; `active` is `valid_to is null and superseded_by is null`. Cap to the `limit` most-connected entities (and the edges among them) so a large graph stays responsive; report the cap in the payload as `{capped: bool, total_nodes, total_edges}`.
- `api.graph(limit, data_dir)` + `service.graph` / `remote.graph` wrappers, and `GET /graph?limit=`.

### Frontend
- New view `dashboard/src/views/ExplorerView.tsx`, added to `NAV` after Memory (`{ id: "explorer", label: "Explorer" }`), with `explorer` added to the `View` union.
- A `ForceGraph.tsx` component using `d3-force` for layout and SVG for rendering (Canvas is a later optimisation for very large graphs). Nodes are circles sized by degree and coloured by a small type-to-colour scale from the design tokens; edges are lines coloured by predicate, dashed and faded when inactive.
- Interactions: drag a node (`d3-drag`), zoom and pan (`d3-zoom`), hover for a tooltip (name, type, degree), click a node to open a detail panel listing its connected triples (predicate plus the other entity, active or superseded).
- The simulation runs to settle then stops (no perpetual animation), and is skipped under `prefers-reduced-motion` (static layout). A legend explains node colour (type) and the faded-edge meaning (superseded). A header shows node and edge counts and the cap note.
- Loading uses `SkeletonCard`, errors `ErrorBanner`, an empty graph an `EmptyState` ("No knowledge graph yet").

### Accessibility
- The SVG carries a summarising `aria-label` (node and edge counts, top entities) and the view includes a visually-hidden text list of the highest-degree entities and their relations, so a screen reader gets the content without the canvas.

### Dependencies
- Add `d3-force`, `d3-selection`, `d3-zoom`, `d3-drag`, `d3-scale` to `dashboard/package.json`. All bundled by Vite at build time; nothing is loaded from a CDN at runtime.

### Testing
- Backend: `kg.graph` returns nodes with degree and edges that reference existing node ids, with the `active` flag and the cap fields; a `/graph` endpoint test.
- Frontend: the build typechecks; a Playwright smoke that the Explorer tab renders an `svg` containing node circles against a seeded graph.

## Phase 2b: live recall (follow-on, not this phase)

The "watch the agent recall live" view: stream memory-activation events as the agent retrieves, pulse the activated nodes and highlight their edges, and add a time scrubber over `valid_from`/`valid_to` for "memory as of T". This needs a memory-activation event source (the retrieve path emitting activation events over the existing A2A SSE stream or an OTel channel), which is its own piece of work. Deferred to keep 2a shippable.

## Constraints
Offline and self-contained (D3 bundled), accessible, reduced-motion respected, and read-only (the Explorer never mutates the graph). Managed mode is unaffected since `/graph` is a read endpoint.
