# Dashboard Phase 2c: Memory Explorer time scrubber ("memory as of T")

Status: design. Small follow-on to Phase 2a/2b (Memory Explorer galaxy + live-recall pulse, merged in PR #170/#171).

## Goal

Let a user drag a time slider in the Memory Explorer and watch the knowledge graph reconstruct itself as it stood at any past instant. This is the temporal-provenance story made visible: facts appear when first learned, and facts that were later replaced fade out of the snapshot. It reuses the KG's existing temporal validity columns, so no new storage or events are needed.

## Data model (already present)

`kg_triples` carries `valid_from REAL NOT NULL` and `valid_to REAL` (null while current). A fact is active at instant `T` when:

```
valid_from <= T AND (valid_to IS NULL OR valid_to >= T)
```

This is the same predicate `query_entity` already uses for point-in-time reads, so the scrubber is just that predicate applied to the whole-graph projection. (`superseded_by` exists in the schema but is not written by the current write paths; supersession is represented purely by `valid_to` being set, so the as-of query relies on `valid_from`/`valid_to` only.)

## Backend: extend `kg.graph()`

Add an optional `as_of: float | None = None` to `TemporalKnowledgeGraph.graph(limit=300, as_of=None)`.

- `as_of is None` (default): unchanged behaviour. All triples counted and returned; `active = valid_to IS NULL AND superseded_by IS NULL`.
- `as_of = T`: degree ranking, the edge set, and `total_nodes`/`total_edges` are all computed over only the triples active at `T` (the predicate above), so the view is the graph as it stood at `T`.
  - `active` keeps its present meaning, "still current today" (`valid_to IS NULL AND superseded_by IS NULL`). So a fact that was live at `T` but has since been replaced is included in the snapshot and rendered faded (inactive). This keeps the All/Current toggle meaningful in scrubber mode: All shows the full `T` snapshot, Current shows only the `T`-era facts that persist to now.

Always return two extra fields so the client can build the slider without a second call:

- `t_min`: `MIN(valid_from)` over all triples (the earliest fact), or `null` if empty.
- `t_max`: the latest temporal point, `max` of `MAX(valid_from)` and `MAX(valid_to)` over all triples, or `null` if empty.

The slider spans `[t_min, t_max]`; releasing it at the far right (`as_of >= t_max`) is equivalent to the current-day view.

## Plumbing (mirror the existing `graph` chain)

`api.graph(limit, as_of=None, data_dir)` -> `service.graph(limit, as_of=None, data_dir)` (forward to remote or api) -> `remote.graph(limit, as_of=None)` adds `as_of` to the `GET /graph` params when set -> `http_server._handle_graph` parses `?as_of=` as a float (400 on a non-numeric value, same as `limit`).

## Frontend: scrubber in `ExplorerView`

- `getGraph(limit, asOf?)` in `api.ts` gains an optional `asOf` that becomes `&as_of=` when present; `Graph` in `types.ts` gains `t_min: number | null` and `t_max: number | null`.
- A slider appears under the All/Current toggle when `t_min`/`t_max` are present and distinct. State: `asOf: number | null` (null = live/current view, the default). Dragging sets `asOf` and refetches `getGraph(300, asOf)`; a "Now" button (or releasing at max) clears it back to null.
- A readable label shows the snapshot instant (localized date/time) next to the slider, plus a small "as of <date>" badge in the header while scrubbing. When `asOf` is null the header reads as today (no badge).
- Refetch is debounced (about 150ms) so dragging does not spam the endpoint. The pulse polling and the All/Current toggle keep working unchanged; the memoized `visibleEdges` already keys off the graph state so a new snapshot recomputes the layout once.
- Accessibility: the slider is a native `<input type="range">` with `aria-label`, `aria-valuetext` set to the human date, and keyboard stepping; reduced-motion is unaffected (the layout is static).

## Testing

- `kg.graph(as_of=T)` over a history built with `add_triple(valid_from=...)` + `invalidate(ended_at=...)`: a fact valid only before `T` is excluded; a fact whose window contains `T` is included; a fact that started after `T` is excluded.
- The `active` flag in as-of mode: a fact live at `T` but invalidated since is present with `active False`; a still-current fact is `active True`.
- `t_min`/`t_max` are the earliest `valid_from` and the latest temporal point; both `null` on an empty graph.
- `as_of=None` regression: identical to today.
- `GET /graph?as_of=<ts>` dispatches and filters; a non-numeric `as_of` returns 400.
- Frontend: rebuild the bundle; verify in a real browser in both themes that dragging the slider reshapes the graph and the date label tracks.

## Out of scope

Animated auto-play of the timeline, per-edge tooltips of valid_from/valid_to, and diff highlighting between two instants. These are later polish if Jay wants them.
