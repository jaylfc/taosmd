import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGraph, getGraphActivations } from "../api";
import type { Graph, GraphNode } from "../types";
import { ForceGraph, colorForType } from "../components/ForceGraph";
import { SkeletonCard } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { EmptyState } from "../components/EmptyState";

type State =
  | { kind: "loading" }
  | { kind: "ready"; graph: Graph }
  | { kind: "error"; message: string };

const cardStyle: React.CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  boxShadow: "var(--shadow-card)",
};

function DetailPanel({ node, graph }: { node: GraphNode; graph: Graph }) {
  const nameById = useMemo(() => {
    const m: Record<string, string> = {};
    graph.nodes.forEach((n) => {
      m[n.id] = n.name;
    });
    return m;
  }, [graph.nodes]);

  const relations = graph.edges
    .filter((e) => e.source === node.id || e.target === node.id)
    .map((e) => {
      const outgoing = e.source === node.id;
      const other = outgoing ? e.target : e.source;
      return {
        predicate: e.predicate,
        other: nameById[other] || other,
        outgoing,
        active: e.active,
      };
    });

  return (
    <div className="rounded-lg p-5" style={cardStyle}>
      <div className="flex items-center gap-2">
        <span
          className="inline-block rounded-full"
          style={{ width: 10, height: 10, background: colorForType(node.type) }}
          aria-hidden="true"
        />
        <h2 className="text-sm font-semibold" style={{ color: "var(--ink)" }}>
          {node.name}
        </h2>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {node.type} &middot; {node.degree} links
        </span>
      </div>
      {relations.length === 0 ? (
        <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
          No relations.
        </p>
      ) : (
        <ul className="mt-3 flex flex-col gap-1.5" role="list">
          {relations.map((r, i) => (
            <li key={i} className="text-sm" style={{ color: "var(--ink)" }}>
              <span style={{ color: "var(--muted)" }}>{r.outgoing ? "" : "is "}</span>
              <span style={{ color: "var(--accent)" }}>{r.predicate}</span>
              <span style={{ color: "var(--muted)" }}>{r.outgoing ? " " : " by "}</span>
              <span>{r.other}</span>
              {!r.active && (
                <span className="ml-1.5 text-xs" style={{ color: "var(--muted)" }}>
                  (superseded)
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const fmtTime = (t: number) => new Date(t * 1000).toLocaleString();

function Scrubber({
  graph,
  asOf,
  setAsOf,
}: {
  graph: Graph;
  asOf: number | null;
  setAsOf: (v: number | null) => void;
}) {
  const lo = graph.t_min;
  const hi = graph.t_max;
  if (lo == null || hi == null || hi <= lo) return null;
  const value = asOf ?? hi;
  return (
    <div className="flex items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
      <span className="shrink-0">Memory as of</span>
      <input
        type="range"
        min={lo}
        max={hi}
        step={Math.max((hi - lo) / 240, 1)}
        value={value}
        onChange={(e) => {
          const v = Number(e.target.value);
          // Dragging fully right returns to the live view (no as_of).
          setAsOf(v >= hi ? null : v);
        }}
        aria-label="Reconstruct the graph as of a past instant"
        aria-valuetext={asOf != null ? fmtTime(asOf) : "now"}
        className="flex-1"
        style={{ accentColor: "var(--accent)" }}
      />
      <span className="shrink-0 tabular-nums" style={{ color: asOf != null ? "var(--accent)" : "var(--muted)" }}>
        {asOf != null ? fmtTime(asOf) : "now"}
      </span>
      {asOf != null && (
        <button
          type="button"
          onClick={() => setAsOf(null)}
          className="shrink-0 rounded px-2 py-0.5 font-medium"
          style={{ border: "1px solid var(--border)", color: "var(--accent)" }}
        >
          Now
        </button>
      )}
    </div>
  );
}

export function ExplorerView() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [showCurrent, setShowCurrent] = useState(false);
  const [activatedIds, setActivatedIds] = useState<Set<string>>(new Set());
  const [asOf, setAsOf] = useState<number | null>(null);
  const firstScrub = useRef(true);

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    setSelected(null);
    try {
      setState({ kind: "ready", graph: await getGraph(300) });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Live recall: poll which entities the retrieve path recently touched and
  // pulse them. Uses last_accessed_at via /graph/activations, no event bus.
  useEffect(() => {
    if (state.kind !== "ready" || state.graph.nodes.length === 0) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const { activations } = await getGraphActivations(45);
        if (!cancelled) setActivatedIds(new Set(activations.map((a) => a.id)));
      } catch {
        // a failed poll leaves the static graph intact
      }
    };
    void poll();
    const id = window.setInterval(poll, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [state]);

  // Time scrubber: when as_of changes, refetch the graph as it stood at that
  // instant (null = the live view). Debounced so dragging the slider does not
  // spam the endpoint. The mount run is skipped because load() already fetched.
  useEffect(() => {
    if (firstScrub.current) {
      firstScrub.current = false;
      return;
    }
    let cancelled = false;
    const t = window.setTimeout(async () => {
      try {
        const g = await getGraph(300, asOf);
        if (!cancelled) setState((s) => (s.kind === "ready" ? { kind: "ready", graph: g } : s));
      } catch {
        // keep the current snapshot if a scrub fetch fails
      }
    }, 150);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [asOf]);

  // Stable identity for the edge set: selecting a node re-renders ExplorerView,
  // and an inline edges.filter(...) would hand ForceGraph a fresh array each
  // time, invalidating its layout memo and snapping the user's pan/zoom back.
  // Only toggling All/Current or loading new graph data should recompute.
  const visibleEdges = useMemo(
    () =>
      state.kind === "ready"
        ? showCurrent
          ? state.graph.edges.filter((e) => e.active)
          : state.graph.edges
        : [],
    [state, showCurrent],
  );

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-6">
      <div>
        <h1 className="text-lg font-semibold" style={{ color: "var(--ink)" }}>
          Memory Explorer
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          The knowledge graph: entities and how they connect. Drag to pan, scroll to zoom,
          click an entity for detail.
        </p>
      </div>

      {state.kind === "loading" && (
        <div aria-busy="true" aria-label="Loading the graph">
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void load()} />
      )}

      {state.kind === "ready" && state.graph.nodes.length === 0 && (
        <EmptyState
          title="No knowledge graph yet"
          description="As memories are extracted into entities and relations, they appear here as a graph you can explore."
        />
      )}

      {state.kind === "ready" && state.graph.nodes.length > 0 && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs" style={{ color: "var(--muted)" }}>
            <span className="flex items-center gap-2">
              <span>
                {state.graph.total_nodes} entities, {state.graph.total_edges} relations
                {state.graph.capped ? ` (showing the ${state.graph.nodes.length} most connected)` : ""}
              </span>
              {activatedIds.size > 0 && (
                <span className="flex items-center gap-1.5" style={{ color: "var(--success)" }}>
                  <span
                    className="inline-block rounded-full"
                    style={{ width: 7, height: 7, background: "var(--success)" }}
                    aria-hidden="true"
                  />
                  {activatedIds.size} recalled live
                </span>
              )}
              {asOf != null && (
                <span className="flex items-center gap-1.5" style={{ color: "var(--accent)" }}>
                  <span
                    className="inline-block rounded-full"
                    style={{ width: 7, height: 7, background: "var(--accent)" }}
                    aria-hidden="true"
                  />
                  as of {fmtTime(asOf)}
                </span>
              )}
            </span>
            <span className="flex items-center gap-3">
              <span
                role="radiogroup"
                aria-label="Show facts"
                className="inline-flex rounded"
                style={{ border: "1px solid var(--border)", overflow: "hidden" }}
              >
                {([["all", "All"], ["current", "Current"]] as const).map(([val, lbl], i) => {
                  const active = showCurrent === (val === "current");
                  return (
                    <button
                      key={val}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      onClick={() => setShowCurrent(val === "current")}
                      className="px-2 py-0.5 text-xs font-medium transition-colors duration-150"
                      style={{
                        background: active ? "var(--accent-dim)" : "transparent",
                        color: active ? "var(--accent)" : "var(--muted-bright)",
                        borderLeft: i === 0 ? "none" : "1px solid var(--border)",
                      }}
                    >
                      {lbl}
                    </button>
                  );
                })}
              </span>
              {!showCurrent && (
                <span className="flex items-center gap-1.5">
                  <span style={{ width: 16, height: 0, borderTop: "1px dashed var(--muted)" }} aria-hidden="true" />
                  superseded
                </span>
              )}
            </span>
          </div>

          <Scrubber graph={state.graph} asOf={asOf} setAsOf={setAsOf} />

          <ForceGraph
            nodes={state.graph.nodes}
            edges={visibleEdges}
            onSelect={setSelected}
            selectedId={selected?.id}
            activatedIds={activatedIds}
          />

          {selected && <DetailPanel node={selected} graph={state.graph} />}

          {/* Screen-reader text fallback for the canvas. */}
          <ul className="sr-only">
            {state.graph.nodes.slice(0, 30).map((n) => (
              <li key={n.id}>
                {n.name} ({n.type}), {n.degree} links
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
