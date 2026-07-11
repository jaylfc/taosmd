import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getGraph, getGraphActivations, getStats } from "../api";
import type { Graph, GraphNode } from "../types";
import { ForceGraph, colorForType } from "../components/ForceGraph";
import { TimeScrubber, fmtDate } from "../components/TimeScrubber";
import { SkeletonCard } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { EmptyState } from "../components/EmptyState";

type State =
  | { kind: "loading" }
  | { kind: "ready"; graph: Graph }
  | { kind: "error"; message: string };

interface TimeRange {
  min: number;
  max: number;
}

// Playback: step the whole timeline in ~40 discrete jumps over ~8s. Each step
// re-queries the graph; the 200ms cadence stays clear of the 150ms scrubber
// debounce so every step resolves (throttled, one query per step).
const PLAYBACK_STEPS = 40;
const PLAYBACK_MS = 200;
const SEEK_DEBOUNCE_MS = 150;

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

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

export function ExplorerView() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [showCurrent, setShowCurrent] = useState(false);
  const [activatedIds, setActivatedIds] = useState<Set<string>>(new Set());

  // Time-travel state. `asOf === null` means "now" (the live graph); a number
  // is a past instant (epoch seconds). `range` comes from stats (earliest
  // triple .. now); `growth` feeds the scrubber's density backdrop.
  const [range, setRange] = useState<TimeRange | null>(null);
  const [growth, setGrowth] = useState<{ date: string; count: number }[]>([]);
  const [asOf, setAsOf] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const reducedMotion = useMemo(prefersReducedMotion, []);
  const isLive = asOf === null;

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    setSelected(null);
    setAsOf(null);
    setPlaying(false);
    try {
      // Graph (current) and stats (for the scrubber range + density) together.
      const [graph, stats] = await Promise.all([getGraph(300), getStats()]);
      setState({ kind: "ready", graph });
      setGrowth(stats.growth);
      if (stats.earliest != null && stats.now > stats.earliest) {
        setRange({ min: stats.earliest, max: stats.now });
      } else {
        setRange(null);
      }
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

  // Re-query the graph as-of the scrubbed instant, debounced so dragging or
  // playback does not flood the server. Skips the initial render (the mount
  // load already fetched the live graph) so returning to "now" refetches but
  // arriving at "now" on mount does not double-fetch.
  const firstSeek = useRef(true);
  useEffect(() => {
    if (firstSeek.current) {
      firstSeek.current = false;
      return;
    }
    let cancelled = false;
    const id = window.setTimeout(async () => {
      try {
        const graph = await getGraph(300, asOf ?? undefined);
        if (!cancelled) setState({ kind: "ready", graph });
      } catch (err) {
        if (!cancelled)
          setState({
            kind: "error",
            message: err instanceof Error ? err.message : String(err),
          });
      }
    }, SEEK_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, [asOf]);

  // Playback: walk earliest -> now in discrete steps, then snap back to live.
  // Disabled under prefers-reduced-motion (manual stepping still works).
  useEffect(() => {
    if (!playing || !range || reducedMotion) return;
    const inc = (range.max - range.min) / PLAYBACK_STEPS;
    let cur = range.min;
    setAsOf(cur);
    const id = window.setInterval(() => {
      cur += inc;
      if (cur >= range.max) {
        window.clearInterval(id);
        setAsOf(null);
        setPlaying(false);
      } else {
        setAsOf(cur);
      }
    }, PLAYBACK_MS);
    return () => window.clearInterval(id);
  }, [playing, range, reducedMotion]);

  // History has no live recall, so drop any pulse when leaving "now".
  useEffect(() => {
    if (!isLive) setActivatedIds(new Set());
  }, [isLive]);

  // Live recall: poll which entities the retrieve path recently touched and
  // pulse them. Only while viewing "now" (the past has no live recall). Uses
  // last_accessed_at via /graph/activations, no event bus.
  useEffect(() => {
    if (state.kind !== "ready" || state.graph.nodes.length === 0 || !isLive) return;
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
  }, [state, isLive]);

  const onScrub = useCallback(
    (v: number) => {
      if (playing) setPlaying(false);
      // Snapping to the right edge returns to the live graph (as_of omitted).
      setAsOf(range && v >= range.max - 1 ? null : v);
    },
    [playing, range],
  );

  const returnToNow = useCallback(() => {
    setPlaying(false);
    setAsOf(null);
  }, []);

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

  const asOfLabel = isLive || asOf == null ? null : fmtDate(asOf);

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-6">
      <div>
        <h1 className="text-lg font-semibold" style={{ color: "var(--ink)" }}>
          Memory Explorer
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          The knowledge graph: entities and how they connect. Drag to pan, scroll to zoom,
          click an entity for detail. Use the time scrubber below to travel to any past state.
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
              {isLive ? (
                <span>
                  {state.graph.total_nodes} entities, {state.graph.total_edges} relations
                  {state.graph.capped ? ` (showing the ${state.graph.nodes.length} most connected)` : ""}
                </span>
              ) : (
                <span>
                  {state.graph.nodes.length} entities, {state.graph.edges.length} relations active on{" "}
                  {asOfLabel}
                </span>
              )}
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
            </span>
            {/* The All/Current split only means something on the live graph; a
                historical view already shows exactly the facts live then. */}
            {isLive && (
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
            )}
          </div>

          {!isLive && (
            <div
              role="status"
              className="flex flex-wrap items-center justify-between gap-2 rounded-lg px-4 py-2 text-sm"
              style={{
                background: "var(--accent-dim)",
                border: "1px solid var(--border)",
                color: "var(--ink)",
              }}
            >
              <span className="flex items-center gap-2">
                <span
                  className="inline-block rounded-full"
                  style={{ width: 7, height: 7, background: "var(--accent)" }}
                  aria-hidden="true"
                />
                Viewing history: the graph as it stood on {asOfLabel}
              </span>
              <button
                type="button"
                onClick={returnToNow}
                className="rounded px-2.5 py-1 text-xs font-medium transition-colors duration-150"
                style={{ background: "var(--accent)", color: "var(--bg)" }}
              >
                Return to now
              </button>
            </div>
          )}

          <ForceGraph
            nodes={state.graph.nodes}
            edges={visibleEdges}
            onSelect={setSelected}
            selectedId={selected?.id}
            activatedIds={activatedIds}
            asOfLabel={asOfLabel}
          />

          {range && (
            <TimeScrubber
              min={range.min}
              max={range.max}
              value={asOf ?? range.max}
              onChange={onScrub}
              growth={growth}
              playing={playing}
              onTogglePlay={() => setPlaying((p) => !p)}
              reducedMotion={reducedMotion}
            />
          )}

          {selected && <DetailPanel node={selected} graph={state.graph} />}

          {/* Screen-reader text fallback for the canvas. */}
          <ul className="sr-only">
            <li>
              {isLive
                ? `Knowledge graph as of now, ${state.graph.nodes.length} entities and ${state.graph.edges.length} relations shown.`
                : `Knowledge graph as of ${asOfLabel}, ${state.graph.nodes.length} entities and ${state.graph.edges.length} relations active then.`}
            </li>
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
