import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
  type SimulationLinkDatum,
} from "d3-force";
import type { GraphNode, GraphEdge } from "../types";

const PALETTE = [
  "var(--accent)",
  "var(--success)",
  "var(--warning)",
  "var(--info)",
  "var(--error)",
  "var(--muted-bright)",
];

export function colorForType(type: string): string {
  let h = 0;
  for (let i = 0; i < type.length; i++) h = (h * 31 + type.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

interface SimNode extends GraphNode, SimulationNodeDatum {}

const W = 1000;

export function ForceGraph({
  nodes,
  edges,
  onSelect,
  selectedId,
  activatedIds,
  height = 480,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onSelect: (n: GraphNode) => void;
  selectedId?: string | null;
  activatedIds?: Set<string>;
  height?: number;
}) {
  const H = height;

  // Static layout: settle the force simulation once per data set, then render.
  // No perpetual animation, so it is cheap and reduced-motion safe. d3's
  // forceLink mutates the link source/target into node refs, so the simulation
  // runs on a throwaway edge copy; rendering uses the original string ids.
  const layout = useMemo(() => {
    const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
    const ids = new Set(simNodes.map((n) => n.id));
    const renderEdges = edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    const simEdges: SimulationLinkDatum<SimNode>[] = renderEdges.map((e) => ({
      source: e.source,
      target: e.target,
    }));
    const sim = forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        forceLink<SimNode, SimulationLinkDatum<SimNode>>(simEdges)
          .id((d) => d.id)
          .distance(70)
          .strength(0.4),
      )
      .force("charge", forceManyBody<SimNode>().strength(-160))
      .force("center", forceCenter(W / 2, H / 2))
      .force("collide", forceCollide<SimNode>().radius((d) => 8 + Math.sqrt(d.degree || 0) * 2.2))
      .stop();
    for (let i = 0; i < 320; i++) sim.tick();
    const posById: Record<string, { x: number; y: number }> = {};
    simNodes.forEach((n) => {
      posById[n.id] = { x: n.x ?? W / 2, y: n.y ?? H / 2 };
    });
    return { simNodes, renderEdges, posById };
  }, [nodes, edges, H]);

  const [t, setT] = useState({ x: 0, y: 0, k: 1 });
  const [hover, setHover] = useState<SimNode | null>(null);
  const pan = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null);

  useEffect(() => {
    setT({ x: 0, y: 0, k: 1 });
  }, [layout]);

  const onWheel = (e: React.WheelEvent) => {
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    setT((p) => ({ ...p, k: Math.min(4, Math.max(0.3, p.k * factor)) }));
  };
  const onBgDown = (e: React.PointerEvent) => {
    pan.current = { sx: e.clientX, sy: e.clientY, ox: t.x, oy: t.y };
  };
  const onMove = (e: React.PointerEvent) => {
    if (!pan.current) return;
    setT((p) => ({
      ...p,
      x: pan.current!.ox + (e.clientX - pan.current!.sx),
      y: pan.current!.oy + (e.clientY - pan.current!.sy),
    }));
  };
  const onUp = () => {
    pan.current = null;
  };

  const label = `Knowledge graph with ${layout.simNodes.length} entities and ${layout.renderEdges.length} relations`;

  return (
    <div
      className="relative overflow-hidden rounded-lg"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)", height: H }}
    >
      <svg
        width="100%"
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={label}
        onWheel={onWheel}
        onPointerDown={onBgDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        style={{ cursor: pan.current ? "grabbing" : "grab", touchAction: "none" }}
      >
        <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
          {layout.renderEdges.map((e, i) => {
            const a = layout.posById[e.source];
            const b = layout.posById[e.target];
            if (!a || !b) return null;
            return (
              <line
                key={i}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={e.active ? "var(--muted-bright)" : "var(--muted)"}
                strokeOpacity={e.active ? 0.45 : 0.18}
                strokeDasharray={e.active ? undefined : "3 3"}
                strokeWidth={1}
              />
            );
          })}
          {layout.simNodes.map((n) => {
            const p = layout.posById[n.id];
            const r = 5 + Math.sqrt(n.degree || 0) * 2.2;
            const sel = selectedId === n.id;
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y})`}
                style={{ cursor: "pointer" }}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelect(n);
                }}
                onMouseEnter={() => setHover(n)}
                onMouseLeave={() => setHover(null)}
              >
                {activatedIds?.has(n.id) && (
                  <circle
                    className="kg-pulse-ring"
                    r={r}
                    fill={colorForType(n.type)}
                    style={{ pointerEvents: "none" }}
                  />
                )}
                <circle
                  r={r}
                  fill={colorForType(n.type)}
                  fillOpacity={sel ? 1 : 0.85}
                  stroke={sel ? "var(--ink)" : "var(--bg)"}
                  strokeWidth={sel ? 2 : 1}
                />
                {(n.degree >= 3 || sel) && (
                  <text x={r + 3} y={3} fontSize={10} fill="var(--ink)" style={{ pointerEvents: "none" }}>
                    {n.name}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {hover && (
        <div
          className="pointer-events-none absolute left-3 top-3 rounded px-2 py-1 text-xs"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          <span style={{ color: "var(--ink)" }}>{hover.name}</span>
          <span style={{ color: "var(--muted)" }}>
            {" "}
            &middot; {hover.type} &middot; {hover.degree} links
          </span>
        </div>
      )}
    </div>
  );
}
