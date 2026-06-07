import React, { useState, useCallback } from "react";
import { getPending } from "../api";
import type { PendingItem } from "../types";
import { Input } from "../components/Input";
import { Button } from "../components/Button";
import { SkeletonCard } from "../components/Skeleton";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";

interface PendingViewProps {
  defaultAgent: string;
}

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "results"; items: PendingItem[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

function PendingRow({ item }: { item: PendingItem }) {
  const triple = [item.subject, item.predicate, item.object]
    .filter(Boolean)
    .join(" → ");
  return (
    <article
      className="rounded-lg p-4"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border-subtle)",
      }}
    >
      <p
        className="text-sm font-medium"
        style={{ color: "var(--ink)" }}
      >
        {triple || item.id}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {item.kind && <MetaChip label="kind" value={item.kind} />}
        {item.id && <MetaChip label="id" value={item.id} mono />}
      </div>
    </article>
  );
}

function MetaChip({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <span
      className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs"
      style={{
        background: "var(--surface)",
        color: "var(--muted)",
      }}
    >
      <span style={{ color: "var(--muted-bright)" }}>{label}</span>
      <span style={{ fontFamily: mono ? "var(--font-mono)" : undefined }}>
        {value}
      </span>
    </span>
  );
}

export function PendingView({ defaultAgent }: PendingViewProps) {
  const [agent, setAgent] = useState(defaultAgent);
  const [state, setState] = useState<State>({ kind: "idle" });

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      const items = await getPending(agent.trim());
      setState(items.length ? { kind: "results", items } : { kind: "empty" });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [agent]);

  return (
    <section className="flex flex-col gap-6 p-6 max-w-2xl w-full mx-auto">
      <div>
        <h1
          className="text-lg font-semibold"
          style={{ color: "var(--ink)" }}
        >
          Pending review
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Knowledge graph updates awaiting your decision. Resolve via{" "}
          <code
            className="rounded px-1 py-0.5 text-xs"
            style={{
              background: "var(--surface-2)",
              color: "var(--muted-bright)",
              fontFamily: "var(--font-mono)",
            }}
          >
            taosmd review
          </code>{" "}
          in the CLI.
        </p>
      </div>

      <div
        className="rounded-lg p-4 flex flex-wrap gap-3 items-end"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
        }}
      >
        <div className="flex-none w-36">
          <Input
            id="pend-agent"
            label="Agent"
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void load()}
            placeholder="default"
            autoComplete="off"
          />
        </div>
        <Button
          onClick={() => void load()}
          disabled={state.kind === "loading"}
        >
          Load queue
        </Button>
      </div>

      {state.kind === "idle" && (
        <EmptyState
          title="Load the pending queue"
          description="Pending decisions are knowledge-graph updates the librarian has deferred for your review."
        />
      )}

      {state.kind === "loading" && (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading pending items">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void load()} />
      )}

      {state.kind === "empty" && (
        <EmptyState
          title="Nothing pending"
          description="The review queue is empty — no decisions are waiting."
        />
      )}

      {state.kind === "results" && (
        <div className="flex flex-col gap-3" role="list" aria-label="Pending decisions">
          {state.items.map((item, i) => (
            <div key={item.id || i} role="listitem">
              <PendingRow item={item} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
