import React, { useState, useCallback } from "react";
import { searchMemory } from "../api";
import type { Hit } from "../types";
import { Input } from "../components/Input";
import { Button } from "../components/Button";
import { SkeletonCard } from "../components/Skeleton";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";

interface MemoryViewProps {
  defaultAgent: string;
}

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "results"; hits: Hit[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

function HitRow({ hit }: { hit: Hit }) {
  return (
    <article
      className="rounded-lg p-4"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border-subtle)",
      }}
    >
      <p
        className="text-sm whitespace-pre-wrap break-words"
        style={{ color: "var(--ink)" }}
      >
        {hit.text}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {hit.source && <Chip label="source" value={hit.source} />}
        {hit.confidence != null && (
          <Chip label="confidence" value={hit.confidence.toFixed(3)} />
        )}
        {hit.timestamp && <Chip label="when" value={hit.timestamp} mono />}
      </div>
    </article>
  );
}

function Chip({
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

export function MemoryView({ defaultAgent }: MemoryViewProps) {
  const [agent, setAgent] = useState(defaultAgent);
  const [query, setQuery] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  const doSearch = useCallback(async () => {
    const a = agent.trim();
    const q = query.trim();
    if (!a || !q) return;
    setState({ kind: "loading" });
    try {
      const hits = await searchMemory(q, a, 10);
      setState(hits.length ? { kind: "results", hits } : { kind: "empty" });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, [agent, query]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") void doSearch();
  };

  return (
    <section className="flex flex-col gap-6 p-6 max-w-2xl w-full mx-auto">
      <div>
        <h1
          className="text-lg font-semibold"
          style={{ color: "var(--ink)" }}
        >
          Memory
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Search what this agent remembers.
        </p>
      </div>

      <div
        className="rounded-lg p-4 flex flex-col gap-4"
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
        }}
      >
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex-none w-36">
            <Input
              id="mem-agent"
              label="Agent"
              value={agent}
              onChange={(e) => setAgent(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="default"
              autoComplete="off"
            />
          </div>
          <div className="flex-1 min-w-48">
            <Input
              id="mem-query"
              label="Query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="what do you want to recall?"
              autoComplete="off"
            />
          </div>
          <Button
            onClick={() => void doSearch()}
            disabled={!agent.trim() || !query.trim() || state.kind === "loading"}
          >
            Search
          </Button>
        </div>
      </div>

      {state.kind === "idle" && (
        <EmptyState
          title="Search this agent's memory"
          description={'Try a question you\'ve told it before, like "What\'s my preferred stack?" or "When did we discuss X?"'}
        />
      )}

      {state.kind === "loading" && (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Searching">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void doSearch()} />
      )}

      {state.kind === "empty" && (
        <EmptyState
          title="No matches found"
          description="The agent hasn't stored anything matching that query yet."
        />
      )}

      {state.kind === "results" && (
        <div className="flex flex-col gap-3" role="list" aria-label="Memory results">
          {state.hits.map((hit, i) => (
            <div key={i} role="listitem">
              <HitRow hit={hit} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
