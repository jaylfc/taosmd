import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getStats, getMemories } from "../api";
import type { Stats, Memory } from "../types";
import { StatCard } from "../components/StatCard";
import { GrowthChart } from "../components/GrowthChart";
import { VerificationDonut } from "../components/VerificationDonut";
import { TopList } from "../components/TopList";
import { ActivityFeed } from "../components/ActivityFeed";
import { MemoryList } from "../components/MemoryList";
import { ScopeSelector } from "../components/ScopeSelector";
import { SkeletonCard } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";
import { EmptyState } from "../components/EmptyState";

type State =
  | { kind: "loading" }
  | { kind: "ready"; stats: Stats }
  | { kind: "error"; message: string };

const cardStyle: React.CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  boxShadow: "var(--shadow-card)",
};

function Ready({ stats, memories }: { stats: Stats; memories: Memory[] }) {
  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total memories"
          value={stats.memories.total.toLocaleString()}
          sublabel={`${stats.memories.disk_mb} MB on disk`}
        />
        <StatCard label="Agents" value={stats.agents} />
        <StatCard label="Projects" value={stats.projects} />
        <StatCard
          label="Verified facts"
          value={stats.verification.supported}
          sublabel="checked against source"
        />
      </div>

      <GrowthChart data={stats.growth} />

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg p-5" style={cardStyle}>
          <h2 className="mb-4 text-sm font-semibold" style={{ color: "var(--ink)" }}>
            Verification coverage
          </h2>
          <VerificationDonut
            supported={stats.verification.supported}
            unverified={stats.verification.unverified}
            flagged={stats.verification.flagged}
            hallucinationRate={stats.verification.hallucination_rate}
          />
        </div>
        <div className="flex flex-col gap-5 rounded-lg p-5" style={cardStyle}>
          <TopList title="Top categories" items={stats.categories} />
          <TopList title="Top agents" items={stats.top_agents} />
          <TopList title="Top projects" items={stats.top_projects} />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg p-5" style={cardStyle}>
          <MemoryList items={memories} />
        </div>
        <div className="rounded-lg p-5" style={cardStyle}>
          <ActivityFeed items={stats.recent_activity} />
        </div>
      </div>
    </>
  );
}

export function OverviewView() {
  const [scope, setScope] = useState("all");
  const [state, setState] = useState<State>({ kind: "loading" });
  const [memories, setMemories] = useState<Memory[]>([]);
  const [agentNames, setAgentNames] = useState<string[]>([]);

  const load = useCallback(async (sc: string) => {
    setState({ kind: "loading" });
    try {
      const [stats, mems] = await Promise.all([getStats(sc), getMemories(sc, 20)]);
      setState({ kind: "ready", stats });
      setMemories(mems);
      if (sc === "all") {
        setAgentNames(stats.top_agents.map((a) => a.name).filter(Boolean));
      }
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  useEffect(() => {
    void load(scope);
  }, [load, scope]);

  const scopeOptions = useMemo(() => {
    const names = Array.from(
      new Set(["user", ...agentNames].filter((n) => n && n !== "all")),
    );
    return [
      { value: "all", label: "All memory" },
      ...names.map((n) => ({ value: n, label: n === "user" ? "User (you)" : n })),
    ];
  }, [agentNames]);

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold" style={{ color: "var(--ink)" }}>
            Home
          </h1>
          <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
            {scope === "all"
              ? "Your memory at a glance."
              : `Memory scoped to ${scope === "user" ? "you" : scope}.`}
          </p>
        </div>
        <ScopeSelector scope={scope} options={scopeOptions} onChange={setScope} />
      </div>

      {state.kind === "loading" && (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading overview">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void load(scope)} />
      )}

      {state.kind === "ready" && state.stats.memories.total === 0 && (
        <EmptyState
          title="No memories yet"
          description="Once memories are stored in this scope, the overview fills in with growth, verification coverage, and recent memories."
        />
      )}

      {state.kind === "ready" && state.stats.memories.total > 0 && (
        <Ready stats={state.stats} memories={memories} />
      )}
    </section>
  );
}
