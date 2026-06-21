import React, { useCallback, useEffect, useState } from "react";
import { getStats } from "../api";
import type { Stats } from "../types";
import { StatCard } from "../components/StatCard";
import { GrowthChart } from "../components/GrowthChart";
import { VerificationDonut } from "../components/VerificationDonut";
import { TopList } from "../components/TopList";
import { ActivityFeed } from "../components/ActivityFeed";
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

function Ready({ stats }: { stats: Stats }) {
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
          <TopList title="Top agents" items={stats.top_agents} />
          <TopList title="Top projects" items={stats.top_projects} />
        </div>
      </div>

      <div className="rounded-lg p-5" style={cardStyle}>
        <ActivityFeed items={stats.recent_activity} />
      </div>
    </>
  );
}

export function OverviewView() {
  const [state, setState] = useState<State>({ kind: "loading" });

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      const stats = await getStats();
      setState({ kind: "ready", stats });
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

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-6 p-6">
      <div>
        <h1 className="text-lg font-semibold" style={{ color: "var(--ink)" }}>
          Home
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          Your memory at a glance.
        </p>
      </div>

      {state.kind === "loading" && (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading overview">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void load()} />
      )}

      {state.kind === "ready" && state.stats.memories.total === 0 && (
        <EmptyState
          title="No memories yet"
          description="Once your agents start storing memories, this overview fills in with growth, verification coverage, and recent activity."
        />
      )}

      {state.kind === "ready" && state.stats.memories.total > 0 && (
        <Ready stats={state.stats} />
      )}
    </section>
  );
}
