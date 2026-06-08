import React, { useState, useCallback, useEffect } from "react";
import { getProjects, getShelves } from "../api";
import type { Project, Shelf } from "../types";
import { Button } from "../components/Button";
import { SkeletonCard } from "../components/Skeleton";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { relativeTime } from "../utils/time";

type State =
  | { kind: "loading" }
  | { kind: "results"; items: Project[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

type ShelfState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "results"; items: Shelf[] }
  | { kind: "empty" }
  | { kind: "error"; message: string };

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="inline-flex items-center rounded px-2 py-0.5 text-xs"
      style={{ background: "var(--surface)", color: "var(--muted)" }}
    >
      {children}
    </span>
  );
}

function ProjectCard({
  project,
  selected,
  onClick,
}: {
  project: Project;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={selected}
      className="w-full rounded-lg p-4 text-left transition-colors duration-150"
      style={{
        background: selected ? "var(--accent-dim)" : "var(--surface-2)",
        border: `1px solid ${selected ? "var(--accent)" : "var(--border-subtle)"}`,
      }}
    >
      <div className="flex items-center justify-between gap-3">
        <span
          className="text-sm font-medium"
          style={{ color: "var(--ink)", fontFamily: "var(--font-mono)" }}
        >
          {project.project_id}
        </span>
        {project.last_ingest > 0 && (
          <span className="text-xs shrink-0" style={{ color: "var(--muted)" }}>
            {relativeTime(project.last_ingest)}
          </span>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <Chip>
          {project.agents.length} agent{project.agents.length === 1 ? "" : "s"}
        </Chip>
        {project.agents.map((a) => (
          <Chip key={a}>{a}</Chip>
        ))}
      </div>
    </button>
  );
}

function ShelfRow({ shelf }: { shelf: Shelf }) {
  return (
    <div
      className="flex items-center justify-between gap-3 rounded px-3 py-2"
      style={{ background: "var(--surface)", border: "1px solid var(--border-subtle)" }}
    >
      <span className="text-sm" style={{ color: "var(--ink)" }}>
        {shelf.agent}
      </span>
      <span className="flex items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
        <span>
          {shelf.facts} memor{shelf.facts === 1 ? "y" : "ies"}
        </span>
        {shelf.last_ingest > 0 && <span>{relativeTime(shelf.last_ingest)}</span>}
      </span>
    </div>
  );
}

export function ProjectsView() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [selected, setSelected] = useState<string | null>(null);
  const [shelves, setShelves] = useState<ShelfState>({ kind: "idle" });

  const loadProjects = useCallback(async () => {
    setState({ kind: "loading" });
    setSelected(null);
    setShelves({ kind: "idle" });
    try {
      const items = await getProjects();
      setState(items.length ? { kind: "results", items } : { kind: "empty" });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  const loadShelves = useCallback(async (project: string) => {
    setSelected(project);
    setShelves({ kind: "loading" });
    try {
      const items = await getShelves(project);
      setShelves(items.length ? { kind: "results", items } : { kind: "empty" });
    } catch (err) {
      setShelves({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  useEffect(() => {
    void loadProjects();
  }, [loadProjects]);

  return (
    <section className="flex flex-col gap-6 p-6 max-w-2xl w-full mx-auto">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold" style={{ color: "var(--ink)" }}>
            Projects
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
            Memories tagged with a project id, and the agent shelves within each.
            Select a project to see which agents have memories in it.
          </p>
        </div>
        <Button onClick={() => void loadProjects()} disabled={state.kind === "loading"}>
          Refresh
        </Button>
      </div>

      {state.kind === "loading" && (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading projects">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void loadProjects()} />
      )}

      {state.kind === "empty" && (
        <EmptyState
          title="No project-tagged memories"
          description="Ingest with a project id (taosmd.ingest(..., project=...)) to group memory across agents working on the same project."
        />
      )}

      {state.kind === "results" && (
        <div className="flex flex-col gap-3" role="list" aria-label="Projects">
          {state.items.map((p) => (
            <div key={p.project_id} role="listitem">
              <ProjectCard
                project={p}
                selected={selected === p.project_id}
                onClick={() => void loadShelves(p.project_id)}
              />
              {selected === p.project_id && (
                <div className="mt-2 ml-3 flex flex-col gap-2" aria-label="Shelves">
                  {shelves.kind === "loading" && (
                    <SkeletonCard />
                  )}
                  {shelves.kind === "error" && (
                    <ErrorBanner
                      message={shelves.message}
                      onRetry={() => void loadShelves(p.project_id)}
                    />
                  )}
                  {shelves.kind === "empty" && (
                    <p className="text-sm" style={{ color: "var(--muted)" }}>
                      No shelves in this project.
                    </p>
                  )}
                  {shelves.kind === "results" &&
                    shelves.items.map((s) => <ShelfRow key={s.agent} shelf={s} />)}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
