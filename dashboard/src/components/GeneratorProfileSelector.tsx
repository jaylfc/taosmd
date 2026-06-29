import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getStats, getGeneratorProfile, setGeneratorProfile } from "../api";
import type { GeneratorProfileState } from "../types";
import { ScopeSelector } from "./ScopeSelector";
import { ErrorBanner } from "./ErrorBanner";

export function GeneratorProfileSelector() {
  const [scope, setScope] = useState("all");
  const [agentNames, setAgentNames] = useState<string[]>([]);
  const [profileState, setProfileState] = useState<GeneratorProfileState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch agent list once for scope options
  useEffect(() => {
    void getStats().then((s) => {
      setAgentNames(s.top_agents.map((a) => a.name).filter(Boolean));
    });
  }, []);

  const scopeOptions = useMemo(() => {
    const names = Array.from(
      new Set(["user", ...agentNames].filter((n) => n && n !== "all")),
    );
    return [
      { value: "all", label: "All agents" },
      ...names.map((n) => ({ value: n, label: n })),
    ];
  }, [agentNames]);

  const loadProfile = useCallback(async (sc: string) => {
    setLoading(true);
    setError(null);
    try {
      const state = await getGeneratorProfile(sc);
      setProfileState(state);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadProfile(scope);
  }, [loadProfile, scope]);

  const handleProfileChange = useCallback(
    async (id: string) => {
      if (!profileState) return;
      const prev = profileState.active;
      setProfileState((s) => s ? { ...s, active: id } : s);
      setError(null);
      try {
        const next = await setGeneratorProfile(id, scope);
        setProfileState(next);
      } catch (err) {
        setProfileState((s) => s ? { ...s, active: prev } : s);
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [profileState, scope],
  );

  const activeProfile = profileState
    ? profileState.profiles.find((p) => p.id === profileState.active)
    : null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label
          htmlFor="generator-profile-select"
          className="text-sm font-medium"
          style={{ color: "var(--ink)" }}
        >
          Generator profile
        </label>
        <ScopeSelector scope={scope} options={scopeOptions} onChange={setScope} />
      </div>

      {error && <ErrorBanner message={error} onRetry={() => void loadProfile(scope)} />}

      <div className="flex flex-col gap-2">
        <select
          id="generator-profile-select"
          aria-label="Generator profile"
          value={profileState?.active ?? ""}
          disabled={loading || !profileState}
          onChange={(e) => void handleProfileChange(e.target.value)}
          className="rounded px-2 py-1.5 text-sm"
          style={{
            background: "var(--surface-2)",
            color: "var(--ink)",
            border: "1px solid var(--border)",
            opacity: loading ? 0.6 : 1,
          }}
        >
          {!profileState && (
            <option value="">Loading...</option>
          )}
          {profileState?.profiles.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>

        {activeProfile && Object.keys(activeProfile.models).length > 0 && (
          <div
            className="rounded px-3 py-2 text-xs"
            style={{ background: "var(--surface-2)", color: "var(--muted)" }}
          >
            {Object.entries(activeProfile.models).map(([tier, model]) => (
              <div key={tier}>
                <span style={{ color: "var(--muted-bright)" }}>{tier}: </span>
                <code style={{ fontFamily: "var(--font-mono)" }}>
                  {model || "retrieval-only"}
                </code>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
