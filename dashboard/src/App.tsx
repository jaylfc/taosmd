import React, { useState, useEffect } from "react";
import { MemoryView } from "./views/MemoryView";
import { PendingView } from "./views/PendingView";
import { A2AView } from "./views/A2AView";
import { ProjectsView } from "./views/ProjectsView";
import { SettingsView } from "./views/SettingsView";
import { OverviewView } from "./views/OverviewView";
import { ExplorerView } from "./views/ExplorerView";
import { getHealth } from "./api";
import { readAppliedScheme, setScheme, type Scheme } from "./theme";
import type { View, HealthInfo } from "./types";

interface NavItemProps {
  id: View;
  label: string;
  active: boolean;
  onClick: () => void;
}

function NavItem({ id, label, active, onClick }: NavItemProps) {
  return (
    <button
      role="tab"
      aria-selected={active}
      aria-controls={`panel-${id}`}
      id={`tab-${id}`}
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded px-3 py-2 text-sm font-medium transition-colors duration-150 text-left"
      style={{
        background: active ? "var(--accent-dim)" : "transparent",
        color: active ? "var(--accent)" : "var(--muted-bright)",
      }}
    >
      {label}
    </button>
  );
}

// Inline theme icons: currentColor-driven, sized to the toggle button, no icon
// library (offline constraint). aria-hidden — the button carries the label.
function SunIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2.5M12 19.5V22M4.22 4.22l1.77 1.77M18.01 18.01l1.77 1.77M2 12h2.5M19.5 12H22M4.22 19.78l1.77-1.77M18.01 5.99l1.77-1.77" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M20.5 14.2A8 8 0 0 1 9.8 3.5a7 7 0 1 0 10.7 10.7Z" />
    </svg>
  );
}

function HealthChip({ info }: { info: HealthInfo | null }) {
  if (!info) {
    return (
      <span className="text-xs" style={{ color: "var(--muted)" }}>
        checking…
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1.5 text-xs"
      style={{ color: "var(--muted)" }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: "var(--success)" }}
        aria-hidden="true"
      />
      v{info.version}
    </span>
  );
}

const NAV: { id: View; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "memory", label: "Memory" },
  { id: "explorer", label: "Explorer" },
  { id: "projects", label: "Projects" },
  { id: "pending", label: "Pending" },
  { id: "a2a", label: "A2A channels" },
  { id: "settings", label: "Settings" },
];

export function App() {
  const [view, setView] = useState<View>("home");
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // The blocking bootstrap in index.html has already applied the correct scheme
  // to <html> before first paint; read it so React hydrates without re-flashing.
  const [scheme, setSchemeState] = useState<Scheme>(readAppliedScheme);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => {});
  }, []);

  // Keyboard nav for tabs
  const handleNavKeyDown = (e: React.KeyboardEvent, idx: number) => {
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      const next = NAV[(idx + 1) % NAV.length];
      setView(next.id);
      document.getElementById(`tab-${next.id}`)?.focus();
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      const prev = NAV[(idx - 1 + NAV.length) % NAV.length];
      setView(prev.id);
      document.getElementById(`tab-${prev.id}`)?.focus();
    }
  };

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: "var(--bg)" }}
    >
      {/* Top bar */}
      <header
        className="flex items-center gap-4 px-5 py-3 shrink-0"
        style={{
          background: "var(--surface)",
          borderBottom: "1px solid var(--border)",
          height: 52,
        }}
      >
        <button
          onClick={() => setSidebarOpen((v) => !v)}
          className="rounded p-1 transition-colors duration-150 lg:hidden"
          style={{ color: "var(--muted)" }}
          aria-label={sidebarOpen ? "Close navigation" : "Open navigation"}
        >
          ☰
        </button>

        <span
          className="text-sm font-semibold"
          style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}
        >
          taOSmd
        </span>
        <span
          className="text-xs hidden sm:inline"
          style={{ color: "var(--muted)" }}
        >
          memory inspector
        </span>

        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={() => {
              const next = scheme === "dark" ? "light" : "dark";
              setScheme(next);
              setSchemeState(next);
            }}
            className="inline-flex items-center justify-center rounded p-1.5 transition-colors duration-150"
            style={{ color: "var(--muted)" }}
            aria-label={scheme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            aria-pressed={scheme === "light"}
          >
            {scheme === "dark" ? <SunIcon /> : <MoonIcon />}
          </button>
          <HealthChip info={health} />
        </div>
      </header>

      {/* Body: sidebar + content */}
      <div className="flex flex-1 min-h-0">
        {/* Left nav */}
        <nav
          role="tablist"
          aria-label="Views"
          className={`flex flex-col gap-1 shrink-0 p-3 ${
            sidebarOpen ? "" : "hidden lg:flex"
          }`}
          style={{
            width: "var(--sidebar-w)",
            background: "var(--surface)",
            borderRight: "1px solid var(--border)",
          }}
        >
          {NAV.map((item, idx) => (
            <NavItem
              key={item.id}
              id={item.id}
              label={item.label}
              active={view === item.id}
              onClick={() => {
                setView(item.id);
                setSidebarOpen(false);
              }}
              {...{
                onKeyDown: (e: React.KeyboardEvent) =>
                  handleNavKeyDown(e, idx),
              }}
            />
          ))}
        </nav>

        {/* Content */}
        <main className="flex flex-1 min-w-0 overflow-y-auto" id={`panel-${view}`}>
          {view === "home" && <OverviewView />}
          {view === "memory" && <MemoryView defaultAgent="default" />}
          {view === "explorer" && <ExplorerView />}
          {view === "projects" && <ProjectsView />}
          {view === "pending" && <PendingView defaultAgent="default" />}
          {view === "a2a" && (
            <div className="flex flex-col flex-1 h-full min-h-0 w-full">
              <A2AView />
            </div>
          )}
          {view === "settings" && <SettingsView />}
        </main>
      </div>
    </div>
  );
}
