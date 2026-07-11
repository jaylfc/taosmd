import React, { useCallback, useMemo, useRef } from "react";

const DAY = 86400;
const WEEK = 7 * DAY;

export function fmtDate(epochSec: number): string {
  return new Date(epochSec * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

// Start-of-day epoch (seconds) for a "YYYY-MM-DD" growth bucket, parsed as
// local time to line the density bars up with the locale-formatted readout.
function dayEpoch(date: string): number {
  const [y, m, d] = date.split("-").map(Number);
  return new Date(y, (m || 1) - 1, d || 1).getTime() / 1000;
}

interface Props {
  /** Oldest point on the timeline (epoch seconds). */
  min: number;
  /** Newest point on the timeline, i.e. "now" (epoch seconds). */
  max: number;
  /** Current as-of position (epoch seconds), controlled. */
  value: number;
  onChange: (v: number) => void;
  /** Per-day memory counts, reused from stats.growth for the density backdrop. */
  growth: { date: string; count: number }[];
  /** Playback state; the parent owns the timer that steps `value`. */
  playing: boolean;
  onTogglePlay: () => void;
  /** When true, auto-play is unavailable (prefers-reduced-motion). */
  reducedMotion: boolean;
}

const SVG_W = 100;
const SVG_H = 24;

export function TimeScrubber({
  min,
  max,
  value,
  onChange,
  growth,
  playing,
  onTogglePlay,
  reducedMotion,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const span = Math.max(1, max - min);
  const clamped = Math.min(max, Math.max(min, value));
  const frac = (clamped - min) / span;
  const atNow = max - clamped < 1;

  // Density backdrop: mirror GrowthChart's SVG bars, but position each day by
  // its place on the [min, max] timeline rather than evenly.
  const bars = useMemo(() => {
    const maxCount = Math.max(1, ...growth.map((d) => d.count));
    const dayFrac = DAY / span; // one day as a fraction of the whole span
    const bw = Math.max(0.4, dayFrac * SVG_W * 0.85);
    return growth
      .map((d) => {
        const f = (dayEpoch(d.date) - min) / span;
        return { f, count: d.count };
      })
      .filter((d) => d.f >= 0 && d.f <= 1)
      .map((d) => ({
        x: d.f * SVG_W - bw / 2,
        h: (d.count / maxCount) * (SVG_H - 1),
        bw,
      }));
  }, [growth, min, span]);

  const setFromClientX = useCallback(
    (clientX: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const f = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      onChange(min + f * span);
    },
    [min, span, onChange],
  );

  const onPointerDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setFromClientX(e.clientX);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) setFromClientX(e.clientX);
  };

  const step = useCallback(
    (delta: number) => onChange(Math.min(max, Math.max(min, clamped + delta))),
    [clamped, min, max, onChange],
  );

  const onKeyDown = (e: React.KeyboardEvent) => {
    let handled = true;
    switch (e.key) {
      case "ArrowLeft":
      case "ArrowDown":
        step(-DAY);
        break;
      case "ArrowRight":
      case "ArrowUp":
        step(DAY);
        break;
      case "PageDown":
        step(-WEEK);
        break;
      case "PageUp":
        step(WEEK);
        break;
      case "Home":
        onChange(min);
        break;
      case "End":
        onChange(max);
        break;
      default:
        handled = false;
    }
    if (handled) e.preventDefault();
  };

  return (
    <div
      role="group"
      aria-label="Time travel: knowledge graph as of a chosen date"
      className="rounded-lg p-4"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-card)",
      }}
    >
      <div className="flex items-center gap-3">
        {!reducedMotion && (
          <button
            type="button"
            aria-pressed={playing}
            aria-label={playing ? "Pause playback" : "Play history from the start"}
            onClick={onTogglePlay}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded transition-colors duration-150"
            style={{
              background: playing ? "var(--accent-dim)" : "transparent",
              color: playing ? "var(--accent)" : "var(--muted-bright)",
              border: "1px solid var(--border)",
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" fill="currentColor">
              {playing ? (
                <>
                  <rect x="2" y="1.5" width="3" height="9" rx="0.5" />
                  <rect x="7" y="1.5" width="3" height="9" rx="0.5" />
                </>
              ) : (
                <path d="M3 1.5 L10.5 6 L3 10.5 Z" />
              )}
            </svg>
          </button>
        )}

        <div className="min-w-0 flex-1">
          <div
            ref={trackRef}
            role="slider"
            tabIndex={0}
            aria-label="As-of date"
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={clamped}
            aria-valuetext={fmtDate(clamped)}
            onKeyDown={onKeyDown}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            className="relative"
            style={{ height: 28, cursor: "pointer", touchAction: "none" }}
          >
            {/* Density backdrop: memories per day behind the track. */}
            <svg
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              preserveAspectRatio="none"
              className="absolute inset-x-0"
              style={{ top: 0, height: SVG_H, width: "100%", opacity: 0.5 }}
              aria-hidden="true"
            >
              {bars.map((b, i) => (
                <rect
                  key={i}
                  x={b.x}
                  y={SVG_H - b.h}
                  width={b.bw}
                  height={b.h}
                  fill="var(--accent)"
                  opacity={0.55}
                />
              ))}
            </svg>
            {/* Track rail. */}
            <div
              className="absolute inset-x-0"
              style={{
                top: SVG_H,
                height: 3,
                borderRadius: 999,
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
              }}
            />
            {/* Filled portion up to the handle. */}
            <div
              className="absolute left-0"
              style={{
                top: SVG_H,
                height: 3,
                width: `${frac * 100}%`,
                borderRadius: 999,
                background: "var(--accent)",
              }}
            />
            {/* Handle. */}
            <div
              className="absolute"
              style={{
                top: SVG_H - 4,
                left: `${frac * 100}%`,
                width: 12,
                height: 12,
                marginLeft: -6,
                borderRadius: 999,
                background: "var(--accent)",
                border: "2px solid var(--bg)",
                boxShadow: "var(--shadow-card)",
              }}
              aria-hidden="true"
            />
          </div>
          <div className="mt-1 flex items-center justify-between text-xs" style={{ color: "var(--muted)" }}>
            <span>{fmtDate(min)}</span>
            <span>now</span>
          </div>
        </div>

        <div className="shrink-0 text-right" style={{ minWidth: 96 }}>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            {atNow ? "Live" : "As of"}
          </div>
          <div
            className="text-sm font-semibold"
            style={{ color: atNow ? "var(--ink)" : "var(--accent)" }}
            aria-live="polite"
          >
            {atNow ? "Now" : fmtDate(clamped)}
          </div>
        </div>
      </div>
    </div>
  );
}
