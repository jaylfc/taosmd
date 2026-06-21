import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getControls, postControls } from "../api";
import type {
  ControlSpec,
  ControlsSchema,
  ControlsSettings,
  ControlValue,
} from "../types";
import { SkeletonCard } from "../components/Skeleton";
import { ErrorBanner } from "../components/ErrorBanner";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; schema: ControlsSchema }
  | { kind: "error"; message: string };

const SCOPE_BADGE: Record<ControlSpec["scope"], string> = {
  runtime: "live",
  store: "re-index",
  consumer: "answer-gen",
};

function ScopeBadge({ scope }: { scope: ControlSpec["scope"] }) {
  return (
    <span
      className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide"
      style={{ background: "var(--surface-2)", color: "var(--muted)" }}
      title={
        scope === "runtime"
          ? "Takes effect on the next search"
          : scope === "store"
            ? "Set at install; changing it re-indexes the store"
            : "Applied in your answer-generation, not in taOSmd core"
      }
    >
      {SCOPE_BADGE[scope]}
    </span>
  );
}

function TradeOffs({ spec }: { spec: ControlSpec }) {
  return (
    <details className="mt-2 group">
      <summary
        className="cursor-pointer text-xs select-none"
        style={{ color: "var(--muted)" }}
      >
        trade-offs
      </summary>
      <div className="mt-1.5 flex flex-col gap-1 text-xs">
        <p style={{ color: "var(--muted-bright)" }}>
          <span style={{ color: "var(--success)" }} aria-hidden="true">
            +&nbsp;
          </span>
          {spec.pros}
        </p>
        <p style={{ color: "var(--muted-bright)" }}>
          <span style={{ color: "var(--warning)" }} aria-hidden="true">
            -&nbsp;
          </span>
          {spec.cons}
        </p>
      </div>
    </details>
  );
}

function Toggle({
  checked,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="relative inline-flex shrink-0 items-center rounded-full transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
      style={{
        width: 40,
        height: 22,
        background: checked ? "var(--accent)" : "var(--surface-2)",
        border: "1px solid var(--border)",
      }}
    >
      <span
        className="rounded-full transition-transform duration-150"
        style={{
          width: 16,
          height: 16,
          background: checked ? "#0b1020" : "var(--muted-bright)",
          transform: checked ? "translateX(19px)" : "translateX(2px)",
        }}
        aria-hidden="true"
      />
    </button>
  );
}

function Segmented({
  options,
  value,
  disabled,
  label,
  onChange,
}: {
  options: { value: ControlValue; label: string }[];
  value: ControlValue;
  disabled?: boolean;
  label: string;
  onChange: (next: ControlValue) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex flex-wrap rounded"
      style={{ border: "1px solid var(--border)", overflow: "hidden" }}
    >
      {options.map((opt, i) => {
        const active = opt.value === value;
        return (
          <button
            key={String(opt.value)}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled}
            onClick={() => onChange(opt.value)}
            className="px-2.5 py-1 text-xs font-medium transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: active ? "var(--accent-dim)" : "var(--surface)",
              color: active ? "var(--accent)" : "var(--muted-bright)",
              borderLeft: i === 0 ? "none" : "1px solid var(--border)",
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function ControlWidget({
  spec,
  value,
  disabled,
  onChange,
}: {
  spec: ControlSpec;
  value: ControlValue;
  disabled?: boolean;
  onChange: (next: ControlValue) => void;
}) {
  if (spec.type === "bool") {
    return (
      <Toggle
        checked={value === true}
        disabled={disabled}
        label={spec.label}
        onChange={onChange}
      />
    );
  }
  if (spec.type === "choice") {
    return (
      <Segmented
        label={spec.label}
        value={value}
        disabled={disabled}
        onChange={onChange}
        options={spec.choices.map((c) => ({ value: c, label: c }))}
      />
    );
  }
  // int: a segmented range over [min, max]
  const [lo, hi] = spec.int_range.length === 2 ? spec.int_range : [0, 0];
  const opts: { value: ControlValue; label: string }[] = [];
  for (let n = lo; n <= hi; n++) opts.push({ value: n, label: String(n) });
  return (
    <Segmented
      label={spec.label}
      value={value}
      disabled={disabled}
      onChange={onChange}
      options={opts}
    />
  );
}

function RuntimeRow({
  spec,
  value,
  pending,
  error,
  onChange,
}: {
  spec: ControlSpec;
  value: ControlValue;
  pending: boolean;
  error?: string;
  onChange: (next: ControlValue) => void;
}) {
  return (
    <div
      className="py-4"
      style={{ borderTop: "1px solid var(--border-subtle)" }}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium" style={{ color: "var(--ink)" }}>
              {spec.label}
            </span>
            <ScopeBadge scope={spec.scope} />
          </div>
          <p className="mt-0.5 text-xs" style={{ color: "var(--muted)" }}>
            {spec.description}
          </p>
          <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
            {spec.cost}
          </p>
        </div>
        <div className="shrink-0">
          <ControlWidget
            spec={spec}
            value={value}
            disabled={pending}
            onChange={onChange}
          />
        </div>
      </div>
      {error && (
        <p className="mt-2 text-xs" style={{ color: "var(--error)" }} role="alert">
          {error}
        </p>
      )}
      <TradeOffs spec={spec} />
    </div>
  );
}

function InfoRow({ spec, value }: { spec: ControlSpec; value: ControlValue }) {
  return (
    <div
      className="py-4"
      style={{ borderTop: "1px solid var(--border-subtle)" }}
    >
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium" style={{ color: "var(--ink)" }}>
          {spec.label}
        </span>
        <ScopeBadge scope={spec.scope} />
      </div>
      <p className="mt-0.5 text-xs" style={{ color: "var(--muted)" }}>
        {spec.description}
      </p>
      <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
        <span style={{ color: "var(--muted-bright)" }}>current: </span>
        <code style={{ fontFamily: "var(--font-mono)" }}>{String(value)}</code>
        <span> &middot; {spec.cost}</span>
      </p>
      <TradeOffs spec={spec} />
    </div>
  );
}

export function SettingsView() {
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [settings, setSettings] = useState<ControlsSettings>({});
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [pendingPreset, setPendingPreset] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      const { settings: s, schema } = await getControls();
      setSettings(s);
      setState({ kind: "ready", schema });
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

  const flashSaved = useCallback(() => {
    setSaved(true);
    window.setTimeout(() => setSaved(false), 1600);
  }, []);

  const applyPreset = useCallback(
    async (id: string) => {
      setPendingPreset(id);
      setRowErrors({});
      try {
        const { settings: s } = await postControls({ preset: id });
        setSettings(s);
        flashSaved();
      } catch (err) {
        setRowErrors({
          _preset: err instanceof Error ? err.message : String(err),
        });
      } finally {
        setPendingPreset(null);
      }
    },
    [flashSaved],
  );

  const changeControl = useCallback(
    async (id: string, value: ControlValue) => {
      setPendingId(id);
      const prev = settings;
      setSettings((s) => ({ ...s, [id]: value }));
      try {
        const { settings: s, errors } = await postControls({ [id]: value });
        if (errors[id]) {
          setSettings(prev);
          setRowErrors((e) => ({ ...e, [id]: errors[id] }));
        } else {
          setSettings(s);
          setRowErrors((e) => {
            const next = { ...e };
            delete next[id];
            return next;
          });
          flashSaved();
        }
      } catch (err) {
        setSettings(prev);
        setRowErrors((e) => ({
          ...e,
          [id]: err instanceof Error ? err.message : String(err),
        }));
      } finally {
        setPendingId(null);
      }
    },
    [settings, flashSaved],
  );

  const schema = state.kind === "ready" ? state.schema : null;

  const runtimeControls = useMemo(
    () => (schema ? schema.controls.filter((c) => c.scope === "runtime") : []),
    [schema],
  );
  const otherControls = useMemo(
    () => (schema ? schema.controls.filter((c) => c.scope !== "runtime") : []),
    [schema],
  );

  const activePreset = useMemo(() => {
    if (!schema) return null;
    for (const p of schema.presets) {
      const match = Object.entries(p.values).every(
        ([k, v]) => settings[k] === v,
      );
      if (match) return p.id;
    }
    return null;
  }, [schema, settings]);

  return (
    <section className="flex w-full max-w-2xl flex-col gap-6 p-6 mx-auto">
      <div>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-semibold" style={{ color: "var(--ink)" }}>
            Settings
          </h1>
          <span
            className="text-xs transition-opacity duration-150"
            style={{ color: "var(--success)", opacity: saved ? 1 : 0 }}
            role="status"
            aria-live="polite"
          >
            {saved ? "Saved" : ""}
          </span>
        </div>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          Tune how memory is retrieved and served. The verified-memory recall
          gate ships on by default.
        </p>
      </div>

      {state.kind === "loading" && (
        <div className="flex flex-col gap-3" aria-busy="true" aria-label="Loading settings">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      )}

      {state.kind === "error" && (
        <ErrorBanner message={state.message} onRetry={() => void load()} />
      )}

      {schema && (
        <>
          {/* Presets */}
          <div>
            <h2
              className="mb-2 text-xs font-semibold uppercase tracking-wide"
              style={{ color: "var(--muted-bright)" }}
            >
              Presets
            </h2>
            <div
              role="group"
              aria-label="Presets"
              className="grid gap-3 sm:grid-cols-3"
            >
              {schema.presets.map((p) => {
                const active = activePreset === p.id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => void applyPreset(p.id)}
                    aria-pressed={active}
                    disabled={pendingPreset !== null}
                    className="rounded-lg p-4 text-left transition-colors duration-150 disabled:opacity-60"
                    style={{
                      background: active ? "var(--accent-dim)" : "var(--surface)",
                      border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className="text-sm font-semibold"
                        style={{ color: active ? "var(--accent)" : "var(--ink)" }}
                      >
                        {p.label}
                      </span>
                      {active && (
                        <span className="text-xs" style={{ color: "var(--accent)" }}>
                          active
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                      {p.description}
                    </p>
                  </button>
                );
              })}
            </div>
            {rowErrors._preset && (
              <p className="mt-2 text-xs" style={{ color: "var(--error)" }} role="alert">
                {rowErrors._preset}
              </p>
            )}
          </div>

          {/* Advanced */}
          <div
            className="rounded-lg"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
            }}
          >
            <button
              type="button"
              onClick={() => setAdvancedOpen((v) => !v)}
              aria-expanded={advancedOpen}
              aria-controls="advanced-controls"
              className="flex w-full items-center justify-between gap-2 px-4 py-3 text-left"
            >
              <span className="text-sm font-medium" style={{ color: "var(--ink)" }}>
                Advanced
              </span>
              <span
                className="text-xs transition-transform duration-150"
                style={{
                  color: "var(--muted)",
                  transform: advancedOpen ? "rotate(90deg)" : "rotate(0deg)",
                }}
                aria-hidden="true"
              >
                &#9654;
              </span>
            </button>

            {advancedOpen && (
              <div id="advanced-controls" className="px-4 pb-2">
                {runtimeControls.map((spec) => (
                  <RuntimeRow
                    key={spec.id}
                    spec={spec}
                    value={settings[spec.id]}
                    pending={pendingId === spec.id || pendingPreset !== null}
                    error={rowErrors[spec.id]}
                    onChange={(v) => void changeControl(spec.id, v)}
                  />
                ))}

                <div
                  className="mt-4 pt-3"
                  style={{ borderTop: "1px solid var(--border)" }}
                >
                  <h3
                    className="text-xs font-semibold uppercase tracking-wide"
                    style={{ color: "var(--muted-bright)" }}
                  >
                    Set at install or in your answer-gen
                  </h3>
                  <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                    These are not live toggles. The embedder and quantization are
                    store-level choices that re-index existing memories;
                    self-verification runs in your own answer-generation.
                  </p>
                </div>
                {otherControls.map((spec) => (
                  <InfoRow key={spec.id} spec={spec} value={settings[spec.id]} />
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
