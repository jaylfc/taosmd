import React from "react";

interface ErrorBannerProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorBanner({ message, onRetry }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-lg px-4 py-3 text-sm"
      style={{
        background: "rgba(248,113,113,0.08)",
        border: "1px solid rgba(248,113,113,0.2)",
        color: "var(--error)",
      }}
    >
      <span className="mt-0.5 shrink-0" aria-hidden="true">⚠</span>
      <span className="flex-1">{message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="shrink-0 text-xs font-medium underline underline-offset-2"
          style={{ color: "var(--error)" }}
        >
          Retry
        </button>
      )}
    </div>
  );
}
