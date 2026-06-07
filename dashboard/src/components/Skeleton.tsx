import React from "react";

interface SkeletonProps {
  className?: string;
  count?: number;
}

function SkeletonLine({ className = "" }: { className?: string }) {
  return (
    <div
      className={`skeleton h-3 rounded ${className}`}
      aria-hidden="true"
    />
  );
}

export function Skeleton({ className = "", count = 1 }: SkeletonProps) {
  return (
    <div className={`space-y-2 ${className}`} aria-busy="true" aria-label="Loading">
      {Array.from({ length: count }, (_, i) => (
        <SkeletonLine
          key={i}
          className={i === count - 1 && count > 1 ? "w-3/4" : ""}
        />
      ))}
    </div>
  );
}

export function SkeletonCard() {
  return (
    <div
      className="rounded-lg p-4 space-y-2"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
      aria-busy="true"
      aria-label="Loading"
    >
      <SkeletonLine className="w-full h-3" />
      <SkeletonLine className="w-5/6 h-3" />
      <SkeletonLine className="w-2/3 h-2.5" />
    </div>
  );
}
