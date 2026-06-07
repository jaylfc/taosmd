import React from "react";

interface EmptyStateProps {
  title: string;
  description?: string;
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <p
        className="text-sm font-medium"
        style={{ color: "var(--muted-bright)" }}
      >
        {title}
      </p>
      {description && (
        <p
          className="mt-1 text-sm max-w-xs"
          style={{ color: "var(--muted)" }}
        >
          {description}
        </p>
      )}
    </div>
  );
}
