import React from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  id: string;
}

export function Input({ label, id, className = "", ...props }: InputProps) {
  return (
    <div className="flex flex-col gap-1">
      <label
        htmlFor={id}
        className="text-xs font-medium"
        style={{ color: "var(--muted-bright)" }}
      >
        {label}
      </label>
      <input
        id={id}
        className={`rounded px-3 py-2 text-sm outline-none transition-[box-shadow] duration-150 ${className}`}
        style={{
          background: "var(--bg)",
          border: "1px solid var(--border)",
          color: "var(--ink)",
          fontFamily: "var(--font-sans)",
        }}
        {...props}
      />
    </div>
  );
}
