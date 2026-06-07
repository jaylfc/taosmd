import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost";
}

export function Button({
  variant = "primary",
  className = "",
  children,
  ...props
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded px-4 py-2 text-sm font-medium transition-[filter,opacity,background] duration-150 disabled:cursor-not-allowed disabled:opacity-40 select-none";

  const styles =
    variant === "primary"
      ? {
          background: "var(--accent)",
          color: "#0b1020",
        }
      : {
          background: "transparent",
          color: "var(--muted-bright)",
          border: "1px solid var(--border)",
        };

  return (
    <button
      className={`${base} ${className}`}
      style={styles}
      {...props}
    >
      {children}
    </button>
  );
}
