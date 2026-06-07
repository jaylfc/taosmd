/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Semantic design-system tokens — evolved from #0f1115 / #6ea8fe palette
        bg: "var(--bg)",
        surface: "var(--surface)",
        border: "var(--border)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        accent: "var(--accent)",
        "error-text": "var(--error)",
        "warning-text": "var(--warning)",
        "success-text": "var(--success)",
        "info-text": "var(--info)",
      },
      fontFamily: {
        sans: [
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SF Mono",
          "Cascadia Mono",
          "Fira Code",
          "Consolas",
          "monospace",
        ],
      },
      fontSize: {
        // Fixed rem scale, ratio ~1.15
        xs: ["0.75rem", { lineHeight: "1.125rem" }],
        sm: ["0.875rem", { lineHeight: "1.25rem" }],
        base: ["1rem", { lineHeight: "1.5rem" }],
        lg: ["1.125rem", { lineHeight: "1.625rem" }],
        xl: ["1.25rem", { lineHeight: "1.75rem" }],
        "2xl": ["1.5rem", { lineHeight: "2rem" }],
      },
      spacing: {
        18: "4.5rem",
        22: "5.5rem",
      },
      transitionDuration: {
        DEFAULT: "150ms",
        200: "200ms",
        250: "250ms",
      },
    },
  },
  plugins: [],
};
