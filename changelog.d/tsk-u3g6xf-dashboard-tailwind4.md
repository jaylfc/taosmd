### Added

- Migrated the dashboard to Tailwind CSS 4 (`tailwindcss` ^4, `@tailwindcss/postcss`
  ^4) and Vite 8.3.5, replacing the v3 PostCSS config and the v3 `@tailwind`
  entry. Removed `autoprefixer` (v4 handles prefixing). The JS theme in
  `tailwind.config.js` is preserved via `@config "../tailwind.config.js";` in
  `src/index.css`, so every token (semantic colors, the rem font scale with line
  heights, spacing 18/22, transition durations 150/200/250) still produces the
  same utilities. The rebuilt `taosmd/webui/ output is committed alongside.
