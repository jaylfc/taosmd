// Dark/light scheme handling for the dashboard, matching taOS's data-scheme
// convention. The scheme is applied as document.documentElement.dataset.scheme
// and persisted to localStorage. This is a real served app, so localStorage is
// fine here (it is not a claude.ai artifact).

const KEY = "taosmd-scheme";
export type Scheme = "dark" | "light";

export function getScheme(): Scheme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // localStorage unavailable; fall back to the OS preference.
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function applyScheme(s: Scheme): void {
  document.documentElement.dataset.scheme = s;
}

export function setScheme(s: Scheme): void {
  try {
    localStorage.setItem(KEY, s);
  } catch {
    // Persistence is best-effort; still apply for this session.
  }
  applyScheme(s);
}

export function initTheme(): Scheme {
  const s = getScheme();
  applyScheme(s);
  return s;
}
