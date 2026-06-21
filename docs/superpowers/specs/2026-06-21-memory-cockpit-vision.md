# taOSmd Memory Cockpit and Capture Lanes (vision and roadmap)

This is a decomposition document, not a single implementation spec. It captures the shape of a large effort and the order to build it in. Each phase below gets its own design spec and implementation plan before any code.

## Goal

Grow taOSmd from a local-first agent-memory backend into a local-first personal-and-work memory engine with a polished cockpit. Inspiration: Memoire (mymemoire.ai) for the cockpit and the share-to-collect idea, ArcRift for the knowledge-graph galaxy, and PiecesOS for the ambient capture layer. We adapt these to taOSmd's identity rather than cloning them.

## Positioning (why this is differentiated, not a clone)

The reference products are either closed or cloud-leaning. PiecesOS captures workflow context locally but its memory is a black box: you query it and get summaries, with no way to see what was captured, trace it to a source, or audit it. Memoire connects to consumer apps through cloud APIs.

taOSmd's identity is the opposite and it is the wedge for all of this:

- Local-first and offline, runs on an 8GB SBC or NPU, can be a shared local hub rather than one machine.
- Zero-loss: every captured item is kept verbatim in an append-only archive.
- Provable and auditable: every extracted fact links back to its source span, a verifier checks it, and unsupported facts are demoted rather than served.
- Open (MIT).

So "taOSmd with a capture layer and a cockpit" is the open, auditable, zero-loss alternative to Pieces, Memoire, and Rewind at once.

## Architecture: three capture lanes, one engine, one cockpit

Three lanes feed the one existing provable store (archive plus vector plus temporal knowledge graph plus claims), and the cockpit reads from it.

1. Active share-to-collect. A browser and desktop extension with a "share to taOSmd" action. The user shares a link, document, or video; it is routed into a source-typed collection (an Amazon collection, a YouTube collection, and so on) and run through a per-type enrichment pipeline. This builds on the Collections concept already on the roadmap.
2. Ambient capture (PiecesOS-style, opt-in). A local service that watches active windows, browser, and IDE via OCR and window context, builds passive memory, with per-app capture toggles, local encryption, and air-gap support. The most ambitious and most privacy-sensitive lane, so it is last.
3. Agent and conversation memory. What taOSmd already is: agents, projects, captured AI chats, and the MCP, HTTP, and API surfaces.

All three land in the same zero-loss, provable engine, so provenance and verification apply uniformly no matter how a memory arrived.

## The cockpit surfaces

The standalone dashboard becomes the cockpit. Surfaces, mapped to taOSmd data:

- Home / Overview: stat cards (total memories, sources, documents, connected assistants), a memory-growth chart, top categories, and a recent-activity feed. Leads with our distinctive number: the share of facts verified against their source.
- Sources: the source-typed collections from lane 1, each with its memory count and a detail panel (synced status, generated memories with confidence and source span, and per-assistant permission toggles).
- Memory Explorer: a D3 force-directed view of the temporal knowledge graph (subject, relation, object triples), nodes sized by connectivity, hover and zoom and drag, bundled and offline. This is the ArcRift galaxy over our own graph.
- Documents: ingested documents and collections.
- AI Assistants: which assistants can access which memory, with per-assistant permissions and query monitoring, building on per-agent scoping and the access tracker.

## Per-type enrichment (lane 1 detail, for later spec)

The share-to-collect pipeline is a set of per-source-type processors over a common interface, so new types are additive:

- Generic link: readability text extraction plus a screenshot, user's choice of either or both.
- Video (YouTube and similar): title, creator, publish date, description, and link; transcript downloaded or generated; optional media download; a comments scan that flags corrections or updated information.
- Product (Amazon and similar): product info, price, and a screenshot.

Processing depth is the user's choice per share (screenshot only, text only, both, or full download).

## Theming and macOS styling

taOS already ships a macOS-like desktop with a real theme system at `tinyagentos/desktop/src/theme/` (`tokens.css`, `theme-config.ts`, and a dark/light theme store). That is the source of truth. The cockpit adopts those tokens and the dark and light themes rather than inventing a palette, so taOSmd and the taOS desktop feel like one product.

## Constraints (apply to every phase)

- Offline and self-contained: no runtime CDN. Charts and the graph view are bundled into the Vite build (hand-rolled SVG for charts, D3 for the graph).
- Accessible: ARIA, labels, keyboard navigation, reduced-motion, as the current dashboard already does.
- Provable and zero-loss preserved: every new capture path keeps the source verbatim and carries it into the claims and verification flow.
- Privacy-first for capture: per-app and per-source toggles, local processing, nothing leaves the machine unless the user shares it.
- Managed mode respected: when taOS owns the install, the standalone write surfaces defer to the taOS app.

## Phased roadmap

Each phase is independently useful and is specced and planned on its own. Build order:

0. Theming foundation. Adopt the taOS desktop tokens, add the dark and light themes and a theme toggle, and restyle the existing dashboard to the macOS-like language. Prerequisite for everything visible.
1. Home / Overview. New aggregate stats endpoints over the existing archive, access-tracker, and claims data, plus the Overview tab (stat cards, memory-growth chart, top categories, recent activity).
2. Memory Explorer. The D3 knowledge-graph galaxy over the temporal KG.
3. Share-to-collect. The browser extension share action, source-typed collections, and the per-type enrichment framework (ship generic link plus the first rich type, video).
4. Sources and permissions. The Sources cockpit surface and per-assistant permission management.
5. Documents. The Documents surface.
6. Ambient capture. The opt-in PiecesOS-style local capture service.

The browser extension built in phase 3 is shared infrastructure that phase 6 extends.

Phases 0 and 1 are the current target.

## Out of scope (for now)

- Consumer-app API connectors (the Memoire model). We use share-to-collect instead.
- Cloud sync and any hosted option. taOSmd stays local-first.
- Pricing, tiers, and business model (these live outside the public repo).

## Open questions (resolved per phase, not here)

- What "categories" means for the Overview top-categories chart (agents and projects, claim types, or a new categorization).
- How taOS token names map onto the dashboard's current token names.
- The exact capture techniques and per-OS permissions for the ambient lane (phase 6).
