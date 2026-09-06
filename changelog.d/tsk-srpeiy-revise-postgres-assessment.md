### Fixed

- Revised the Postgres assessment inventory: added the missing `insights.db` store, removed the phantom `pending-decisions.db` (it is the `kg_pending_decisions` table inside `knowledge-graph.db`), corrected `mentions.db` and `receipts.db` to `a2a-mentions.db` and `a2a-receipts.db`, and unified the `vector-memory.db` classification as OPTIONAL / Local-Only.
- Corrected the LIMIT semantics citation from tsk-7wau2y to tsk-2hnss2 and anchored it to `taosmd/archive.py:380` instead of a generic description.
- Replaced the aggregate "All Other Required Stores" verdict with a measured per-store verdict for every row in the inventory table.
- Removed the unsourced "approximately 40 percent higher" maintenance figure and the six-week implementation timeline.
