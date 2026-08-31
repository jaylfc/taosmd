### Fixed
- MentionStore now opens its SQLite connection with `check_same_thread=False` at
  `taosmd/mentions.py:31`, matching the thread contract pattern used by
  `ReceiptStore` in `taosmd/receipts.py:56`.
- The changelog count of remaining _db.connect call sites is corrected from
  "sixteen" to "seventeen" for all sites in `taosmd/`, with the enumeration
  based on actual call sites rather than a grep for the fixed pattern.
- The remaining 13 _db.connect sites in `taosmd/` are documented as defence in
  depth with no end-to-end coverage; they do not record the thread contract in
  their class docstrings, unlike `receipts.py` and `mentions.py`.
- The overlap with PR #431 (which fixes mentions.py:31) is noted: this change
  completes the thread-safety fix that #431 began at the mentions store level.