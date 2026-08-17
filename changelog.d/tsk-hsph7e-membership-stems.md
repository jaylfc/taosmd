### Fixed
- Membership-stem measurement now reads from EVENT_A2A archive rows instead of bus-spool senders, producing correct population counts
- Fixed `obj.get("body", "")` returning `None` for JSON `null` body in `_collect_from_bus_spool`
- Fixed file descriptor leak in `_collect_from_bus_spool` by using `with open()` pattern
- Fixed tautological condition `is_bare_form(s) or is_at_form(s)` that was a no-op; now correctly excludes canonical mint-stamped principals from twin detection
- Fixed `async_main` `--data-dir` fallback that silently measured from `~/.taosmd/bus-spool.jsonl` instead of the selected data directory
- Added regression test for two distinct mint-stamped principals asserting `canonical_twins` is empty