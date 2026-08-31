### Fixed
- Correct changelog claim: database already had 5000 ms busy timeout (Python sqlite3 default); the actual change was journal mode moving to WAL and connections routing through `_db.connect`.
- Fixed assertion that could not discriminate timeout value; replaced vacuous claim with a test that pins `_db.py:58` busy_timeout PRAGMA.
