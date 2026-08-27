### Fixed

- Removed the dead `importlib.resources` try-block from `_webui_dir()` in
  `taosmd/http_server.py` and corrected its docstring. The function always
  resolved `webui` relative to `__file__`, which works for both wheel and
  source installs, so the unreachable importlib path and its load-bearing
  dummy assignment have been deleted. Added a unit test that proves the
  path-resolution uses the `__file__`-relative candidate.
