### Fixed
- `pip install taosmd[mcp]` no longer installs mcp 2.x by accident: the extra is now pinned `mcp>=1,<2`. mcp 2.x renamed and moved `FastMCP`, so `from mcp.server.fastmcp import FastMCP` raised `ModuleNotFoundError` and `taosmd mcp` failed on a fresh install.
- `taosmd mcp` now reports an incompatible installed `mcp` version as a one-line `incompatible version` error (`IncompatibleMCPVersionError`) with a non-zero exit instead of a traceback, and instead of telling the user to install an SDK they already have.
