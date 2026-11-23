### Fixed

- MCP server now supports mcp 2.x SDK: `FastMCP` was renamed to `MCPServer`
  (`from mcp.server.mcpserver import MCPServer`), with automatic fallback to
  `mcp.server.fastmcp.FastMCP` for mcp 1.x. The `mcp` extra is pinned to
  `mcp>=1.0,<3` so `pip install taosmd[mcp]` works with both major versions.
