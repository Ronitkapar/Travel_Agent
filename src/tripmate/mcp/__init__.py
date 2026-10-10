"""MCP integrations.

``client`` holds the shared multi-server client used by the agents.
``weather_server`` is the bundled stdio server; it is deliberately *not*
imported here so the web process never loads the MCP server runtime.
"""
