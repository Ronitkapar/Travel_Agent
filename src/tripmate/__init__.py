"""TripMate AI: a LangGraph + MCP multi-agent travel planner.

The package is split by responsibility:

* ``config``      environment-driven settings
* ``api``         HTTP layer (schemas, dependencies, routes)
* ``agents``      individual agent implementations (graph nodes)
* ``graph``       state, routing, and workflow assembly
* ``mcp``         MCP client wiring and the bundled weather MCP server
* ``services``    cross-cutting services: LLM access, checkpointers, planner
"""

__all__ = ["__version__"]

__version__ = "2.1.0"
