"""MCP client wiring.

``MultiServerMCPClient`` is created lazily and one server is loaded per call so
a broken weather or AviationStack server cannot take down Tavily search.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import (
    AVATIONSTACK_MCP_PACKAGE,
    TAVILY_MCP_URL,
    get_settings,
)
from ..logging_config import get_logger
from ..observability import traced_span

logger = get_logger(__name__)

SERVER_TAVILY = "tavily"
SERVER_AVIATION = "aviationstack"
SERVER_WEATHER = "weather"


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


def build_connections() -> dict[str, dict[str, Any]]:
    settings = get_settings()
    uvx = shutil.which("uvx") or "uvx"

    return {
        SERVER_TAVILY: {
            "transport": "streamable_http",
            "url": f"{TAVILY_MCP_URL}?tavilyApiKey={settings.tavily_api_key or ''}",
        },
        SERVER_AVIATION: {
            "transport": "stdio",
            "command": uvx,
            "args": [AVATIONSTACK_MCP_PACKAGE],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=settings.aviationstack_api_key
            ),
        },
        SERVER_WEATHER: {
            "transport": "stdio",
            # Run the bundled server with the interpreter that owns this process
            # so it never depends on a separately activated virtualenv.
            "command": sys.executable,
            "args": [str(settings.weather_server_path)],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=settings.openweather_api_key
            ),
        },
    }


_client: MultiServerMCPClient | None = None


def get_client() -> MultiServerMCPClient:
    global _client

    if _client is None:
        _client = MultiServerMCPClient(build_connections())

    return _client


def reset_client() -> None:
    global _client
    _client = None


def _preflight(server_name: str) -> None:
    settings = get_settings()

    if server_name == SERVER_TAVILY and not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    if server_name == SERVER_AVIATION:
        if not settings.aviationstack_api_key:
            raise RuntimeError("AVIATION_STACK_API_KEY is not configured.")
        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found on PATH. Install uv (https://docs.astral.sh/uv/) "
                "so the AviationStack MCP server can be launched."
            )

    if server_name == SERVER_WEATHER:
        if not settings.openweather_api_key:
            raise RuntimeError("OPENWEATHER_API_KEY is not configured.")
        if not settings.weather_server_path.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found at {settings.weather_server_path}"
            )


async def get_tool(server_name: str, tool_name: str) -> BaseTool:
    _preflight(server_name)

    settings = get_settings()
    with traced_span(
        "mcp.get_tool",
        enabled=settings.tracing_enabled,
        server=server_name,
        tool=tool_name,
    ):
        tools = await get_client().get_tools(server_name=server_name)

    tool = next((item for item in tools if item.name == tool_name), None)

    if tool is None:
        available = ", ".join(sorted(item.name for item in tools)) or "none"
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found on server '{server_name}'. "
            f"Available tools: {available}"
        )

    return tool


async def _invoke(
    server_name: str, tool_name: str, tool_args: dict[str, Any]
) -> Any:
    """Discover and call an MCP tool inside a single span.

    Tool arguments and results are deliberately excluded: search results and
    weather payloads are large and can contain user-supplied data.
    """

    settings = get_settings()

    with traced_span(
        "mcp.call",
        enabled=settings.tracing_enabled,
        server=server_name,
        tool=tool_name,
    ):
        tool = await get_tool(server_name, tool_name)
        return await tool.ainvoke(tool_args or {})


async def tavily_search(query: str) -> Any:
    return await _invoke(SERVER_TAVILY, "tavily_search", {"query": query})


async def aviation_call(tool_name: str, tool_args: dict[str, Any] | None = None) -> Any:
    return await _invoke(SERVER_AVIATION, tool_name, tool_args or {})


async def current_weather(city: str) -> Any:
    return await _invoke(SERVER_WEATHER, "get_current_weather", {"city": city})


async def weather_forecast(city: str) -> Any:
    return await _invoke(SERVER_WEATHER, "get_forecast", {"city": city})


async def probe_servers() -> dict[str, str]:
    """Report per-server reachability. Used by the readiness endpoint."""

    report: dict[str, str] = {}

    for server_name in (SERVER_TAVILY, SERVER_AVIATION, SERVER_WEATHER):
        try:
            tools = await get_client().get_tools(server_name=server_name)
            report[server_name] = "ok" if tools else "no tools"
        except Exception as exc:
            report[server_name] = f"error: {type(exc).__name__}"

    return report
