"""Optional Pydantic Logfire tracing.

Tracing is off unless ``LOGFIRE_TOKEN`` is set, so the app always starts. When
enabled, :func:`configure_observability` installs three instrumentors:

* FastAPI -- one span per request with status and latency.
* psycopg  -- one span per SQL statement.
* httpx    -- outbound HTTP, which covers the Groq and Tavily clients.

LangGraph, LangChain, and Groq have no maintained Logfire instrumentor, so LLM
calls and MCP tool calls are recorded manually in :mod:`tripmate.services.llm`
and :mod:`tripmate.mcp.client` through :func:`traced_span`.

No prompt text, tool responses, credentials, or connection strings are recorded.
Spans carry only dimensions (model, server name, tool name, token counts), and
:func:`scrub` shortens free-form error text before it leaves the process.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .config import Settings, get_settings
from .logging_config import get_logger, redact_url

logger = get_logger(__name__)

MAX_ERROR_TEXT = 200

_CONFIGURED = False

# Never let a span carry a secret or an unbounded blob, whatever a caller
# passes in.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "conninfo",
        "conn_info",
        "conn_str",
        "connection_string",
        "credential",
        "database_url",
        "env",
        "key",
        "password",
        "prompt",
        "secret",
        "system_prompt",
        "token",
        "user_prompt",
    }
)


def scrub(value: object, limit: int = MAX_ERROR_TEXT) -> str:
    """Reduce free-form text to a short, single-line summary."""

    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _safe_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop sensitive keys and clamp long values."""

    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if key.lower() in _SENSITIVE_KEYS:
            continue
        if isinstance(value, str):
            safe[key] = scrub(value)
        else:
            safe[key] = value
    return safe


@contextlib.contextmanager
def traced_span(
    name: str, enabled: bool = True, **attributes: Any
) -> Iterator[None]:
    """Record a span when tracing is on, and do nothing when it is off.

    The no-op path avoids importing or calling logfire at all so that tracing
    stays free for deployments that have not enabled it.
    """

    if not enabled or not _CONFIGURED:
        yield
        return

    import logfire

    with logfire.span(name, **_safe_attributes(attributes)):
        yield


@contextlib.asynccontextmanager
async def traced_async_span(
    name: str, enabled: bool = True, **attributes: Any
) -> AsyncIterator[None]:
    """Async counterpart to :func:`traced_span`.

    ``logfire.span`` is a synchronous context manager that only records on
    enter/exit, so it does not block the event loop.
    """

    if not enabled or not _CONFIGURED:
        yield
        return

    import logfire

    with logfire.span(name, **_safe_attributes(attributes)):
        yield


def is_configured() -> bool:
    return _CONFIGURED


def _redact_httpx_url(span: Any, request: Any) -> None:
    """Overwrite the instrumentor's captured URL with a masked one.

    ``instrument_httpx`` records the full request URL. The Tavily MCP server
    carries its credential in the query string, so the raw attribute would ship
    a live API key into the trace backend. The hook runs after the instrumentor
    sets attributes, so overwriting here is what actually takes effect.
    """

    url = getattr(request, "url", None)
    if url is None:
        return
    try:
        safe = redact_url(str(url))
    except Exception:
        safe = "<unredactable-url>"
    for attribute in ("url.full", "url"):
        with contextlib.suppress(Exception):
            span.set_attribute(attribute, safe)


def configure_observability(
    app: Any | None = None, settings: Settings | None = None
) -> bool:
    """Configure Logfire and instrument ``app``.

    Returns ``True`` when tracing is active. Any failure is logged and
    swallowed: observability must never stop the app from serving traffic.
    """

    global _CONFIGURED

    settings = settings or get_settings()

    if not settings.tracing_enabled:
        logger.info("Logfire tracing disabled (LOGFIRE_TOKEN is not set).")
        return False

    if _CONFIGURED:
        return True

    try:
        import logfire

        logfire.configure(
            token=settings.logfire_token,
            service_name="tripmate-ai",
            service_version=_version(),
            environment=settings.logfire_environment,
            # Prompt bodies are never captured; console output would otherwise
            # duplicate spans into stdout on every request.
            console=False,
            send_to_logfire=True,
        )

        logfire.instrument_httpx(
            capture_all=False,
            request_hook=_redact_httpx_url,
        )
        logfire.instrument_psycopg()

        if app is not None:
            logfire.instrument_fastapi(app, capture_headers=False)

        _CONFIGURED = True
        logger.info(
            "Logfire tracing enabled (environment=%s).",
            settings.logfire_environment,
        )
        return True
    except Exception:
        logger.exception(
            "Logfire configuration failed; continuing without tracing."
        )
        _CONFIGURED = False
        return False


def _version() -> str:
    from . import __version__

    return __version__


def shutdown() -> None:
    """Flush pending spans. Called during app shutdown."""

    global _CONFIGURED

    if not _CONFIGURED:
        return

    try:
        import logfire

        logfire.force_flush()
        logfire.shutdown()
    except Exception:
        logger.debug("Logfire shutdown failed", exc_info=True)
    finally:
        _CONFIGURED = False


def reset() -> None:
    """Clear the configured flag. Used by tests."""

    global _CONFIGURED
    _CONFIGURED = False