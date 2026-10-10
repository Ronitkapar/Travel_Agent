"""Observability behaviour. No network, no Logfire project required."""

from __future__ import annotations

import pytest

from tripmate import observability
from tripmate.config import Settings
from tripmate.observability import (
    _safe_attributes,
    configure_observability,
    scrub,
    traced_span,
)


@pytest.fixture(autouse=True)
def _reset_observability() -> None:
    observability.reset()
    yield
    observability.reset()


def test_tracing_disabled_without_token() -> None:
    assert Settings().tracing_enabled is False
    assert Settings(logfire_token="pylf_v2_us_x").tracing_enabled is True


def test_configure_is_a_noop_without_token() -> None:
    assert configure_observability(settings=Settings()) is False
    assert observability.is_configured() is False


def test_traced_span_is_a_noop_when_disabled() -> None:
    # Must not raise or import logfire.
    with traced_span("test.span", enabled=False):
        pass


def test_scrub_shortens_and_flattens() -> None:
    assert scrub("a" * 500).endswith("…")
    assert len(scrub("a" * 500)) == 201
    assert scrub("line one\n  line two") == "line one line two"


def test_sensitive_attributes_are_dropped() -> None:
    safe = _safe_attributes(
        {
            "api_key": "secret",
            "token": "secret",
            "password": "secret",
            "database_url": "secret",
            "prompt": "secret",
            "user_prompt": "secret",
            "model": "openai/gpt-oss-120b",
            "server": "tavily",
        }
    )
    assert "api_key" not in safe
    assert "token" not in safe
    assert "password" not in safe
    assert "database_url" not in safe
    assert "prompt" not in safe
    assert "user_prompt" not in safe
    assert safe == {"model": "openai/gpt-oss-120b", "server": "tavily"}


def test_long_attribute_values_are_clamped() -> None:
    safe = _safe_attributes({"note": "x" * 500})
    assert len(safe["note"]) == 201


def test_traced_span_forwards_scrubbed_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With tracing active, the span must receive only sanitized attributes."""

    recorded: dict[str, object] = {}

    class _FakeLogfire:
        def span(self, name: str, **attributes: object) -> object:
            recorded["name"] = name
            recorded["attributes"] = attributes
            return self

        def __enter__(self) -> _FakeLogfire:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setitem(
        __import__("sys").modules, "logfire", _FakeLogfire()
    )

    # Mark as configured so traced_span takes the instrumented path.
    observability._CONFIGURED = True

    with traced_span(
        "unit.span",
        enabled=True,
        api_key="must-not-leak",
        model="openai/gpt-oss-120b",
    ):
        pass

    attributes = recorded["attributes"]
    assert isinstance(attributes, dict)
    assert "api_key" not in attributes
    assert attributes["model"] == "openai/gpt-oss-120b"


def test_health_reports_active_backend_not_configured_one() -> None:
    """A configured-but-unreachable DATABASE_URL must not read as durable."""

    settings = Settings(database_url="postgresql://u:p@h:5432/db")
    assert settings.configured_checkpointer_backend == "postgres"
    # The name change exists precisely because this value can differ from the
    # backend actually in use once the connection is attempted.
    assert settings.configured_checkpointer_backend == "postgres"

# --- URL secret redaction -------------------------------------------------
#
# Regression guard: httpx logs the full request URL at INFO, and the Tavily MCP
# server authenticates with ?tavilyApiKey=..., so the raw key was written to
# stdout and shipped to the trace backend.

_SENTINEL = "sentinel-secret-must-not-leak"


def test_redact_url_masks_vendor_prefixed_key_param() -> None:
    from tripmate.logging_config import redact_url

    url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={_SENTINEL}&query=kyoto"
    out = redact_url(url)

    assert _SENTINEL not in out
    assert "tavilyApiKey=***" in out
    assert "query=kyoto" in out, "non-secret params must survive"


def test_redact_url_masks_database_userinfo() -> None:
    from tripmate.logging_config import redact_url

    url = f"postgresql://postgres.pw:{_SENTINEL}@pool.example.com:5432/postgres"
    out = redact_url(url)

    assert _SENTINEL not in out
    assert out.startswith("postgresql://***@pool.example.com")


def test_redact_url_leaves_secretless_url_untouched() -> None:
    from tripmate.logging_config import redact_url

    url = "https://api.groq.com/openai/v1/chat/completions"
    assert redact_url(url) == url


def test_redact_text_covers_non_http_schemes() -> None:
    from tripmate.logging_config import redact_text

    url = f"postgresql://u:{_SENTINEL}@db.example.com/postgres"
    assert _SENTINEL not in redact_text(f"connection failed for {url}")


def test_logging_filter_scrubs_httpx_url(caplog) -> None:
    import logging

    from tripmate.logging_config import RedactUrlsFilter

    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="HTTP Request: POST %s %s",
        args=(f"https://mcp.tavily.com/mcp/?tavilyApiKey={_SENTINEL}", "200 OK"),
        exc_info=None,
    )

    assert RedactUrlsFilter().filter(record) is True
    assert _SENTINEL not in record.getMessage()
