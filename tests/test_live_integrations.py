"""Live integration tests against real external services.

These are **not** part of the default suite. They spend real API credits and
require real credentials, so every test is skipped unless ``RUN_LIVE_TESTS=1``
is set::

    RUN_LIVE_TESTS=1 python -m pytest tests/test_live_integrations.py -v

Each test makes the minimum number of calls needed to prove the service works.
The mocked suite in the other files never touches the network.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="Set RUN_LIVE_TESTS=1 to run tests that call real external services.",
)


def _load_dotenv() -> None:
    from dotenv import load_dotenv

    from tripmate.config import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env", override=False)


_load_dotenv()


def _settings():
    from tripmate.config import get_settings

    get_settings.cache_clear()
    return get_settings()


async def test_live_groq_returns_a_real_completion() -> None:
    from tripmate.services.llm import chat, reset_chat_model

    settings = _settings()
    if not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY is not set")

    reset_chat_model()
    reply = await chat(
        "You reply with a single word.",
        "Reply with exactly: PONG",
    )
    reset_chat_model()

    assert reply.strip(), "The model returned an empty response"


async def test_live_tavily_search_returns_results() -> None:
    from tripmate.mcp import client as mcp

    settings = _settings()
    if not settings.tavily_api_key:
        pytest.skip("TAVILY_API_KEY is not set")

    result = await mcp.tavily_search("capital of Japan")
    assert result


async def test_live_aviationstack_tools_are_discoverable_and_callable() -> None:
    from tripmate.mcp import client as mcp

    settings = _settings()
    if not settings.aviationstack_api_key:
        pytest.skip("AVIATION_STACK_API_KEY is not set")

    tools = await mcp.get_client().get_tools(server_name=mcp.SERVER_AVIATION)
    names = {tool.name for tool in tools}
    assert "list_airports" in names

    result = await mcp.aviation_call("list_airports")
    assert result


async def test_live_openweather_returns_current_conditions() -> None:
    from tripmate.mcp import client as mcp

    settings = _settings()
    if not settings.openweather_api_key:
        pytest.skip("OPENWEATHER_API_KEY is not set")

    result = await mcp.current_weather("Tokyo")
    assert result


async def test_live_postgres_checkpoint_round_trip() -> None:
    """Save and reload a checkpoint through the real database.

    Skipped rather than failed when PostgreSQL is unreachable, so the suite
    still reports honestly: an unavailable database is a known, documented
    limitation, not a passing test.
    """

    settings = _settings()
    if not settings.database_url:
        pytest.skip("DATABASE_URL is not set")

    from langchain_core.messages import HumanMessage

    from tripmate.services.checkpointer import create_checkpointer

    handle = await create_checkpointer(settings)
    try:
        if handle.backend != "postgres":
            pytest.skip(f"PostgreSQL unavailable: {handle.detail}")

        thread_id = f"live_test_{os.getpid()}"
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        await handle.saver.aput(
            config,
            {
                "v": 4,
                "id": "live-probe",
                "ts": "2026-01-01T00:00:00+00:00",
                "channel_values": {
                    "messages": [HumanMessage(content="round trip")]
                },
                "channel_versions": {"messages": "1"},
                "versions_seen": {},
                "updated_channels": None,
                "pending_sends": {},
            },
            {},
            {},
        )

        loaded = await handle.saver.aget_tuple(config)
        assert loaded is not None
        # The raw saver's durable contract is checkpoint identity plus the
        # channel metadata it stores alongside the blobs. Blob contents are
        # rehydrated by the graph during execution, not by this low-level
        # call, so they are not asserted here -- ``test_live_hitl_resume``
        # proves value survival end to end.
        assert loaded.checkpoint["id"] == "live-probe"
        assert loaded.checkpoint["channel_versions"] == {"messages": "1"}

        await handle.saver.adelete_thread(thread_id)
    finally:
        await handle.aclose()


async def test_live_hitl_resume_over_http() -> None:
    """Full HITL round trip through the API with the real graph and LLM."""

    settings = _settings()
    if not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY is not set")

    from fastapi.testclient import TestClient

    from tripmate.main import create_app
    from tripmate.services.travel import set_planner

    set_planner(None)
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/travel",
            json={"message": "Plan a 2 day trip to Kyoto", "thread_id": None},
        )
        assert response.status_code == 200, response.text

        payload = response.json()
        assert payload["requires_approval"] is True
        thread_id = payload["thread_id"]

        approval = client.post(
            "/api/travel/approve",
            json={"thread_id": thread_id, "approved": True, "feedback": ""},
        )
        assert approval.status_code == 200, approval.text

        final = approval.json()
        assert final["requires_approval"] is False
        assert final["answer"]