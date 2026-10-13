"""Shared fixtures.

The LLM and MCP layers are replaced with deterministic fakes so the graph can
be exercised end-to-end without network access or API keys.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from tripmate.config import get_settings
from tripmate.services import llm as llm_service
from tripmate.services import travel as travel_service


class FakeResponse:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason}


class FakeChatModel:
    """Returns canned replies chosen by a marker found in the system prompt."""

    def __init__(
        self,
        replies: dict[str, str] | None = None,
        default: str = "ok",
        finish_reason: str = "stop",
    ) -> None:
        self.replies = replies or {}
        self.default = default
        self.finish_reason = finish_reason
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, messages: Any) -> FakeResponse:
        system, human = messages[0].content, messages[1].content
        self.calls.append((system, human))

        for marker, reply in self.replies.items():
            if marker in system:
                return FakeResponse(reply, self.finish_reason)

        return FakeResponse(self.default, self.finish_reason)


ALLOW_JSON = json.dumps({"allowed": True, "reason": ""})
BLOCK_JSON = json.dumps({"allowed": False, "reason": "Not a travel request."})


def supervisor_json(*agents: str) -> str:
    return json.dumps(
        {
            "selected_agents": list(agents),
            "trip_constraints": {"destination": "Tokyo"},
            "reasoning": "Selected specialists for a Japan trip.",
        }
    )


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch):
    """Run every test against a fresh, credential-populated settings object.

    Skipped entirely for live runs: those must reach real providers with the
    real credentials from the environment, not the placeholders below.
    """

    if os.getenv("RUN_LIVE_TESTS") == "1":
        get_settings.cache_clear()
        llm_service.reset_chat_model()
        yield
        get_settings.cache_clear()
        llm_service.reset_chat_model()
        return

    for key, value in {
        "GROQ_API_KEY": "test-groq-key",
        "TAVILY_API_KEY": "test-tavily-key",
        "AVIATION_STACK_API_KEY": "test-aviation-key",
        "OPENWEATHER_API_KEY": "test-openweather-key",
        "DATABASE_URL": "",
        "LOGFIRE_TOKEN": "",
        "LOG_LEVEL": "warning",
        "CORS_ORIGINS": "http://localhost:3000",
    }.items():
        monkeypatch.setenv(key, value)

    llm_service.reset_chat_model()
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    llm_service.reset_chat_model()


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    model = FakeChatModel(
        replies={
            "input guardrail": ALLOW_JSON,
            "route work to travel specialist agents": supervisor_json(
                "flight_agent", "hotel_agent", "weather_agent", "budget_agent"
            ),
            "destination name only": "Tokyo",
        },
        default="Generated travel content.",
    )
    monkeypatch.setattr(llm_service, "get_chat_model", lambda: model)
    return model


@pytest.fixture
def blocked_llm(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    model = FakeChatModel(replies={"input guardrail": BLOCK_JSON})
    monkeypatch.setattr(llm_service, "get_chat_model", lambda: model)
    return model


@pytest.fixture
async def planner() -> AsyncIterator[travel_service.TravelPlanner]:
    instance = travel_service.TravelPlanner()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.aclose()
