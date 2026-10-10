"""Full-stack integration: real graph, real routes, HTTP transport.

Only the LLM and MCP boundaries are faked, so this exercises the same code
path the deployed service uses for the human-in-the-loop round trip.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeChatModel
from tripmate.main import create_app
from tripmate.services.travel import TravelPlanner, set_planner


@pytest.fixture
async def live_client(
    monkeypatch: pytest.MonkeyPatch, fake_llm: FakeChatModel
) -> TestClient:
    async def _unavailable(*args, **kwargs):
        raise RuntimeError("MCP disabled in tests")

    monkeypatch.setattr("tripmate.mcp.client.aviation_call", _unavailable)
    monkeypatch.setattr("tripmate.mcp.client.tavily_search", _unavailable)
    monkeypatch.setattr("tripmate.mcp.client.current_weather", _unavailable)
    monkeypatch.setattr("tripmate.mcp.client.weather_forecast", _unavailable)

    planner = TravelPlanner()
    await planner.start()
    set_planner(planner)

    # TestClient's context manager drives the app lifespan, which closes the
    # planner on exit.
    with TestClient(create_app()) as client:
        yield client

    set_planner(None)


async def test_hitl_round_trip_over_http(
    live_client: TestClient, fake_llm: FakeChatModel
) -> None:
    first = live_client.post("/api/travel", json={"message": "Plan a trip to Tokyo"})

    assert first.status_code == 200
    draft = first.json()
    assert draft["success"] is True
    assert draft["requires_approval"] is True

    thread_id = draft["thread_id"]
    assert thread_id

    second = live_client.post(
        "/api/travel/approve",
        json={"thread_id": thread_id, "approved": True, "feedback": ""},
    )

    assert second.status_code == 200
    final = second.json()
    assert final["success"] is True
    assert final["requires_approval"] is False
    assert final["approved"] is True
    assert final["thread_id"] == thread_id
    assert final["answer"]


async def test_guardrail_block_is_returned_without_approval(
    live_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import BLOCK_JSON
    from tests.conftest import FakeChatModel as _M

    model = _M(replies={"input guardrail": BLOCK_JSON})
    monkeypatch.setattr("tripmate.services.llm.get_chat_model", lambda: model)

    response = live_client.post(
        "/api/travel", json={"message": "Write me a bash script"}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["guardrail_allowed"] is False
    assert body["requires_approval"] is False
    assert body["answer"] == "Not a travel request."


async def test_state_is_shared_between_the_two_http_requests(
    live_client: TestClient, fake_llm: FakeChatModel
) -> None:
    """The draft generated in request 1 must reach the final agent in request 2."""

    draft = live_client.post(
        "/api/travel", json={"message": "Plan a trip to Kyoto"}
    ).json()

    live_client.post(
        "/api/travel/approve",
        json={
            "thread_id": draft["thread_id"],
            "approved": False,
            "feedback": "Add a food tour",
        },
    )

    final_prompt = fake_llm.calls[-1][1]
    assert "Add a food tour" in final_prompt
    assert draft["itinerary"] in final_prompt
