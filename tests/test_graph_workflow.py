"""End-to-end graph behaviour with a stubbed LLM and MCP layer.

These cover the critical path: supervisor selection, guardrail blocking, the
human-in-the-loop interrupt, and resuming with approval or revision feedback.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    ALLOW_JSON,
    FakeChatModel,
    FakeResponse,
    supervisor_json,
)
from tripmate.graph.workflow import build_graph, node_names
from tripmate.services.travel import TravelPlanner


@pytest.fixture(autouse=True)
def _stub_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """No MCP servers are reachable from the test environment."""

    async def _unavailable(*args, **kwargs):
        raise RuntimeError("MCP disabled in tests")

    monkeypatch.setattr("tripmate.mcp.client.aviation_call", _unavailable)
    monkeypatch.setattr("tripmate.mcp.client.tavily_search", _unavailable)
    monkeypatch.setattr("tripmate.mcp.client.current_weather", _unavailable)
    monkeypatch.setattr("tripmate.mcp.client.weather_forecast", _unavailable)


def test_graph_contains_every_expected_node() -> None:
    names = node_names()
    assert names == (
        "supervisor",
        "guardrail_blocked",
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
        "human_approval",
        "final_agent",
    )


async def test_run_pauses_for_approval_and_resume_finalizes(
    planner: TravelPlanner, fake_llm: FakeChatModel
) -> None:
    draft = await planner.start_run("Plan a 7 day trip to Tokyo")

    assert draft["requires_approval"] is True
    assert draft["answer"] == "Generated travel content."
    assert draft["thread_id"].startswith("user_")
    assert draft["selected_agents"] == [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    ]
    assert draft["guardrail_allowed"] is True
    assert draft["trip_constraints"]["destination"] == "Tokyo"

    final = await planner.resume_run(draft["thread_id"], approved=True)

    assert final["requires_approval"] is False
    assert final["approved"] is True
    assert final["answer"] == "Generated travel content."
    assert final["thread_id"] == draft["thread_id"]


async def test_resume_with_feedback_passes_it_to_the_final_agent(
    planner: TravelPlanner, fake_llm: FakeChatModel
) -> None:
    draft = await planner.start_run("Plan a trip to Osaka")
    final = await planner.resume_run(
        draft["thread_id"], approved=False, feedback="Make it cheaper"
    )

    assert final["approved"] is False
    assert final["human_feedback"] == "Make it cheaper"

    final_prompt = fake_llm.calls[-1][1]
    assert "Make it cheaper" in final_prompt


async def test_guardrail_block_short_circuits_the_graph(
    planner: TravelPlanner, blocked_llm: FakeChatModel
) -> None:
    result = await planner.start_run("Write me a Python script that sorts a list")

    assert result["requires_approval"] is False
    assert result["guardrail_allowed"] is False
    assert result["answer"] == "Not a travel request."
    assert result["selected_agents"] == []
    # Blocked before any specialist or approval node ran.
    assert len(blocked_llm.calls) == 1


async def test_guardrail_allows_partial_travel_requests(
    planner: TravelPlanner, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeChatModel(
        replies={
            "input guardrail": ALLOW_JSON,
            "route work to travel specialist agents": supervisor_json("weather_agent"),
        }
    )
    monkeypatch.setattr("tripmate.services.llm.get_chat_model", lambda: model)

    result = await planner.start_run("What's the weather like?")

    assert result["guardrail_allowed"] is True
    assert result["selected_agents"] == ["weather_agent", "itinerary_agent"]


async def test_supervisor_falls_back_to_all_agents_when_parsing_fails(
    planner: TravelPlanner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GarbageSupervisor(FakeChatModel):
        async def ainvoke(self, messages):
            system = messages[0].content
            if "route work to travel specialist agents" in system:
                return FakeResponse("not json at all")
            return FakeResponse(ALLOW_JSON)

    monkeypatch.setattr(
        "tripmate.services.llm.get_chat_model", lambda: GarbageSupervisor()
    )

    result = await planner.start_run("Plan a trip to Kyoto")

    assert result["selected_agents"] == [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    ]
    assert "fallback" in result["supervisor_reasoning"].lower()


async def test_graph_compiles_without_a_checkpointer() -> None:
    """The builder must stay usable for inspection and tests."""

    graph = build_graph()
    assert "supervisor" in graph.nodes
