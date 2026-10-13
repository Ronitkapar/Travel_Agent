"""Guardrail and supervisor decision logic."""

from __future__ import annotations

import pytest

from tests.conftest import ALLOW_JSON, BLOCK_JSON, FakeChatModel
from tripmate.agents.guardrails import (
    FALLBACK_REASON,
    GENERIC_BLOCK_REASON,
    evaluate_guardrail,
)
from tripmate.agents.supervisor import plan_trip
from tripmate.services import llm as llm_service


def _stub(monkeypatch: pytest.MonkeyPatch, replies: dict[str, str]) -> FakeChatModel:
    model = FakeChatModel(replies=replies)
    monkeypatch.setattr(llm_service, "get_chat_model", lambda: model)
    return model


async def test_travel_request_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, {"input guardrail": ALLOW_JSON})

    decision = await evaluate_guardrail("Plan a trip to Rome")

    assert decision.allowed is True


async def test_unrelated_request_is_blocked_with_the_model_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, {"input guardrail": BLOCK_JSON})

    decision = await evaluate_guardrail("Debug my Postgres query")

    assert decision.allowed is False
    assert decision.reason == "Not a travel request."


async def test_guardrail_fails_open_on_a_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken(FakeChatModel):
        async def ainvoke(self, messages):
            raise RuntimeError("upstream 429")

    monkeypatch.setattr(llm_service, "get_chat_model", lambda: Broken())

    decision = await evaluate_guardrail("Plan a trip to Rome")

    assert decision.allowed is True
    assert decision.reason == FALLBACK_REASON


async def test_guardrail_fails_open_on_unparsable_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, {"input guardrail": "Sure! Here is your trip."})

    decision = await evaluate_guardrail("Plan a trip to Rome")

    assert decision.allowed is True


async def test_block_without_a_reason_uses_the_generic_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, {"input guardrail": '{"allowed": false}'})

    decision = await evaluate_guardrail("Write me code")

    assert decision.allowed is False
    assert not decision.reason
    assert GENERIC_BLOCK_REASON


async def test_planner_always_includes_the_itinerary_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(
        monkeypatch,
        {
            "route work to travel specialist agents":
                '{"selected_agents": ["hotel_agent"]}'
        },
    )

    plan = await plan_trip("A weekend somewhere warm")

    assert plan.selected_agents == ["hotel_agent", "itinerary_agent"]


async def test_planner_preserves_supplied_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(
        monkeypatch,
        {
            "route work to travel specialist agents": (
                '{"selected_agents": [], "trip_constraints": '
                '{"destination": "Lisbon", "budget": "500"}, "reasoning": "cheap city"}'
            )
        },
    )

    plan = await plan_trip("Trip to Lisbon under 500")

    assert plan.trip_constraints["destination"] == "Lisbon"
    assert plan.trip_constraints["budget"] == "500"
    # Unspecified keys keep their empty defaults.
    assert plan.trip_constraints["origin"] == ""
    assert plan.reasoning == "cheap city"


async def test_planner_falls_back_to_the_full_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, {"route work to travel specialist agents": "nope"})

    plan = await plan_trip("Trip to Rome")

    assert plan.selected_agents == [
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    ]
    assert "fallback" in plan.reasoning.lower()


def test_truncate_for_prompt_bounds_input() -> None:
    from tripmate.services.llm import truncate_for_prompt

    assert truncate_for_prompt("short", 100) == "short"

    long_value = "x" * 5000
    bounded = truncate_for_prompt(long_value, 100)
    assert len(bounded) == 100 + len("\n…[truncated]")
    assert bounded.endswith("[truncated]")


async def test_agent_prompts_stay_within_the_groq_token_ceiling(
    monkeypatch,
) -> None:
    """Free-tier Groq keys cap at 8k TPM; oversized prompts fail with a 413.

    Each specialist feeds its output into later prompts, so a verbose MCP
    response must never propagate unbounded into the itinerary or final agent.
    """

    import tripmate.agents.itinerary as itinerary_module
    from tripmate.agents import final_agent, itinerary_agent
    from tripmate.graph.state import initial_state

    captured: dict[str, str] = {}

    async def fake_chat(system_prompt: str, user_prompt: str) -> str:
        captured["prompt"] = user_prompt
        return "generated"

    monkeypatch.setattr(itinerary_module, "chat", fake_chat)

    state = initial_state("Plan a 7 day Japan trip under 2 lakhs")
    state.update(
        {
            "flight_results": "f" * 200_000,
            "hotel_results": "h" * 200_000,
            "weather_results": "w" * 200_000,
            "budget_results": "b" * 200_000,
            "itinerary": "i" * 200_000,
        }
    )

    await itinerary_agent(state)
    assert len(captured["prompt"]) < 20_000

    state["approved"] = True
    await final_agent(state)
    assert len(captured["prompt"]) < 20_000


async def test_completion_limit_is_set_on_the_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq defaults max_tokens to 3072 when unset.

    Left unset, a long itinerary is cut off mid-sentence and returned as a normal
    success response, so the ceiling has to be raised explicitly.
    """

    from tripmate.config import DEFAULT_MAX_COMPLETION_TOKENS, get_settings
    from tripmate.services import llm as module

    model = module._build_chat_model()

    assert model.max_tokens == DEFAULT_MAX_COMPLETION_TOKENS
    assert model.max_tokens > 3072
    assert get_settings().max_completion_tokens == model.max_tokens


async def test_a_length_truncated_reply_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reply the provider cut short must not be passed off as complete."""

    from tripmate.services import llm as module

    model = FakeChatModel(default="1. **Stay hydra", finish_reason="length")
    monkeypatch.setattr(module, "get_chat_model", lambda: model)

    reply = await module.chat("system", "user")

    assert reply.startswith("1. **Stay hydra")
    assert module.TRUNCATION_NOTICE.strip() in reply


async def test_a_complete_reply_is_left_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tripmate.services import llm as module

    model = FakeChatModel(default="1. Stay hydrated.", finish_reason="stop")
    monkeypatch.setattr(module, "get_chat_model", lambda: model)

    assert await module.chat("system", "user") == "1. Stay hydrated."


async def test_a_reply_without_metadata_is_returned_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older providers and stubs may omit response_metadata entirely."""

    from tripmate.services import llm as module

    class Bare:
        content = "plain reply"

    class BareModel:
        async def ainvoke(self, messages):
            return Bare()

    monkeypatch.setattr(module, "get_chat_model", lambda: BareModel())

    assert await module.chat("system", "user") == "plain reply"
