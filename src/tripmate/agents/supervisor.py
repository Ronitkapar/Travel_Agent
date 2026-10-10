"""Supervisor agent: guardrail gate plus specialist selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

from ..graph.routing import AGENT_ORDER, normalize_agents
from ..graph.state import TravelState, empty_constraints
from ..logging_config import get_logger
from ..services.llm import chat, parse_json_object
from .guardrails import blocked_state, evaluate_guardrail

logger = get_logger(__name__)

SUPERVISOR_NODE = "supervisor"

AGENT_DESCRIPTIONS = """- flight_agent: flights, airports, airlines, routes, airfare, or
  booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included"""

SYSTEM_PROMPT = (
    "You route work to travel specialist agents. Return strict JSON only."
)

FALLBACK_REASONING = (
    "Supervisor parsing failed, so the full travel workflow was selected "
    "as a safe fallback."
)


@dataclass(frozen=True, slots=True)
class SupervisorPlan:
    selected_agents: list[str] = field(default_factory=list)
    trip_constraints: dict[str, Any] = field(default_factory=empty_constraints)
    reasoning: str = ""


async def plan_trip(query: str) -> SupervisorPlan:
    prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
{AGENT_DESCRIPTIONS}

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent",
                     "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        raw = await chat(SYSTEM_PROMPT, prompt)
        parsed = parse_json_object(raw)
    except Exception as exc:
        # The original workflow (every specialist) is the safe fallback.
        logger.warning("Supervisor fallback used: %s", exc)
        return SupervisorPlan(
            selected_agents=list(AGENT_ORDER),
            reasoning=FALLBACK_REASONING,
        )

    constraints = empty_constraints()
    parsed_constraints = parsed.get("trip_constraints", {})
    if isinstance(parsed_constraints, dict):
        constraints.update(parsed_constraints)

    return SupervisorPlan(
        selected_agents=normalize_agents(parsed.get("selected_agents", [])),
        trip_constraints=constraints,
        reasoning=str(parsed.get("reasoning", "")).strip(),
    )


async def supervisor_node(state: TravelState) -> dict:
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    decision = await evaluate_guardrail(query)
    if decision.allowed:
        llm_calls += 1

    if not decision.allowed:
        return {**blocked_state(decision), "llm_calls": llm_calls}

    plan = await plan_trip(query)
    llm_calls += 1

    return {
        "guardrail_allowed": True,
        "guardrail_reason": decision.reason,
        "selected_agents": plan.selected_agents,
        "trip_constraints": plan.trip_constraints,
        "supervisor_reasoning": plan.reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }
