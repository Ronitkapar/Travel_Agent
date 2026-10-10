"""Supervisor routing rules."""

from __future__ import annotations

from collections.abc import Callable

from .state import TravelState

# Canonical execution order. The itinerary agent is always last because it
# integrates the output of whichever specialists were selected.
AGENT_ORDER = (
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
)
KNOWN_AGENTS = frozenset(AGENT_ORDER)

GUARDRAIL_BLOCKED = "guardrail_blocked"
ITINERARY = "itinerary_agent"

ROUTE_MAP = {
    GUARDRAIL_BLOCKED: GUARDRAIL_BLOCKED,
    **{agent: agent for agent in AGENT_ORDER},
}


def normalize_agents(requested: object) -> list[str]:
    """Keep only known agents, in canonical order, with itinerary always present."""

    requested_set = requested if isinstance(requested, (set, list, tuple)) else []

    selected = [agent for agent in AGENT_ORDER if agent in requested_set]

    if ITINERARY not in selected:
        selected.append(ITINERARY)

    return selected


def selected_agents(state: TravelState) -> list[str]:
    return [agent for agent in AGENT_ORDER if agent in state.get("selected_agents", [])]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return GUARDRAIL_BLOCKED

    selected = selected_agents(state)
    return selected[0] if selected else ITINERARY


def route_after_agent(current_agent: str) -> Callable[[TravelState], str]:
    """Jump to the next selected agent, skipping unselected specialists."""

    current_index = AGENT_ORDER.index(current_agent)

    def route(state: TravelState) -> str:
        selected = set(selected_agents(state))

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return ITINERARY

    return route
