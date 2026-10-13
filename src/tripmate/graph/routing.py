"""Supervisor routing rules."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END

from ..config import get_settings
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
HUMAN_APPROVAL = "human_approval"
FINAL_AGENT = "final_agent"
END_NODE = END

ROUTE_MAP = {
    GUARDRAIL_BLOCKED: GUARDRAIL_BLOCKED,
    **{agent: agent for agent in AGENT_ORDER},
}

# Conditional targets for the review loop. Like ROUTE_MAP, this maps each
# destination onto itself; the keys must therefore be exactly the values the
# review routers return, END included.
REVIEW_ROUTES = {
    HUMAN_APPROVAL: HUMAN_APPROVAL,
    FINAL_AGENT: FINAL_AGENT,
    END_NODE: END_NODE,
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


def route_after_approval(state: TravelState) -> str:
    """Decide what happens once the reviewer has answered.

    Approval on a plan that was never revised still needs the final agent to
    polish the draft. Approval on an already-revised plan is finished: the
    reviewer has seen that exact text. Feedback always routes through the final
    agent so the revision is regenerated before the next interruption.
    """

    if state.get("approved", False):
        return END_NODE if state.get("revision_count", 0) else FINAL_AGENT

    return FINAL_AGENT


def route_after_final(state: TravelState) -> str:
    """Return to review, or finish once the reviewer is satisfied.

    ``MAX_REVISIONS`` bounds the loop so a user who keeps asking for changes
    cannot exhaust the graph recursion limit; past the cap the plan is finalized
    on the last revision.
    """

    if state.get("approved", False):
        return END_NODE

    if state.get("revision_count", 0) >= get_settings().max_revisions:
        return END_NODE

    return HUMAN_APPROVAL
