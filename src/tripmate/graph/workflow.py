"""Graph assembly.

``build_graph`` is pure: it returns an uncompiled ``StateGraph`` so the
checkpointer can be chosen by the caller.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..agents import (
    budget_agent,
    final_agent,
    flight_agent,
    guardrail_blocked_node,
    hotel_agent,
    human_approval_node,
    itinerary_agent,
    supervisor_node,
    weather_agent,
)
from .routing import (
    AGENT_ORDER,
    GUARDRAIL_BLOCKED,
    REVIEW_ROUTES,
    ROUTE_MAP,
    route_after_agent,
    route_after_approval,
    route_after_final,
    route_from_supervisor,
)
from .state import TravelState

SUPERVISOR = "supervisor"
FINAL_AGENT = "final_agent"
ITINERARY = "itinerary_agent"
APPROVAL = "human_approval"

# Specialists that chain into the next selected agent, in execution order.
# itinerary_agent is excluded: it has a fixed edge to the approval node.
CHAINED_SPECIALISTS: dict[str, object] = {
    "flight_agent": flight_agent,
    "hotel_agent": hotel_agent,
    "weather_agent": weather_agent,
    "budget_agent": budget_agent,
}


def build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(TravelState)

    builder.add_node(SUPERVISOR, supervisor_node)
    builder.add_node(GUARDRAIL_BLOCKED, guardrail_blocked_node)
    builder.add_node(ITINERARY, itinerary_agent)
    builder.add_node(APPROVAL, human_approval_node)
    builder.add_node(FINAL_AGENT, final_agent)

    for agent, node in CHAINED_SPECIALISTS.items():
        builder.add_node(agent, node)

    builder.add_edge(START, SUPERVISOR)
    builder.add_conditional_edges(
        SUPERVISOR, route_from_supervisor, ROUTE_MAP
    )

    for agent in CHAINED_SPECIALISTS:
        builder.add_conditional_edges(
            agent, route_after_agent(agent), ROUTE_MAP
        )

    builder.add_edge(ITINERARY, APPROVAL)

    # The review loop: approval either ends the run or hands off to the final
    # agent, and the final agent either finishes or returns to review so the
    # reviewer can keep giving feedback until they are satisfied.
    builder.add_conditional_edges(APPROVAL, route_after_approval, REVIEW_ROUTES)
    builder.add_conditional_edges(FINAL_AGENT, route_after_final, REVIEW_ROUTES)
    builder.add_edge(GUARDRAIL_BLOCKED, END)

    return builder


def node_names() -> tuple[str, ...]:
    return (
        SUPERVISOR,
        GUARDRAIL_BLOCKED,
        *CHAINED_SPECIALISTS,
        ITINERARY,
        APPROVAL,
        FINAL_AGENT,
    )


__all__ = ["AGENT_ORDER", "build_graph", "node_names"]
