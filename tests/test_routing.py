"""Supervisor routing rules."""

from __future__ import annotations

from tripmate.graph.routing import (
    AGENT_ORDER,
    GUARDRAIL_BLOCKED,
    ITINERARY,
    normalize_agents,
    route_after_agent,
    route_from_supervisor,
)


def state(**overrides):
    return {"selected_agents": [], "guardrail_allowed": True, **overrides}


def test_itinerary_is_always_appended() -> None:
    assert normalize_agents(["flight_agent"]) == ["flight_agent", ITINERARY]
    assert normalize_agents([]) == [ITINERARY]


def test_unknown_agents_are_dropped_and_order_is_canonical() -> None:
    selected = normalize_agents(["budget_agent", "root_agent", "flight_agent"])
    assert selected == ["flight_agent", "budget_agent", ITINERARY]


def test_normalize_agents_tolerates_garbage_input() -> None:
    assert normalize_agents(None) == [ITINERARY]
    assert normalize_agents("flight_agent") == [ITINERARY]


def test_blocked_requests_route_away_from_the_specialists() -> None:
    blocked = state(guardrail_allowed=False, selected_agents=["flight_agent"])
    assert route_from_supervisor(blocked) == GUARDRAIL_BLOCKED


def test_supervisor_starts_at_the_first_selected_agent() -> None:
    assert (
        route_from_supervisor(state(selected_agents=["hotel_agent", "flight_agent"]))
        == "flight_agent"
    )


def test_supervisor_falls_back_to_itinerary_when_nothing_is_selected() -> None:
    assert route_from_supervisor(state()) == ITINERARY


def test_chain_skips_unselected_specialists() -> None:
    route = route_after_agent("flight_agent")
    selected = state(selected_agents=["flight_agent", "budget_agent", ITINERARY])
    assert route(selected) == "budget_agent"


def test_chain_ends_at_the_itinerary_agent() -> None:
    route = route_after_agent("budget_agent")
    assert route(state(selected_agents=["flight_agent", "budget_agent"])) == ITINERARY


def test_every_routing_target_is_a_real_node() -> None:
    from tripmate.graph.workflow import node_names

    nodes = set(node_names())
    assert set(AGENT_ORDER) <= nodes
    assert GUARDRAIL_BLOCKED in nodes
