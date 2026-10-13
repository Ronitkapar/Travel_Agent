"""LangGraph state definition."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    revision_count: int
    llm_calls: int


def empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def initial_state(user_query: str) -> TravelState:
    """A fully-populated starting state, so resumed threads never read a missing key."""

    return TravelState(
        messages=[],
        user_query=user_query,
        guardrail_allowed=True,
        guardrail_reason="",
        selected_agents=[],
        trip_constraints=empty_constraints(),
        supervisor_reasoning="",
        flight_results="",
        hotel_results="",
        weather_results="",
        budget_results="",
        itinerary="",
        approval_request="",
        approved=False,
        human_feedback="",
        final_response="",
        revision_count=0,
        llm_calls=0,
    )
