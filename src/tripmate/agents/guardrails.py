"""Input guardrail.

Classifies the request as travel-related or not, and short-circuits the graph
when it is not. The check fails *open*: a model or parser hiccup must not take
down travel planning.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage

from ..graph.state import TravelState, empty_constraints
from ..logging_config import get_logger
from ..services.llm import chat, parse_json_object

logger = get_logger(__name__)

GENERIC_BLOCK_REASON = (
    "TripMate AI can only help with travel-planning requests. "
    "Please ask about a destination, flight, hotel, weather, budget, or itinerary."
)
FALLBACK_REASON = "Guardrail validation fallback allowed the request."
BLOCKED_NODE = "guardrail_blocked"


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    allowed: bool
    reason: str


SYSTEM_PROMPT = (
    "You are the input guardrail for a travel-planning application. "
    "Return strict JSON only."
)


async def evaluate_guardrail(query: str) -> GuardrailDecision:
    prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    try:
        raw = await chat(SYSTEM_PROMPT, prompt)
        parsed = parse_json_object(raw)
    except Exception as exc:
        logger.warning("Guardrail fallback used: %s", exc)
        return GuardrailDecision(allowed=True, reason=FALLBACK_REASON)

    return GuardrailDecision(
        allowed=bool(parsed.get("allowed", True)),
        reason=str(parsed.get("reason", "")).strip(),
    )


async def guardrail_blocked_node(state: TravelState) -> dict:
    reason = (
        state.get("final_response")
        or state.get("guardrail_reason")
        or "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


def blocked_state(decision: GuardrailDecision) -> dict:
    reason = decision.reason or GENERIC_BLOCK_REASON

    return {
        "guardrail_allowed": False,
        "guardrail_reason": reason,
        "selected_agents": [],
        "trip_constraints": empty_constraints(),
        "supervisor_reasoning": reason,
        "final_response": reason,
        "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
    }
