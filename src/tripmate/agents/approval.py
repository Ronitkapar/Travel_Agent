"""Human-in-the-Loop approval node.

``interrupt()`` must not be wrapped in try/except: LangGraph uses the raised
``GraphInterrupt`` to pause the thread, and swallowing it would resume execution
before the human has answered.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from ..graph.state import TravelState

APPROVAL_NODE = "human_approval"


def build_approval_payload(state: TravelState) -> dict:
    return {
        "question": "Do you approve this itinerary?",
        "draft_itinerary": state.get("itinerary", ""),
        "approval_request": state.get("approval_request", ""),
        "selected_agents": state.get("selected_agents", []),
        "supervisor_reasoning": state.get("supervisor_reasoning", ""),
        "expected_response": {
            "approved": True,
            "feedback": "Optional revision feedback",
        },
    }


async def human_approval_node(state: TravelState) -> dict:
    review = interrupt(build_approval_payload(state))

    review = review if isinstance(review, dict) else {}

    return {
        "approved": bool(review.get("approved", False)),
        "human_feedback": str(review.get("feedback", "")).strip(),
        "messages": [AIMessage(content="Human approval step completed.")],
    }
