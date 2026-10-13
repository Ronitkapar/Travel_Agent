"""Itinerary draft and final-response agents."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..graph.state import TravelState
from ..services.llm import chat, truncate_for_prompt

APPROVAL_REQUEST = (
    "Please review the generated draft itinerary. Approve it to create the "
    "final polished plan, or provide feedback for revision."
)

REVISION_APPROVAL_REQUEST = (
    "Here is the revised plan. Approve it when you are satisfied, or describe "
    "further changes."
)


async def itinerary_agent(state: TravelState) -> dict:
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{truncate_for_prompt(state.get('flight_results', ''), 1500)}

Hotel Results:
{truncate_for_prompt(state.get('hotel_results', ''), 1500)}

Weather Results:
{truncate_for_prompt(state.get('weather_results', ''), 1500)}

Budget Results:
{truncate_for_prompt(state.get('budget_results', ''), 1500)}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    draft = await chat("You are an expert travel planner.", prompt)

    return {
        "itinerary": draft,
        "approval_request": APPROVAL_REQUEST,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def final_agent(state: TravelState) -> dict:
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        feedback = (
            state.get("human_feedback")
            or "Improve the draft before finalizing it."
        )
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{feedback}

Reply with the complete revised plan, not a diff. Preserve every section of the
draft that the feedback does not ask you to change, so the user never loses
content between rounds.
"""

    prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{truncate_for_prompt(state.get('flight_results', ''), 1200)}

Hotels:
{truncate_for_prompt(state.get('hotel_results', ''), 1200)}

Weather:
{truncate_for_prompt(state.get('weather_results', ''), 1200)}

Budget Analysis:
{truncate_for_prompt(state.get('budget_results', ''), 1200)}

Draft Itinerary:
{truncate_for_prompt(state.get('itinerary', ''), 6000)}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is
  unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = await chat(
        "You are a professional AI travel booking assistant.",
        prompt,
    )

    revising = not state.get("approved", False)
    revision_count = state.get("revision_count", 0) + (1 if revising else 0)

    # On a revision the reply becomes the next draft under review, so the
    # interrupt payload shows the reviewer exactly what they are judging. On the
    # first approval the polished reply is only the final answer.
    updates: dict = {
        "final_response": response,
        "messages": [AIMessage(content=response)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

    if revising:
        updates["itinerary"] = response
        updates["approval_request"] = REVISION_APPROVAL_REQUEST
        updates["revision_count"] = revision_count

    return updates
