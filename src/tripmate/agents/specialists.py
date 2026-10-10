"""Specialist agents: flight, hotel, weather, and budget research."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from ..graph.state import TravelState
from ..logging_config import get_logger
from ..mcp import client as mcp
from ..services.llm import chat, extract_destination, truncate_for_prompt

logger = get_logger(__name__)

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

HOTEL_UNAVAILABLE = (
    "Live hotel search is temporarily unavailable. Provide general accommodation "
    "and neighborhood guidance based on the destination and clearly label it as "
    "non-live advice."
)


async def flight_agent(state: TravelState) -> dict:
    try:
        airports = await mcp.aviation_call("list_airports")
        airlines = await mcp.aviation_call("list_airlines")

        flight_data = await chat(
            "You are an expert travel flight planner.",
            FLIGHT_AGENT_PROMPT.format(
                query=state["user_query"],
                airport_data=truncate_for_prompt(airports, 3000),
                airline_data=truncate_for_prompt(airlines, 3000),
            ),
        )
    except Exception as exc:
        logger.warning("Flight agent data unavailable: %s", exc)
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def hotel_agent(state: TravelState) -> dict:
    try:
        query = f"Best hotels for {state['user_query']}"
        hotel_results = await mcp.tavily_search(query)
    except Exception as exc:
        logger.warning("Hotel agent MCP error: %s", exc)
        hotel_results = HOTEL_UNAVAILABLE

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def weather_agent(state: TravelState) -> dict:
    # Destination extraction is an LLM call that can fail on its own; keep it
    # inside the guard so a transient model error degrades instead of aborting
    # the whole run.
    try:
        city = await extract_destination(state["user_query"])
    except Exception as exc:
        logger.warning("Destination extraction failed: %s", exc)
        city = state["user_query"]

    try:
        weather_data = await mcp.current_weather(city)
        forecast_data = await mcp.weather_forecast(city)
        weather_results = (
            f"Current Weather:\n{weather_data}\n\nForecast:\n{forecast_data}"
        )
    except Exception as exc:
        logger.warning("Weather agent MCP error for %s: %s", city, exc)
        weather_results = (
            f"Live weather information for {city} is temporarily unavailable. "
            "Give general seasonal guidance and advise the traveler to verify "
            "the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def budget_agent(state: TravelState) -> dict:
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

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

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = await chat("You are a practical travel budget analyst.", prompt)

    return {
        "budget_results": response,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
