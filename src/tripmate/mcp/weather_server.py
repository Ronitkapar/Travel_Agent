"""Bundled weather MCP server.

Launched as a stdio subprocess by :mod:`tripmate.mcp.client`. Run it directly
only when debugging the server in isolation.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

REQUEST_TIMEOUT_SECONDS = 20
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

mcp = FastMCP("Weather MCP Server")


def _api_key() -> str:
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set in the server environment.")
    return key


def _require_city(city: str) -> str:
    cleaned = city.strip()
    if not cleaned:
        raise ValueError("city cannot be empty")
    return cleaned


def _request(path: str, city: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{WEATHER_BASE_URL}/{path}",
            params={"q": city, "appid": _api_key(), "units": "metric"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = getattr(exc, "response", None)
        suffix = f" Response: {detail.text[:500]}" if detail is not None else ""
        raise RuntimeError(f"OpenWeather request failed: {exc}.{suffix}") from exc


@mcp.tool()
def get_current_weather(city: str) -> dict[str, Any]:
    """Return the current weather for a city."""

    city = _require_city(city)
    data = _request("weather", city)

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@mcp.tool()
def get_forecast(city: str) -> dict[str, Any]:
    """Return the first five three-hour forecast entries for a city."""

    city = _require_city(city)
    data = _request("forecast", city)

    return {
        "city": data.get("city", {}).get("name", city),
        "forecast": [
            {
                "datetime": item["dt_txt"],
                "temperature_c": item["main"]["temp"],
                "condition": item["weather"][0]["description"],
            }
            for item in data.get("list", [])[:5]
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
