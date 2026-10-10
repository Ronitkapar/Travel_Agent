"""LLM access.

Every agent talks to the model through this module so there is exactly one
place to swap in a stub during tests.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ..config import get_settings
from ..logging_config import get_logger
from ..observability import traced_span

logger = get_logger(__name__)

# Free-tier Groq keys are commonly capped at 8k tokens per minute, and every
# agent prompt carries the previous agents' output. Without a ceiling, one
# verbose MCP response can push a downstream prompt over the limit and fail the
# whole run with a 413.
MAX_PROMPT_CHARS = 12_000


def truncate_for_prompt(value: object, limit: int = MAX_PROMPT_CHARS) -> str:
    """Clamp interpolated data to a bounded, prompt-safe length."""

    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated]"


def _build_chat_model() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(model=settings.model, api_key=settings.require("GROQ_API_KEY"))


# Held as a module-level name rather than looked up through ``get_chat_model``
# so resetting the cache still works when the accessor is patched out.
_cached_chat_model = lru_cache(maxsize=1)(_build_chat_model)


def get_chat_model() -> ChatGroq:
    return _cached_chat_model()


def reset_chat_model() -> None:
    """Drop the cached model. Used by tests and after a credential change."""

    _cached_chat_model.cache_clear()


async def chat(system_prompt: str, user_prompt: str) -> str:
    """Return the model's raw text reply."""

    settings = get_settings()

    # Only the model id and outcome are recorded; prompts and replies stay out
    # of the trace.
    with traced_span(
        "llm.chat",
        enabled=settings.tracing_enabled,
        model=settings.model,
    ):
        response = await get_chat_model().ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

    return str(response.content)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the outermost JSON object from a model reply."""

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model returned JSON that was not an object.")

    return parsed


async def extract_destination(query: str) -> str:
    """Pull the destination city or country out of a free-form trip request."""

    destination = (
        await chat(
            "You extract travel destinations. Reply with the destination name only.",
            f"Extract only the destination city or country.\n\n"
            f"Travel request:\n{query}\n\n"
            f"Return only the destination name with no explanation.",
        )
    ).strip()

    if not destination:
        raise ValueError("The destination could not be extracted.")

    return destination
