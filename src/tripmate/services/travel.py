"""Travel planning service: owns the compiled graph and the HITL lifecycle."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..config import Settings, get_settings
from ..graph.state import initial_state
from ..graph.workflow import build_graph
from ..logging_config import get_logger
from ..observability import traced_span
from .checkpointer import CheckpointerHandle, create_checkpointer

logger = get_logger(__name__)

THREAD_ID_PREFIX = "user_"


class ThreadNotFoundError(RuntimeError):
    """Raised when a thread id has no checkpoint to resume."""


def new_thread_id() -> str:
    return f"{THREAD_ID_PREFIX}{uuid.uuid4().hex}"


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or ()
    if not interrupts:
        return None

    first = interrupts[0]
    payload = getattr(first, "value", first)
    return payload if isinstance(payload, dict) else {"value": payload}


def serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Flatten a graph result into the API response contract."""

    interrupt_payload = _interrupt_payload(result)
    messages = result.get("messages") or []
    last_message = messages[-1].content if messages else ""

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get("itinerary", "")
        itinerary = interrupt_payload.get("draft_itinerary", "")
        approval_request = interrupt_payload.get("approval_request", "")
    else:
        answer = result.get("final_response") or last_message
        itinerary = result.get("itinerary", "")
        approval_request = result.get("approval_request", "")

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": approval_request,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": itinerary,
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


class TravelPlanner:
    """Lifecycle wrapper around the compiled LangGraph workflow."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._lock = asyncio.Lock()
        self._handle: CheckpointerHandle | None = None
        self._graph: CompiledStateGraph | None = None

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            raise RuntimeError("TravelPlanner.start() has not completed.")
        return self._graph

    @property
    def checkpointer(self) -> CheckpointerHandle | None:
        return self._handle

    @property
    def checkpointer_info(self) -> dict[str, Any]:
        if self._handle is None:
            return {"backend": "uninitialized", "detail": "Planner has not started."}
        return self._handle.describe()

    async def start(self) -> None:
        """Idempotent startup.

        Safe to call again after a transient failure: a checkpointer that was
        unreachable during boot is retried on the next request.
        """

        async with self._lock:
            if self._graph is not None:
                return

            self._handle = await create_checkpointer(self._settings)
            self._graph = build_graph().compile(checkpointer=self._handle.saver)
            logger.info(
                "Travel graph compiled with %s checkpointer.",
                self._handle.backend,
            )

    async def aclose(self) -> None:
        async with self._lock:
            if self._handle is not None:
                await self._handle.aclose()
            self._handle = None
            self._graph = None

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, Any]]:
        return {"configurable": {"thread_id": thread_id}}

    async def start_run(
        self, user_input: str, thread_id: str | None = None
    ) -> dict[str, Any]:
        """Run the graph until it interrupts for human approval."""

        await self.start()

        thread_id = thread_id or new_thread_id()
        state = initial_state(user_input)
        state["messages"] = [HumanMessage(content=user_input)]

        with traced_span(
            "travel.start_run",
            enabled=self._settings.tracing_enabled,
            thread_id=thread_id,
            phase="start",
        ):
            result = await self.graph.ainvoke(state, config=self._config(thread_id))

        return serialize_result(result, thread_id)

    async def resume_run(
        self, thread_id: str, approved: bool, feedback: str = ""
    ) -> dict[str, Any]:
        """Resume a thread that is paused at the human approval node."""

        await self.start()

        if not thread_id or not thread_id.strip():
            raise ValueError("thread_id is required to resume a travel plan.")

        thread_id = thread_id.strip()

        if not await self._thread_exists(thread_id):
            raise ThreadNotFoundError(
                f"No paused draft found for thread '{thread_id}'. "
                "It may have been started before a server restart; "
                "submit a new request to create a fresh draft."
            )

        with traced_span(
            "travel.resume_run",
            enabled=self._settings.tracing_enabled,
            thread_id=thread_id,
            approved=approved,
            phase="resume",
        ):
            result = await self.graph.ainvoke(
                Command(resume={"approved": approved, "feedback": feedback.strip()}),
                config=self._config(thread_id),
            )

        return serialize_result(result, thread_id)

    async def _thread_exists(self, thread_id: str) -> bool:
        """True when the thread has a checkpoint saved on this backend."""

        if self._handle is None or self._handle.backend == "memory":
            # MemorySaver has no cheap index; rely on the invoke result to fail
            # loudly instead of a false negative.
            return True

        snapshot = await self.graph.aget_state(self._config(thread_id))
        return bool(snapshot and snapshot.values)


_planner: TravelPlanner | None = None


def get_planner() -> TravelPlanner:
    """Process-wide planner singleton."""

    global _planner

    if _planner is None:
        _planner = TravelPlanner()

    return _planner


def set_planner(planner: TravelPlanner | None) -> None:
    """Override the singleton. Used by tests."""

    global _planner
    _planner = planner
