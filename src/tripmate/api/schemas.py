"""Pydantic request and response models.

The response models document the exact JSON contract the frontend depends on.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TravelRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


class TravelResponse(BaseModel):
    success: bool = True
    thread_id: str
    answer: Any = ""
    requires_approval: bool = False
    approval_request: str = ""
    flight_results: Any = ""
    hotel_results: Any = ""
    weather_results: Any = ""
    budget_results: Any = ""
    itinerary: Any = ""
    selected_agents: list[str] = Field(default_factory=list)
    trip_constraints: dict[str, Any] = Field(default_factory=dict)
    supervisor_reasoning: str = ""
    guardrail_allowed: bool = True
    guardrail_reason: str = ""
    approved: bool | None = None
    human_feedback: str = ""
    revision_count: int = 0
    max_revisions: int | None = None
    llm_calls: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    features: list[str]
    checkpointer: dict[str, Any]
    missing_credentials: list[str]
