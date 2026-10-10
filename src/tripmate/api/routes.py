"""HTTP routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from .. import __version__
from ..logging_config import get_logger
from ..services.travel import ThreadNotFoundError
from .dependencies import Planner, SettingsDep
from .schemas import ApprovalRequest, TravelRequest

logger = get_logger(__name__)

router = APIRouter()

FEATURES = ("supervisor_agent", "input_guardrail", "human_in_the_loop")


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": message},
    )


def _ok(payload: dict[str, Any]) -> JSONResponse:
    return JSONResponse(content={"success": True, **payload})


@router.post("/api/travel")
async def create_travel_plan(request: TravelRequest, planner: Planner):
    user_message = request.message.strip()

    if not user_message:
        return _error("Message cannot be empty.", status.HTTP_400_BAD_REQUEST)

    try:
        result = await planner.start_run(
            user_input=user_message, thread_id=request.thread_id
        )
    except Exception as exc:
        logger.exception("Travel planning failed")
        return _error(str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    return _ok(result)


@router.post("/api/travel/approve")
async def approve_travel_plan(request: ApprovalRequest, planner: Planner):
    feedback = request.feedback.strip()

    if not request.approved and not feedback:
        return _error(
            "Please provide revision feedback when rejecting the draft.",
            status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = await planner.resume_run(
            thread_id=request.thread_id,
            approved=request.approved,
            feedback=feedback,
        )
    except ThreadNotFoundError as exc:
        return _error(str(exc), status.HTTP_404_NOT_FOUND)
    except ValueError as exc:
        return _error(str(exc), status.HTTP_400_BAD_REQUEST)
    except Exception as exc:
        logger.exception("Travel approval failed")
        return _error(str(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    return _ok(result)


@router.get("/health")
async def health(planner: Planner, settings: SettingsDep) -> dict[str, Any]:
    """Liveness plus a config report.

    Deliberately performs no network I/O so Render's health check stays cheap
    and the endpoint still answers when a third-party service is down.
    """

    missing = list(settings.missing_credentials())
    status_value = "ok" if not missing else "degraded"

    checkpointer = planner.checkpointer_info
    backend = checkpointer.get("backend", "uninitialized")
    durable = backend == "postgres"

    return {
        "status": status_value,
        "version": __version__,
        # Reports the backend actually in use. A configured DATABASE_URL that
        # failed to connect falls back to memory, and this must not imply the
        # thread is durable when it is not.
        "environment": backend,
        "durable": durable,
        "features": list(FEATURES),
        "checkpointer": checkpointer,
        "observability": {
            "tracing": settings.tracing_enabled,
            "backend": "logfire" if settings.tracing_enabled else "disabled",
        },
        "missing_credentials": missing,
    }
