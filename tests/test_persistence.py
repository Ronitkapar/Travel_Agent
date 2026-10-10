"""Thread persistence semantics.

These document what survives between the planning request and the approval
request, which is the main deployment risk on Render's free tier.
"""

from __future__ import annotations

import pytest

from tests.conftest import FakeChatModel
from tripmate.services import travel as travel_service
from tripmate.services.travel import (
    ThreadNotFoundError,
    TravelPlanner,
    new_thread_id,
    serialize_result,
)


def test_thread_ids_are_unique_and_prefixed() -> None:
    first, second = new_thread_id(), new_thread_id()
    assert first.startswith("user_")
    assert first != second


def test_interrupt_result_is_flagged_for_approval() -> None:
    payload = serialize_result(
        {
            "__interrupt__": (
                {"draft_itinerary": "Draft", "approval_request": "Review"},
            )
        },
        "user_1",
    )

    assert payload["requires_approval"] is True
    assert payload["answer"] == "Draft"
    assert payload["itinerary"] == "Draft"
    assert payload["approval_request"] == "Review"


def test_final_result_is_not_flagged_for_approval() -> None:
    payload = serialize_result(
        {"final_response": "Done", "messages": []}, "user_1"
    )

    assert payload["requires_approval"] is False
    assert payload["answer"] == "Done"


def test_serializer_falls_back_to_the_last_message() -> None:
    class Message:
        content = "last message"

    payload = serialize_result({"messages": [Message()]}, "user_1")
    assert payload["answer"] == "last message"


def test_checkpointer_defaults_to_memory_without_a_database_url(
    planner: TravelPlanner,
) -> None:
    assert planner.checkpointer_info["backend"] == "memory"


async def test_resuming_an_unstarted_thread_on_postgres_is_a_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a durable checkpointer an unknown thread is detected before invoke."""

    instance = TravelPlanner()
    await instance.start()

    # Simulate a durable backend whose lookup returns an empty snapshot.
    instance._handle.backend = "postgres"

    async def _no_state(config):  # noqa: ANN001
        class Snapshot:
            values = {}

        return Snapshot()

    monkeypatch.setattr(instance.graph, "aget_state", _no_state)

    with pytest.raises(ThreadNotFoundError):
        await instance.resume_run("user_unknown", approved=True)

    await instance.aclose()


async def test_blank_thread_id_is_rejected(planner: TravelPlanner) -> None:
    with pytest.raises(ValueError, match="thread_id"):
        await planner.resume_run("   ", approved=True)


async def test_state_survives_between_the_two_requests(
    planner: TravelPlanner, fake_llm: FakeChatModel
) -> None:
    """The approval request must see the draft produced by the planning request."""

    draft = await planner.start_run("Plan a 3 day trip to Prague")
    final = await planner.resume_run(draft["thread_id"], approved=True)

    # The final prompt embeds the draft produced during the first request.
    assert draft["itinerary"] in fake_llm.calls[-1][1]
    assert final["thread_id"] == draft["thread_id"]


async def test_a_failed_start_recovers_on_the_next_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpointer that was down during boot must be retried on demand."""

    instance = TravelPlanner()
    attempts: list[int] = []
    real_create = travel_service.create_checkpointer

    async def flaky_create(settings):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("database unreachable")
        return await real_create(settings)

    monkeypatch.setattr(travel_service, "create_checkpointer", flaky_create)

    with pytest.raises(RuntimeError):
        await instance.start()

    assert instance.checkpointer_info["backend"] == "uninitialized"

    # A subsequent request retries and succeeds.
    await instance.start()
    assert instance.checkpointer_info["backend"] == "memory"
    assert len(attempts) == 2

    await instance.aclose()
