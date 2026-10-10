"""HTTP contract tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tripmate.main import create_app
from tripmate.services.travel import ThreadNotFoundError, set_planner

RESPONSE_KEYS = {
    "thread_id",
    "answer",
    "requires_approval",
    "approval_request",
    "flight_results",
    "hotel_results",
    "weather_results",
    "budget_results",
    "itinerary",
    "selected_agents",
    "trip_constraints",
    "supervisor_reasoning",
    "guardrail_allowed",
    "guardrail_reason",
    "approved",
    "human_feedback",
    "llm_calls",
}


class StubPlanner:
    """Stands in for TravelPlanner so the API contract can be asserted alone."""

    checkpointer_info = {"backend": "memory", "detail": "test"}

    def __init__(self) -> None:
        self.resume_calls: list[dict] = []
        self.start_calls: list[dict] = []
        self.raise_on_start: Exception | None = None
        self.raise_on_resume: Exception | None = None

    @staticmethod
    def _payload(**overrides):
        base = {key: "" for key in RESPONSE_KEYS}
        base.update(
            {
                "selected_agents": [],
                "trip_constraints": {},
                "guardrail_allowed": True,
                "requires_approval": False,
                "approved": None,
                "llm_calls": 0,
            }
        )
        base.update(overrides)
        return base

    async def start_run(self, user_input: str, thread_id: str | None = None):
        if self.raise_on_start:
            raise self.raise_on_start
        self.start_calls.append({"user_input": user_input, "thread_id": thread_id})
        return self._payload(
            thread_id=thread_id or "user_abc123",
            answer="Draft itinerary",
            itinerary="Draft itinerary",
            requires_approval=True,
            approval_request="Please review",
            selected_agents=["itinerary_agent"],
            trip_constraints={"destination": "Tokyo"},
            supervisor_reasoning="r",
            llm_calls=3,
        )

    async def resume_run(self, thread_id: str, approved: bool, feedback: str = ""):
        if self.raise_on_resume:
            raise self.raise_on_resume
        self.resume_calls.append(
            {"thread_id": thread_id, "approved": approved, "feedback": feedback}
        )
        return self._payload(
            thread_id=thread_id,
            answer="Final plan",
            approved=approved,
            human_feedback=feedback,
            llm_calls=4,
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from tripmate.api import dependencies, routes

    stub = StubPlanner()
    monkeypatch.setattr(dependencies, "get_planner", lambda: stub)
    monkeypatch.setattr(routes, "get_planner", lambda: stub, raising=False)

    app = create_app()
    with TestClient(app) as test_client:
        test_client.stub = stub  # type: ignore[attr-defined]
        yield test_client
    set_planner(None)


def test_health_reports_status_and_features(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["missing_credentials"] == []
    assert body["checkpointer"]["backend"] == "memory"
    assert "human_in_the_loop" in body["features"]


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "TripMate AI" in response.text


def test_static_assets_are_served(client: TestClient) -> None:
    assert client.get("/assets/script.js").status_code == 200
    assert client.get("/assets/config.js").status_code == 200


def test_travel_endpoint_returns_the_full_contract(client: TestClient) -> None:
    response = client.post("/api/travel", json={"message": "  Trip to Tokyo  "})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert set(body) >= RESPONSE_KEYS
    assert body["requires_approval"] is True
    # The message is trimmed before it reaches the graph.
    assert client.stub.start_calls[0]["user_input"] == "Trip to Tokyo"


def test_blank_message_is_rejected(client: TestClient) -> None:
    # Whitespace passes schema validation but is empty after trimming.
    blank = client.post("/api/travel", json={"message": "   "})
    assert blank.status_code == 400
    assert blank.json() == {"success": False, "error": "Message cannot be empty."}

    # A zero-length message fails schema validation outright.
    empty = client.post("/api/travel", json={"message": ""})
    assert empty.status_code == 422

    assert client.stub.start_calls == []


def test_rejecting_without_feedback_is_a_400(client: TestClient) -> None:
    response = client.post(
        "/api/travel/approve",
        json={"thread_id": "user_abc123", "approved": False, "feedback": "  "},
    )

    assert response.status_code == 400
    assert "feedback" in response.json()["error"].lower()


def test_approval_resumes_the_thread(client: TestClient) -> None:
    response = client.post(
        "/api/travel/approve",
        json={"thread_id": "user_abc123", "approved": True, "feedback": "  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["answer"] == "Final plan"
    assert body["approved"] is True
    assert client.stub.resume_calls[0]["thread_id"] == "user_abc123"


def test_unknown_thread_returns_404(client: TestClient) -> None:
    client.stub.raise_on_resume = ThreadNotFoundError("thread is gone")

    response = client.post(
        "/api/travel/approve",
        json={"thread_id": "user_missing", "approved": True},
    )

    assert response.status_code == 404
    assert "thread" in response.json()["error"].lower()


def test_agent_failures_return_500_with_a_message(client: TestClient) -> None:
    client.stub.raise_on_start = RuntimeError("model exploded")

    response = client.post("/api/travel", json={"message": "Trip to Oslo"})

    assert response.status_code == 500
    assert response.json() == {"success": False, "error": "model exploded"}


def test_cors_preflight_allows_the_configured_origin(client: TestClient) -> None:
    response = client.options(
        "/api/travel",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
