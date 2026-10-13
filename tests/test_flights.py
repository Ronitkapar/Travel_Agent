"""Airport-code resolution and flight-provider failure reporting.

Both exist because of a specific way the flight section used to mislead: a
provider refusal was reported as "could not be matched to an airport code",
which blamed the user's request for a billing problem, and a rate limit was
reported as a plan gap, which sent the reader to the wrong remedy.
"""

from __future__ import annotations

import pytest

from tripmate.agents.specialists import _classify, _flight_routes, _readable_error
from tripmate.services.iata import code_from_text, from_table, looks_like_iata

# ─────────────────────────── table resolution ───────────────────────────


@pytest.mark.parametrize(
    ("place", "expected"),
    [
        ("Bengaluru", "BLR"),
        ("Bengaluru, India", "BLR"),
        ("bengaluru karnataka india", "BLR"),
        ("Bangalore", "BLR"),
        ("Jaipur", "JAI"),
        ("Kochi", "COK"),
        ("Cochin", "COK"),
        # A hill station has no airport of its own; the nearest gateway is the
        # only answer that produces a real route.
        ("Munnar", "COK"),
        ("Kochi and Munnar, Kerala, India", "COK"),
        ("New Delhi", "DEL"),
        ("Tokyo", "HND"),
        ("San Francisco", "SFO"),
    ],
)
def test_common_places_resolve_without_a_provider(place: str, expected: str) -> None:
    assert from_table(place) == expected


@pytest.mark.parametrize("place", ["", "   ", "Atlantis", "Zz-9", None])
def test_an_unmapped_place_returns_none(place: str | None) -> None:
    """None, not a guess: a wrong code yields a confidently wrong route."""

    assert from_table(place) is None


# ─────────────────────── model-reply extraction ───────────────────────


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("BLR", "BLR"),
        ("  JAI  ", "JAI"),
        ("BLR (Kempegowda International Airport)", "BLR"),
    ],
)
def test_a_bare_code_is_read_back(reply: str, expected: str) -> None:
    assert code_from_text(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "I am not sure",
        "NONE",
        "The code is BLR and the airline is IND",
        "",
    ],
)
def test_an_ambiguous_reply_yields_no_code(reply: str) -> None:
    assert code_from_text(reply) is None


def test_a_lowercase_word_is_not_mistaken_for_a_code() -> None:
    """'not' and 'the' are three letters; only case separates them from BLR."""

    assert code_from_text("I am not sure") is None
    assert looks_like_iata("BLR") is True
    assert looks_like_iata("BLRX") is False


# ───────────────────── provider failure classification ─────────────────────


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        ("api_error (function_access_restricted): not supported", "plan"),
        ("api_error (rate_limit_reached): too many requests", "rate"),
        # The rate-limit message also says "subscription plan"; order matters.
        (
            "api_error (rate_limit_reached): exceeded the limit allowed on your "
            "subscription plan",
            "rate",
        ),
        ("something else entirely", "other"),
    ],
)
def test_provider_errors_are_classified(error: str, kind: str) -> None:
    assert _classify(error) == kind


def test_a_rate_limit_is_not_reported_as_a_plan_gap() -> None:
    """They need opposite advice: wait, versus upgrade."""

    message = _readable_error(
        {
            "error": (
                "api_error (rate_limit_reached): You have exceeded the maximum "
                "rate limitation allowed on your subscription plan."
            )
        }
    )
    assert message is not None
    assert "rate-limiting" in message
    assert "plan does not include" not in message


def test_a_plan_restriction_names_the_remedy() -> None:
    message = _readable_error(
        {
            "error": (
                "api_error (function_access_restricted): Your current "
                "subscription plan does not support this API function."
            )
        }
    )
    assert message is not None
    assert "account upgrade" in message
    assert "function_access_restricted" in message


def test_an_empty_envelope_error_reads_as_nothing() -> None:
    assert _readable_error({}) is None


# ─────────────────────────── route lookup ───────────────────────────


async def test_an_unmapped_destination_is_named_specifically() -> None:
    """Origin resolved, destination did not: say which one, not 'the origin'."""

    routes, sources, note = await _flight_routes(
        {"origin": "Bengaluru", "destination": "Atlantis"}
    )

    assert routes == []
    # No provider call is made when resolution fails, so there is no source.
    assert sources == []
    assert note is not None
    assert "Atlantis" in note


async def test_a_missing_origin_and_destination_are_reported_together() -> None:
    routes, _, note = await _flight_routes({"origin": "", "destination": ""})

    assert routes == []
    assert note == "No origin city was given, so no routes were looked up."


async def test_an_identical_origin_and_destination_is_not_a_route() -> None:
    routes, _, note = await _flight_routes(
        {"origin": "Bengaluru", "destination": "Bangalore"}
    )

    assert routes == []
    assert note == "Origin and destination resolved to the same airport."


async def test_a_mapped_pair_issues_exactly_one_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two lookups would exhaust a free key's rate limit inside one run."""

    calls: list[tuple[str, int]] = []

    async def record(airport_iata: str, *, limit: int = 100):
        calls.append((airport_iata, limit))
        return [
            {
                "type": "text",
                "text": '{"ok": true, "data": [{"airline": "IndiGo", '
                '"flight_number": "6E469", "arrival_airport_code": "JAI"}]}',
            }
        ]

    monkeypatch.setattr("tripmate.mcp.client.flight_departures", record)

    routes, sources, note = await _flight_routes(
        {"origin": "Bengaluru", "destination": "Jaipur"}
    )

    assert note is None
    assert len(calls) == 1
    assert calls[0][0] == "BLR"
    assert sources == ["aviationstack:flight_arrival_departure_schedule"]

    assert len(routes) == 1
    assert routes[0]["dep_iata"] == "BLR"
    assert routes[0]["arr_iata"] == "JAI"
    assert routes[0]["airline_name"] == "IndiGo"
    assert routes[0]["flight_number"] == "6E469"
    # The board carries no IATA code; leaving it null keeps the contract honest.
    assert routes[0]["airline_iata"] is None