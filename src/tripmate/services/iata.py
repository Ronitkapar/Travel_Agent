"""City name to airport IATA code.

Resolution is local first. The configured aviation provider restricts its
airport-search endpoint to paid plans, so asking it to turn "Bengaluru" into
``BLR`` fails on a key that can still return flight data. Sending an unresolvable
lookup down that path produces an empty list, which used to be reported as
"could not be matched to an airport code" -- a claim about the user's request
when the truth was a provider refusal.

A bundled table answers the common cases without a network call, and the model
handles the long tail. Neither source can fail a run: an unknown city returns
``None`` and the caller says so, rather than guessing a code that would yield a
confidently wrong route list.
"""

from __future__ import annotations

import re

from ..logging_config import get_logger

logger = get_logger(__name__)

IATA_PATTERN = re.compile(r"^[A-Z]{3}$")

# One entry per airport a traveller actually departs from or arrives at. The
# value is the airport's own IATA code, not the metro code, because the timetable
# endpoint filters on airports.
AIRPORT_CODES: dict[str, str] = {
    # India
    "bengaluru": "BLR",
    "bangalore": "BLR",
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "new delhi del": "DEL",
    "chennai": "MAA",
    "madras": "MAA",
    "hyderabad": "HYD",
    "kolkata": "CCU",
    "calcutta": "CCU",
    "pune": "PNQ",
    "ahmedabad": "AMD",
    "jaipur": "JAI",
    "goa": "GOI",
    "goi": "GOI",
    "kochi": "COK",
    "cochin": "COK",
    "kochi cochin": "COK",
    "thiruvananthapuram": "TRV",
    "trivandrum": "TRV",
    "thiruvananthapuram trivandrum": "TRV",
    "munnar": "COK",
    "kumarakom": "COK",
    "alappuzha": "COK",
    "thekkady": "COK",
    "kanyakumari": "IXC",
    "lucknow": "LKO",
    "kanpur": "KNU",
    "nagpur": "NAG",
    "indore": "IDR",
    "bhopal": "BHO",
    "patna": "PAT",
    "guwahati": "GAU",
    "chandigarh": "IXC",
    "dehradun": "DED",
    "ranchi": "IXR",
    "bhubaneswar": "BBI",
    "coimbatore": "CJB",
    "mysore": "BLR",
    "mysuru": "BLR",
    "madurai": "IXM",
    "kozhikode": "CCJ",
    "calicut": "CCJ",
    "visakhapatnam": "VTZ",
    "port blair": "IXZ",
    "agra": "AGR",
    "ludhiana": "LUH",
    "varanasi": "VAR",
    "jammu": "IXJ",
    "amritsar": "AIP",
    "dehradun ddn": "DED",
    # Asia-Pacific
    "tokyo": "HND",
    "tokyo haneda": "HND",
    "tokyo narita": "NRT",
    "osaka": "KIX",
    "kyoto": "KIX",
    "seoul": "ICN",
    "bangkok": "BKK",
    "singapore": "SIN",
    "kuala lumpur": "KUL",
    "jakarta": "CGK",
    "bali": "DPS",
    "denpasar": "DPS",
    "manila": "MNL",
    "hong kong": "HKG",
    "shanghai": "PVG",
    "beijing": "PEK",
    "shenzhen": "SZX",
    "guangzhou": "CAN",
    "chengdu": "TFU",
    "taipei": "TPE",
    "hanoi": "HAN",
    "ho chi minh city": "SGN",
    "ho chi minh": "SGN",
    "colombo": "CMB",
    "kathmandu": "KTM",
    "dhaka": "DAC",
    "karachi": "KHI",
    "lahore": "LHE",
    "islamabad": "ISB",
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "doha": "DOH",
    "riyadh": "RUH",
    "jeddah": "JED",
    "muscat": "MCT",
    "baku": "GYD",
    "tbilisi": "TBS",
    # Europe
    "london": "LHR",
    "paris": "CDG",
    "amsterdam": "AMS",
    "frankfurt": "FRA",
    "munich": "MUC",
    "berlin": "BER",
    "zurich": "ZRH",
    "geneva": "GVA",
    "milan": "MXP",
    "rome": "FCO",
    "madrid": "MAD",
    "barcelona": "BCN",
    "lisbon": "LIS",
    "dublin": "DUB",
    "vienna": "VIE",
    "prague": "PRG",
    "warsaw": "WAW",
    "stockholm": "ARN",
    "oslo": "OSL",
    "copenhagen": "CPH",
    "helsinki": "HEL",
    "athens": "ATH",
    "istanbul": "IST",
    "moscow": "SVO",
    "athens ath": "ATH",
    # Americas
    "new york": "JFK",
    "new york city": "JFK",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "chicago": "ORD",
    "seattle": "SEA",
    "boston": "BOS",
    "miami": "MIA",
    "dallas": "DFW",
    "houston": "IAH",
    "denver": "DEN",
    "atlanta": "ATL",
    "washington": "IAD",
    "washington dc": "IAD",
    "toronto": "YYZ",
    "vancouver": "YVR",
    "montreal": "YUL",
    "mexico city": "MEX",
    "sao paulo": "GRU",
    "rio de janeiro": "GIG",
    "buenos aires": "EZE",
    "bogota": "BOG",
    "lima": "LIM",
    # Oceania / Africa
    "sydney": "SYD",
    "melbourne": "MEL",
    "brisbane": "BNE",
    "perth": "PER",
    "auckland": "AKL",
    "fiji": "NAN",
    "suva": "SUV",
    "johannesburg": "JNB",
    "cape town": "CPT",
    "nairobi": "NBO",
    "cairo": "CAI",
    "lagos": "LOS",
    "casablanca": "CMN",
    "marrakesh": "RAK",
    "tunis": "TUN",
    "dakar": "DSS",
}

# Words that identify a place without distinguishing an airport inside it.
_ADMINISTRATIVE = frozenset(
    {
        "india",
        "kerala",
        "karnataka",
        "tamil nadu",
        "rajasthan",
        "maharashtra",
        "gujarat",
        "telangana",
        "andhra pradesh",
        "west bengal",
        "punjab",
        "haryana",
        "himachal pradesh",
        "uttarakhand",
        "goa",
        "city",
        "state",
        "region",
        "province",
        "the",
    }
)


def _normalize(value: str) -> str:
    """Lowercase, de-punctuate, and collapse whitespace."""

    text = value.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _candidates(city: str) -> list[str]:
    """Lookup keys for one place name, most specific first.

    A supervisor-written constraint looks like "Kochi and Munnar, Kerala, India"
    or "Bengaluru, Karnataka, India". The table has neither of those as written,
    so the segments are tried individually after the whole string: the
    administrative words are dropped first, because "kerala kochi" is noise
    rather than a place anyone calls it.
    """

    normalized = _normalize(city)
    if not normalized:
        return []

    ordered: list[str] = []

    def add(key: str) -> None:
        if key and key not in ordered:
            ordered.append(key)

    add(normalized)

    parts = [part for part in normalized.split(" ") if part]
    stripped = [part for part in parts if part not in _ADMINISTRATIVE]
    add(" ".join(stripped))

    # "Kochi and Munnar" -> try each side on its own, in order.
    joined = " ".join(stripped)
    for split in (" and ",):
        if split in joined:
            left, _, right = joined.partition(split)
            add(left.strip())
            add(right.strip())

    for part in stripped:
        add(part)

    return ordered


def from_table(city: str | None) -> str | None:
    """Resolve from the bundled table alone. No network, no model."""

    for key in _candidates(city or ""):
        code = AIRPORT_CODES.get(key)
        if code:
            return code

    return None


def looks_like_iata(value: str) -> bool:
    """True for a bare three-letter airport code."""

    candidate = (value or "").strip().upper()
    return bool(IATA_PATTERN.fullmatch(candidate))


def code_from_text(value: str) -> str | None:
    """Extract an IATA code from a model reply, if there is exactly one.

    Two guards keep a chatty or unsure reply from becoming an airport code. The
    token must already be uppercase, which is the code's own convention and also
    the thing that separates ``BLR`` from the English word "not"; and it must be
    the only candidate, so a reply naming both a city code and an airline code
    cannot be read as an answer.
    """

    found = set(re.findall(r"\b[A-Z]{3}\b", value or ""))
    if len(found) != 1:
        return None
    return found.pop()


__all__ = [
    "AIRPORT_CODES",
    "code_from_text",
    "from_table",
    "looks_like_iata",
]