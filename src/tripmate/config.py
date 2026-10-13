"""Environment-driven settings for TripMate AI.

Nothing in this package reads ``os.environ`` directly, so a test can build a
``Settings`` instance without mutating the process environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import certifi
from dotenv import load_dotenv

from .logging_config import get_logger

logger = get_logger(__name__)

# Verified available against the Groq API for the configured key. Override with
# GROQ_MODEL if your key has access to a different model.
DEFAULT_MODEL = "openai/gpt-oss-120b"
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
AVATIONSTACK_MCP_PACKAGE = "aviationstack-mcp"

# Groq leaves max_tokens unset by default, which caps replies at 3072 completion
# tokens. The itinerary and final-response agents exceed that on a long trip, so
# the cap is set explicitly. Override with LLM_MAX_TOKENS.
DEFAULT_MAX_COMPLETION_TOKENS = 8_192

# Upper bound on human feedback rounds before a plan is finalized anyway. The
# review loop is otherwise unbounded and would exhaust the graph recursion
# limit. Override with MAX_REVISIONS.
DEFAULT_MAX_REVISIONS = 5

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def frontend_dir(environ: dict[str, str] | None = None) -> Path:
    """Locate the static frontend across install layouts.

    The project uses a ``src/`` layout, so ``PROJECT_ROOT`` is the repository
    root for an editable or in-place run. A non-editable install (a container,
    for example) resolves inside ``site-packages`` instead, so the working
    directory is checked too. ``FRONTEND_DIR`` overrides everything.
    """

    env = os.environ if environ is None else environ
    override = env.get("FRONTEND_DIR")

    candidates = [
        Path(override) if override else None,
        PROJECT_ROOT / "frontend",
        PACKAGE_ROOT / "frontend",
        Path.cwd() / "frontend",
    ]

    for candidate in candidates:
        if candidate is not None and (candidate / "index.html").is_file():
            return candidate

    # Nothing matched; return the repository-relative default so the error
    # message names a sensible path.
    return PROJECT_ROOT / "frontend"


# Host, or host:port. Credentials, paths, and stray punctuation are rejected.
_NETLOC_PATTERN = re.compile(
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*"
    r"(?::[0-9]{1,5})?|\[[0-9A-Fa-f:.]+\](?::[0-9]{1,5})?"
)


def _is_valid_origin(origin: str) -> bool:
    """True for a bare ``scheme://host[:port]`` origin.

    CORS compares the serialized origin against this value, so anything with a
    path, credentials, or surrounding markup can never match a real browser.
    """

    try:
        parsed = urlsplit(origin)
    except ValueError:
        # Malformed IPv6 literals, stray brackets, and similar raise here.
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    # urlsplit is permissive inside netloc, so a stray ')' or other stray
    # character can survive parsing. Re-check against a strict host[:port]
    # shape; the port must also parse as an integer.
    try:
        if parsed.port is None and ":" in parsed.netloc.rsplit("]", 1)[-1]:
            return False
    except ValueError:
        return False

    if parsed.path or parsed.query or parsed.fragment or parsed.username:
        return False

    return _NETLOC_PATTERN.fullmatch(parsed.netloc) is not None


def _split_origins(raw: str) -> tuple[str, ...]:
    """Parse ``CORS_ORIGINS``, discarding anything that is not a usable origin.

    A malformed value silently disables CORS in a way that is hard to diagnose,
    so each rejected entry is logged with the reason.
    """

    origins: list[str] = []

    for candidate in raw.split(","):
        origin = candidate.strip().rstrip("/")
        if not origin:
            continue

        if not _is_valid_origin(origin):
            logger.warning(
                "Ignoring invalid CORS_ORIGINS entry %r. Expected a bare origin "
                "such as https://your-app.vercel.app.",
                origin[:120],
            )
            continue

        if origin not in origins:
            origins.append(origin)

    return tuple(origins)


def _normalize_database_url(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None

    url = raw.strip()
    # Managed PostgreSQL providers require TLS; the driver defaults to
    # preferring SSL but some hosted proxies reject the implicit handshake.
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"

    return url


def _positive_int(raw: str | None, default: int) -> int:
    """Parse a positive integer, falling back to ``default`` on bad input.

    A malformed value must not stop the app from booting, so the default is
    used instead of letting the conversion error escape.
    """

    try:
        value = int(raw) if raw and raw.strip() else default
    except ValueError:
        logger.warning("Ignoring malformed integer value %r; using %s.", raw, default)
        return default

    return value if value > 0 else default


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str | None = None
    tavily_api_key: str | None = None
    aviationstack_api_key: str | None = None
    openweather_api_key: str | None = None
    database_url: str | None = None
    logfire_token: str | None = None
    logfire_environment: str = "local"

    model: str = DEFAULT_MODEL
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    max_revisions: int = DEFAULT_MAX_REVISIONS

    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    weather_server_path: Path = PACKAGE_ROOT / "mcp" / "weather_server.py"
    frontend_dir: Path = field(default_factory=frontend_dir)

    @property
    def assets_dir(self) -> Path:
        return self.frontend_dir / "assets"

    @property
    def configured_checkpointer_backend(self) -> str:
        """Backend implied by configuration alone, before any connection attempt."""

        return "postgres" if self.database_url else "memory"

    @property
    def tracing_enabled(self) -> bool:
        """Tracing stays off unless a token is present.

        Defaulting to off keeps the app runnable with no Logfire project and
        avoids emitting spans that have nowhere to go.
        """

        return bool(self.logfire_token)

    def missing_credentials(self) -> tuple[str, ...]:
        """Credential names required for a full end-to-end planning run."""

        required = {
            "GROQ_API_KEY": self.groq_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "AVIATION_STACK_API_KEY": self.aviationstack_api_key,
            "OPENWEATHER_API_KEY": self.openweather_api_key,
        }
        return tuple(sorted(name for name, value in required.items() if not value))

    def require(self, name: str) -> str:
        value = getattr(self, name.lower())
        if not value:
            raise RuntimeError(
                f"{name} is not configured. Set it in the environment "
                f"or in a local .env file before starting TripMate."
            )
        return value


def settings_from_env(environ: dict[str, str] | None = None) -> Settings:
    env = os.environ if environ is None else environ

    # AVIATIONSTACK_API_KEY is the historical spelling; keep it working.
    aviation_key = env.get("AVIATION_STACK_API_KEY") or env.get(
        "AVIATIONSTACK_API_KEY"
    )

    return Settings(
        groq_api_key=env.get("GROQ_API_KEY"),
        tavily_api_key=env.get("TAVILY_API_KEY"),
        aviationstack_api_key=aviation_key,
        openweather_api_key=env.get("OPENWEATHER_API_KEY"),
        database_url=_normalize_database_url(env.get("DATABASE_URL")),
        logfire_token=env.get("LOGFIRE_TOKEN"),
        logfire_environment=env.get("LOGFIRE_ENVIRONMENT") or "local",
        model=env.get("GROQ_MODEL") or DEFAULT_MODEL,
        host=env.get("HOST") or "0.0.0.0",
        port=_positive_int(env.get("PORT"), 8000),
        log_level=(env.get("LOG_LEVEL") or "info").lower(),
        max_completion_tokens=_positive_int(
            env.get("LLM_MAX_TOKENS"), DEFAULT_MAX_COMPLETION_TOKENS
        ),
        max_revisions=_positive_int(env.get("MAX_REVISIONS"), DEFAULT_MAX_REVISIONS),
        cors_origins=_split_origins(
            env.get("CORS_ORIGINS")
            or "http://localhost:5173,http://localhost:3000"
        ),
        weather_server_path=Path(
            env.get("WEATHER_SERVER_PATH")
            or PACKAGE_ROOT / "mcp" / "weather_server.py"
        ),
        frontend_dir=frontend_dir(env),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv_file()
    return settings_from_env()


def load_dotenv_file() -> None:
    """Load a local ``.env`` if present.

    Platform environments provide real variables, so this is a no-op there.
    ``override=False`` keeps a real environment variable authoritative over a
    stale local file.
    """

    load_dotenv(PROJECT_ROOT / ".env", override=False)


def configure_tls() -> None:
    """Point Python and requests at the bundled CA store.

    Hosted TLS interception by some platforms breaks the default certifi-less
    lookup, so both variables are set before any HTTP client is constructed.
    """

    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
