"""Logging setup shared by the API and agent layers."""

from __future__ import annotations

import logging
import re
import sys
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

_CONFIGURED = False
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

# Secrets travel in query strings for some MCP servers (the Tavily server
# authenticates with ?tavilyApiKey=...), so any URL that reaches a log handler
# or a trace has to be scrubbed first. Vendor-prefixed names such as
# ``tavilyApiKey`` must be caught too, hence substring matching rather than
# exact comparison.
_SENSITIVE_QUERY_SUBSTRINGS = (
    "access_token",
    "api_key",
    "apikey",
    "appid",
    "auth",
    "credential",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "subscription",
    "token",
)


def _is_sensitive(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SENSITIVE_QUERY_SUBSTRINGS)


def redact_url(url: str) -> str:
    """Return ``url`` with sensitive query values and userinfo masked."""

    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable-url>"

    netloc = parts.netloc
    if "@" in netloc:  # scheme://user:password@host
        netloc = "***@" + netloc.rsplit("@", 1)[1]

    if not parts.query:
        return urlunsplit((parts.scheme, netloc, parts.path, "", parts.fragment))

    # Built by hand rather than urlencode() so the mask stays readable
    # instead of being percent-escaped to %2A%2A%2A.
    masked = [
        f"{k}=***" if _is_sensitive(k) else f"{k}={quote(v, safe='')}"
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(
        (parts.scheme, netloc, parts.path, "&".join(masked), parts.fragment)
    )


_URL_IN_TEXT = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://\S+")


def redact_text(text: str) -> str:
    """Mask sensitive query parameters inside every URL found in ``text``."""

    return _URL_IN_TEXT.sub(lambda m: redact_url(m.group(0)), text)


class RedactUrlsFilter(logging.Filter):
    """Scrub URLs from log records before handlers format them.

    ``httpx`` logs the full request URL at INFO, which would otherwise write an
    API key to stdout and to any shipped log drain.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True

        cleaned = redact_text(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def configure_logging(level: str = "info") -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    install_url_redaction()
    _CONFIGURED = True


def install_url_redaction() -> None:
    """Make sure no handler can render a URL containing a secret.

    ``Logger.filter`` only sees records logged through that exact logger, and
    records from ``httpx`` merely *propagate* to the root logger's handlers.
    So the filter has to sit on the emitting loggers and on every handler --
    attaching it to the root logger alone silently does nothing.
    """

    redact_filter = RedactUrlsFilter()

    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).addFilter(redact_filter)

    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RedactUrlsFilter) for f in handler.filters):
            handler.addFilter(redact_filter)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# Installed on import rather than inside configure_logging() so that redaction
# never depends on startup order: any entry point that imports this module --
# including a bare `python -c` script or a test that never builds the app --
# still cannot print a credential through a third-party logger.
install_url_redaction()
