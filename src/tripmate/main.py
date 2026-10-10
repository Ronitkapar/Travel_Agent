"""Application factory and ASGI entrypoint.

Run locally with::

    uvicorn tripmate.main:app --reload --host 0.0.0.0 --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import router
from .config import configure_tls, get_settings
from .logging_config import configure_logging, get_logger
from .observability import (
    configure_observability,
    traced_span,
)
from .observability import (
    shutdown as shutdown_observability,
)
from .services.travel import get_planner

logger = get_logger(__name__)

TITLE = "TripMate AI"
DESCRIPTION = (
    "LangGraph multi-agent travel planner with a supervisor, input guardrails, "
    "human-in-the-loop approval, MCP data sources, and a FastAPI frontend."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_observability(app, settings)

    planner = get_planner()
    try:
        with traced_span("planner.start", enabled=settings.tracing_enabled):
            await planner.start()
    except Exception:
        # Never block boot on persistence: /health must answer so Render can
        # report the instance, and planning routes will retry on demand.
        logger.exception("Planner failed to start; the app will run degraded.")

    try:
        yield
    finally:
        await planner.aclose()
        shutdown_observability()


def create_app() -> FastAPI:
    # Must run before any HTTP client is constructed so TLS trust is in place.
    configure_tls()

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
        logger.info("CORS origins: %s", ", ".join(settings.cors_origins))
    else:
        logger.warning("CORS_ORIGINS is empty; cross-origin frontend calls will fail.")

    app.include_router(router)

    index_file = settings.frontend_dir / "index.html"

    if settings.assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(settings.assets_dir)),
            name="assets",
        )

    if not index_file.is_file():
        logger.error(
            "Frontend not found at %s. Set FRONTEND_DIR to the directory "
            "containing index.html.",
            index_file,
        )

    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> Any:
        if not index_file.is_file():
            return JSONResponse(
                status_code=500,
                content={"detail": "frontend/index.html is missing from this build."},
            )
        return FileResponse(index_file)

    @app.get("/favicon.ico", include_in_schema=False, response_model=None)
    async def favicon() -> Any:
        return JSONResponse(content={}, status_code=204)

    return app


app = create_app()
