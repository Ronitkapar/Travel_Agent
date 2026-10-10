FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# uv backs the AviationStack MCP server, which the flight agent launches via
# `uvx`. Without it the flight agent degrades to an "unavailable" message.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the packaging metadata first so dependency installation stays
# cached across source-only changes.
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY frontend ./frontend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn tripmate.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
