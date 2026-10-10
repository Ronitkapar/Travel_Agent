# TripMate AI

A multi-agent travel planner built on **LangGraph** and the **Model Context Protocol (MCP)**.

A single request is routed by a supervisor agent to only the specialists it actually needs, grounded in live MCP data sources. The workflow produces a draft itinerary and then **pauses for human approval** before returning the final plan.

---

## Why this project exists

Most agent demos run every agent on every request. This one does not.

- **The supervisor decides who runs.** Ask about flights and only the flight agent executes.
- **Off-topic requests are rejected before any work happens**, by a guardrail rather than by prompt convention.
- **A human signs off before the final answer.** The graph suspends mid-run and resumes on demand.
- **Third-party failures are contained.** A dead weather provider degrades one agent, not the request.

---

## How it works

### Request flow

```
POST /api/travel
      │
      ▼
  supervisor ──── not travel-related ──► guardrail_blocked ──► END
      │
      │ travel-related
      ▼
  selects only the needed specialists
      │
      ▼
  flight ─► hotel ─► weather ─► budget        (each conditional on selection)
      │
      ▼
  itinerary_agent  ──► draft
      │
      ▼
  human_approval   ──► interrupt() ──► SUSPENDED
      │
      │  POST /api/travel/approve
      ▼
  final_agent      ──► approved plan
```

### Graph topology

Nodes are routed dynamically, so unselected specialists are skipped entirely.

```
START → supervisor ─┬─(blocked)──► guardrail_blocked → END
                    │
                    └─(allowed)─► flight ⇄ hotel ⇄ weather ⇄ budget
                                                   │
                                                   ▼
                                        itinerary_agent → human_approval
                                                                      │
                                                                      ▼
                                                                final_agent → END
```

`human_approval` calls LangGraph's `interrupt()`. The graph suspends and checkpoints its state under the request's `thread_id`. An approval request resumes that same thread with `Command(resume={...})`.

### The five agents

| Agent | Responsibility | Data source |
|---|---|---|
| **supervisor** | Guardrail gate and specialist selection | — |
| **flight_agent** | Airports, airlines, routes, airfare | MCP · AviationStack |
| **hotel_agent** | Accommodation suggestions | MCP · Tavily |
| **weather_agent** | Current conditions and forecast | MCP · weather server |
| **budget_agent** | Cost breakdown and feasibility | — |

`itinerary_agent` and `final_agent` synthesise the draft and the approved plan from whatever the specialists returned.

### Guardrails

Input is classified as travel-related or not before any specialist runs. The check **fails open**: a model or parser error must never take down travel planning, so a failed check allows the request through rather than blocking it.

### Human-in-the-loop

Approval is a real suspension, not a UI affordance. The graph stops mid-execution and its state is checkpointed:

- **With `DATABASE_URL`** — Postgres via `AsyncPostgresSaver`. A draft waiting for approval survives a cold start, restart, or redeploy.
- **Without it** — in-memory state. A pending approval is lost whenever the process restarts.

---

## Architecture

| Decision | Rationale |
|---|---|
| **Fully async** (async nodes, `ainvoke`, `AsyncPostgresSaver`) | MCP tools are already async; one event loop avoids `nest_asyncio` and per-call `asyncio.run()`. |
| **Postgres is optional** | The app must start without a database. Persistence is opt-in via `DATABASE_URL`. |
| **Startup never blocks on persistence** | `/health` must answer immediately; an unreachable database is logged, not fatal. |
| **Per-server MCP loading** | Servers load one at a time, so a broken provider degrades only its own agent. |
| **Failures contained per agent** | Every specialist catches its own errors and returns a degraded result instead of aborting the run. |
| **One LLM seam** | Every agent calls `services/llm.py`, so the model is swappable in one place. |
| **Prompts and replies stay out of traces** | Only model id and outcome are recorded. |
| **`src/` layout** | One dependency list, standard tooling, importable after a plain install. |
| **Dependency-free frontend** | Plain HTML, CSS and JavaScript. No build step. |

---

## API

### `POST /api/travel`

```json
{ "message": "Plan a 7 day Japan trip under 2 lakhs", "thread_id": null }
```

Runs the graph and returns the draft with `requires_approval: true`. Omit `thread_id` to start a new conversation; the generated id comes back in the response.

### `POST /api/travel/approve`

```json
{ "thread_id": "user_…", "approved": true, "feedback": "" }
```

Resumes the suspended thread. When `approved` is `false`, `feedback` is required (`400` otherwise). Returns the final plan with `requires_approval: false`.

### `GET /health`

Liveness probe. Makes no external calls, so it stays fast and keeps answering even when a provider is down. Reports the checkpointer backend, so a degraded (in-memory) instance is visible rather than silent.

---

## Project structure

```
src/tripmate/
├── main.py               # create_app(), lifespan, CORS, static serving
├── config.py             # Settings, env parsing, path resolution
├── logging_config.py     # logging setup + URL secret redaction
├── observability.py      # Logfire instrumentation, credential scrubbing
│
├── api/
│   ├── routes.py         # /api/travel, /api/travel/approve, /health
│   ├── schemas.py        # Pydantic request/response models
│   └── dependencies.py   # DI wiring
│
├── agents/
│   ├── supervisor.py     # guardrail gate + specialist selection
│   ├── guardrails.py     # input classification (fails open)
│   ├── specialists.py    # flight, hotel, weather, budget
│   ├── itinerary.py      # draft + final response
│   └── approval.py       # human-in-the-loop interrupt
│
├── graph/
│   ├── state.py          # TravelState, initial_state()
│   ├── routing.py        # AGENT_ORDER, conditional routing
│   └── workflow.py       # build_graph() → compiled StateGraph
│
├── mcp/
│   ├── client.py         # multi-server MCP client + per-server helpers
│   └── weather_server.py # bundled stdio MCP server
│
└── services/
    ├── llm.py            # single LLM seam, JSON extraction
    ├── checkpointer.py   # postgres (durable) | memory
    └── travel.py         # run / resume / serialize

frontend/                 # static assets, no build step
tests/                    # unit, integration and opt-in live tests
```

---

## Setup

Requires **Python 3.11+**.

```bash
pip install -e ".[dev]"
cp .env.example .env      # then fill in your keys
uvicorn tripmate.main:app --reload
```

### Configuration

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | LLM access for every agent |
| `TAVILY_API_KEY` | for search | Hotel and web search |
| `AVIATION_STACK_API_KEY` | for flights | Airport, airline and flight data |
| `OPENWEATHER_API_KEY` | for weather | Weather MCP server |
| `DATABASE_URL` | no | Postgres checkpointer; omit for in-memory state |
| `LOGFIRE_TOKEN` | no | Tracing stays off when unset |
| `CORS_ORIGINS` | no | Comma-separated allowed frontend origins |

Without `DATABASE_URL` the app still runs — approval state simply does not survive a restart.

---

## Testing

```bash
pytest                      # offline; no API keys or network needed
RUN_LIVE_TESTS=1 pytest tests/test_live_integrations.py   # opt-in
```

The default suite runs entirely against stubs. Live tests exercise real providers and a real Postgres checkpointer, including cross-process HITL durability, and are skipped unless `RUN_LIVE_TESTS=1`.

---

## Tech stack

Python 3.11 · FastAPI · LangGraph · LangChain · MCP · Groq · Postgres · Logfire

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
