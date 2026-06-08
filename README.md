# Policy Support Bot

A production-ready AI customer-support bot for **policy queries** (insurance, terms-and-conditions, coverage documents).

Built with **FastAPI + LangGraph + PageIndex (MongoDB) + Groq llama-3.1-8b-instant**, packaged with **uv**.

This project mirrors the architecture of the apple_chat_bot reference but is fully isolated, single-provider (Groq), and focused on the policy domain.

---

## Architecture

```
POST /query
  │
  ├─ Middleware: JWT auth · rate limiting (slowapi) · input guard
  │
  ├─ Semantic cache check (GPTCache server)
  │     └─ HIT → return immediately
  │
  └─ LangGraph graph (LangSmith traces everything)
       ├─ safety_gate       Presidio PII scrub + Rival attack detection (parallel)
       ├─ query_intelligence  1 structured Groq call → intent, sub_queries, complexity
       ├─ session_memory    LangGraph PostgresSaver checkpointer (per-session_id)
       ├─ context_retrieval PageIndex tree search over MongoDB (policy_bot.trees)
       ├─ execution         Groq llama-3.1-8b-instant — single / parallel sub-queries (Send API)
       ├─ output_validation Custom Groq faithfulness + completeness LLM-as-judge
       └─ cache_store       GPTCache write + structlog summary
```

## Stack

| Layer | Tool |
|---|---|
| API framework | FastAPI |
| Graph orchestration | LangGraph |
| Package manager | **uv** |
| Auth | python-jose |
| Rate limiting | slowapi |
| PII scrubbing | presidio-analyzer + presidio-anonymizer |
| Attack detection | rival-ai (Bhairava-0.4B, separate microservice) |
| Semantic caching | GPTCache (server mode) |
| RAG retrieval | PageIndex tree (MongoDB `policy_bot.trees`) |
| Session memory | LangGraph PostgresSaver + asyncpg |
| LLM (everywhere) | **Groq llama-3.1-8b-instant** (langchain-groq) |
| Faithfulness check | Custom LLM-as-judge (Groq) — replaces Ragas/OpenAI |
| Completeness check | Custom LLM-as-judge (Groq) |
| Observability (LLM) | LangSmith |
| Observability (app) | structlog |
| Retries | tenacity |
| Circuit breaking | pybreaker |
| HTTP client | httpx |

## Quickstart

### 1. Install uv (if you don't have it)

```bash
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and configure

```bash
cd policy_chat_bot
cp .env.example .env
# Fill in GROQ_API_KEY (and LANGCHAIN_API_KEY if you want LangSmith traces)
uv sync
```

### 3. Ingest your policy PDF (one-time)

The ingestion step builds a PageIndex tree and stores it at MongoDB
`policy_bot.trees` with `doc_name="policy_dc"`. **You have already done this step.**

If you ever need to re-ingest:

```bash
# Local PageIndex (uses Groq llama-3.1-8b-instant)
uv run --extra ingest python -m prep.build_tree

# OR: hosted PageIndex API (higher quality, requires PAGEINDEX_API_KEY)
PAGEINDEX_API_KEY=... uv run --extra ingest python -m prep.build_tree_api
```

Both write to `policy_bot.trees` with `{ "doc_name": "policy_dc", "tree": [...] }`.

### 4. Run locally

```bash
# Make sure MongoDB is running on localhost:27017 with policy_bot.trees populated
# Make sure Postgres is running on localhost:5432 with database `policy_bot`

uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. (Optional) Full docker-compose stack

```bash
docker compose up --build
```

This starts: `app` (8000), `rival-service` (8002), `gptcache` (8001),
`mongodb` (27017), `postgres` (5432).

### 6. Make a request

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Does my policy cover water damage from a burst pipe?", "session_id": "session-abc"}'
```

## Project structure

```
policy_chat_bot/
├── main.py                        # FastAPI app, /query endpoint
├── config.py                      # Settings via pydantic-settings
├── pyproject.toml                 # uv package metadata + deps
├── .python-version                # 3.11
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── graph/
│   ├── state.py                   # SupportBotState TypedDict
│   ├── graph.py                   # StateGraph definition + compile
│   └── nodes/
│       ├── safety_gate.py         # Presidio + Rival (parallel)
│       ├── query_intelligence.py  # Structured Groq call
│       ├── session_memory.py      # History trimming
│       ├── context_retrieval.py   # PageIndex + MongoDB (policy_bot.trees)
│       ├── execution.py           # Single + Send fan-out (Groq)
│       ├── output_validation.py   # Faithfulness + completeness (parallel)
│       └── cache_store.py         # GPTCache write + final log
├── services/
│   └── rival_service/
│       ├── main.py                # Standalone FastAPI microservice
│       ├── requirements.txt
│       └── Dockerfile
├── metrics/
│   ├── completeness.py            # LLM-as-judge completeness (Groq)
│   └── faithfulness.py            # LLM-as-judge faithfulness (Groq)
├── prompts/
│   └── v1/
│       ├── query_intelligence.txt
│       ├── generation.txt
│       ├── completeness_judge.txt
│       └── faithfulness_judge.txt
├── prep/
│   ├── build_tree.py              # Local pageindex (Groq) → MongoDB
│   └── build_tree_api.py          # Hosted PageIndex API → MongoDB
├── resilience/
│   ├── breakers.py                # pybreaker circuit breakers
│   └── retry.py                   # tenacity retry decorators
├── middleware/
│   ├── auth.py                    # JWT verification
│   ├── rate_limit.py              # slowapi limiter
│   └── input_guard.py             # Length + encoding check
└── observability/
    └── logging.py                 # structlog setup
```

## Configuration

All configuration is via environment variables. See `.env.example` for the full list.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | _(required)_ | Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Single model used everywhere |
| `MONGO_DB` | `policy_bot` | Mongo database holding the tree |
| `MONGO_COLLECTION` | `trees` | Collection holding the tree document |
| `POLICY_DOC_NAME` | `policy_dc` | `doc_name` used to look up the tree |
| `FAITHFULNESS_THRESHOLD` | `0.7` | Below this triggers a validation warning |
| `COMPLETENESS_THRESHOLD` | `0.6` | Below this triggers a validation warning |
| `MAX_INPUT_CHARS` | `4000` | Queries longer than this are rejected with 400 |
| `MAX_SESSION_TURNS` | `10` | Conversation turns to keep in context |

## How retrieval works

1. Lookup `policy_bot.trees` by `doc_name="policy_dc"`.
2. Strip the `text` field from every node so the search prompt stays small.
3. Ask Groq llama-3.1-8b-instant which `node_id`s likely contain the answer
   (returns JSON like `{"node_list": ["1.2", "3.1"]}`).
4. Resolve those IDs back to full node objects (with text) via a flat node map.
5. Return the list of text chunks as `retrieved_context`.

The node tolerates both ingestion shapes:
- raw list of nodes (from `prep/build_tree.py`)
- the full PageIndex API result dict (from `prep/build_tree_api.py`)

## Differences vs the apple_chat_bot reference

| Concern | Reference (apple) | This project (policy) |
|---|---|---|
| LLM provider | Gemini (Flash + Pro) + OpenAI for Ragas | **Groq llama-3.1-8b-instant only** |
| Faithfulness | Ragas + GPT-4o-mini | Custom LLM-as-judge using Groq |
| Package manager | pip + requirements.txt | **uv + pyproject.toml** |
| MongoDB layout | `support_bot.document_trees` keyed by `doc_id` | `policy_bot.trees` keyed by `doc_name` |
| Domain | Apple devices & services | Policy / insurance customer support |

## Circuit breakers and fallbacks

| Dependency | Breaker opens after | Fallback behaviour |
|---|---|---|
| Rival (attack detection) | 5 failures | Allow request through, log warning |
| PageIndex / MongoDB | 5 failures | Empty context, LLM answers without grounding |
| GPTCache | 10 failures | Skip cache, continue normally |
| Groq LLM | — (tenacity x3) | 503 to user |
