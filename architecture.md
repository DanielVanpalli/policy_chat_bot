# Policy Support Bot — Architecture Diagram (Azure AI Foundry)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           POLICY SUPPORT BOT — AZURE AI FOUNDRY                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

  CLIENT
  ──────
  Browser / Mobile / API Consumer
        │
        │  POST /query  (Bearer JWT)
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                       API LAYER  (FastAPI)                        │
│                                                                   │
│   ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│   │  auth_middleware │  │ input_guard_mid  │  │ slowapi        │  │
│   │  (JWT / HS256)  │  │ (max 4000 chars) │  │ 30 req/min     │  │
│   │  → Azure EntraID│  └──────────────────┘  │ → Redis store  │  │
│   └─────────────────┘                        └────────────────┘  │
│                                                                   │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │                 Semantic Cache Check                      │   │
│   │          Azure Cache for Redis (GPTCache layer)           │   │
│   │    hit ─────────────────────────────────► return early    │   │
│   │    miss ──────────────────────────────────────────────┐   │   │
│   └───────────────────────────────────────────────────────┼───┘   │
└───────────────────────────────────────────────────────────┼───────┘
                                                            │
                                                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH PIPELINE                              │
│                (PostgreSQL checkpointer)                          │
│                                                                   │
│  STAGE 1 — SAFETY  (parallel)                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐  │
│  │    pii_scrub         │    │         attack_detect            │  │
│  │  Microsoft Presidio  │    │  Azure AI Foundry                │  │
│  │  (spaCy en_core_web) │    │  llama-3.1-8b  [guard prompt]    │  │
│  │  → scrubbed_query    │    │  → is_attack, confidence         │  │
│  └──────────┬──────────┘    └──────────────┬───────────────────┘  │
│             └──────────────┬───────────────┘                      │
│                            ▼                                      │
│                    safety_merge                                    │
│                 is_attack? → 403 END                               │
│                            │                                      │
│  STAGE 2 — INTELLIGENCE                                           │
│                            ▼                                      │
│              ┌─────────────────────────┐                          │
│              │    query_intelligence   │                          │
│              │  Azure AI Foundry       │                          │
│              │  llama-3.1-8b           │                          │
│              │  structured output →    │                          │
│              │  intent, sub_queries,   │                          │
│              │  complexity, needs_decomp│                         │
│              └───────────┬─────────────┘                          │
│                          │                                        │
│                          ▼                                        │
│              ┌─────────────────────────┐                          │
│              │    session_memory       │                          │
│              │  trim to MAX_TURNS=10   │                          │
│              │  (PostgreSQL history)   │                          │
│              └───────────┬─────────────┘                          │
│                                                                   │
│  STAGE 3 — RETRIEVAL                                              │
│                          ▼                                        │
│              ┌────────────────────────────────────────────────┐   │
│              │         context_retrieval                      │   │
│              │                                                │   │
│              │  BM25 (keyword, k=4, w=0.4)                   │   │
│              │          +                                     │   │
│              │  FAISS dense (all-MiniLM-L6-v2, k=4, w=0.6)  │   │
│              │          │                                     │   │
│              │          ▼  EnsembleRetriever (8 candidates)  │   │
│              │  cross-encoder rerank → top 4 chunks          │   │
│              │  (ms-marco-MiniLM-L-6-v2)                     │   │
│              │                                                │   │
│              │  [pageindex_breaker: fails=5, reset=30s]      │   │
│              └───────────┬────────────────────────────────────┘   │
│                                                                   │
│  STAGE 4 — EXECUTION  (conditional routing)                       │
│                          ▼                                        │
│         route_execution()                                         │
│           ├── complexity=low  ──────► generate_flash              │
│           ├── complexity=high ──────► generate_pro                │
│           └── needs_decomp=true ────► generate_subquery × N       │
│                                           (Send fan-out)          │
│                                                                   │
│  All execution nodes use:                                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                  Azure AI Foundry                           │  │
│  │                                                             │  │
│  │    Endpoint: https://<hub>.services.ai.azure.com/          │  │
│  │    Deployment: llama-3.1-8b-instant  (or GPT-4o-mini)      │  │
│  │    Auth: Azure Managed Identity / API Key                   │  │
│  │    Retry: tenacity (3 attempts, 2–20s exp backoff)         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│           generate_subquery × N ──► merge_subqueries             │
│                                                                   │
│  STAGE 5 — VALIDATION  (parallel)                                 │
│              ┌────────────────┐    ┌───────────────────────────┐  │
│              │  faithfulness  │    │       completeness        │  │
│              │  Azure AI Fndry│    │      Azure AI Foundry     │  │
│              │  LLM-as-judge  │    │      LLM-as-judge         │  │
│              │  threshold=0.7 │    │      threshold=0.6        │  │
│              └───────┬────────┘    └──────────────┬────────────┘  │
│                      └─────────────┬──────────────┘              │
│                                    ▼                              │
│                           validation_merge                        │
│                           → final_response                        │
│                                    │                              │
│                                    ▼                              │
│                           cache_store                             │
│                     (write to Azure Cache for Redis)              │
└────────────────────────────────────┬──────────────────────────────┘
                                     │
                                     ▼
                              QueryResponse
                     (response, scores, session_id, model_used)


═══════════════════════════════════════════════════════════════════════
                    AZURE SERVICES MAP
═══════════════════════════════════════════════════════════════════════

  ┌────────────────────────────────────────────────────────────────┐
  │                  Azure AI Foundry Hub                          │
  │                                                                │
  │  ┌──────────────────────────────────────────────────────────┐  │
  │  │  Model Deployments                                       │  │
  │  │  ┌──────────────────┐  ┌──────────────────────────────┐  │  │
  │  │  │ llama-3.1-8b     │  │  GPT-4o-mini (optional)      │  │  │
  │  │  │ guard / intel /  │  │  upgrade path for generate   │  │  │
  │  │  │ generate / judge │  │  _pro node only              │  │  │
  │  │  └──────────────────┘  └──────────────────────────────┘  │  │
  │  └──────────────────────────────────────────────────────────┘  │
  │  Content Safety filter  │  Managed Identity auth              │
  │  Azure Monitor / App Insights traces                           │
  └────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐   ┌──────────────────────────────────────┐
  │  Azure Database for  │   │  Azure Cache for Redis               │
  │  PostgreSQL Flexible │   │                                      │
  │  - LangGraph session │   │  - Semantic cache (GPTCache layer)   │
  │    checkpointer      │   │  - slowapi distributed rate limit    │
  │  - connection pool   │   │  - TTL-based expiry                  │
  │    (2–10 async)      │   └──────────────────────────────────────┘
  └──────────────────────┘

  ┌────────────────────────┐  ┌───────────────────────────────────┐
  │  Azure Container Apps  │  │  Azure Blob Storage               │
  │  (or AKS)              │  │                                   │
  │  - app service         │  │  - FAISS index files              │
  │  - rival-service       │  │    (mounted at startup or baked   │
  │  - gptcache sidecar    │  │     into container image)         │
  │  - auto-scale replicas │  └───────────────────────────────────┘
  └────────────────────────┘

  ┌────────────────────────────────────────────────────────────────┐
  │  Azure Monitor + Application Insights                          │
  │  - structlog → Log Analytics Workspace                        │
  │  - LangSmith traces (or replace with Azure AI Foundry evals)  │
  │  - Circuit breaker / retry telemetry                          │
  └────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                    INGESTION PIPELINE  (offline)
═══════════════════════════════════════════════════════════════════════

  policy_dc.pdf
       │
       ▼
  parse_doc.py  (docling — layout-aware: headers, tables, lists)
       │
       ├──► chunk_doc.py         (semantic chunks → FAISS)
       └──► chunk_structural.py  (structural chunks → faiss_index_structural/)
                │
                ▼
          build_index.py
          → FAISS index (all-MiniLM-L6-v2 embeddings)
          → Upload to Azure Blob Storage
          → Tree metadata → MongoDB (policy_bot.trees)
```

---

## Key changes: Groq → Azure AI Foundry

| Component | Current (Groq) | With Azure AI Foundry |
|---|---|---|
| LLM API | `langchain-groq` / `AsyncGroq` | `langchain-openai` with Azure endpoint, or `azure-ai-inference` SDK |
| Auth | API key in `.env` | Azure Managed Identity (no secrets in env) |
| Model | `llama-3.1-8b-instant` | Deploy same model in Foundry, or swap `generate_pro` → GPT-4o-mini |
| Content safety | Manual attack_detect node | Azure AI Content Safety can be layered on top |
| Observability | LangSmith | Azure Monitor + App Insights (or keep LangSmith) |
| Rate limiting | In-process slowapi | slowapi + Azure Cache for Redis backend |
| FAISS index | Local disk | Azure Blob Storage mount, or migrate to Azure AI Search |

### Code change surface (5 files)

Swap `ChatGroq(model=...)` → `AzureChatOpenAI(azure_deployment=..., azure_endpoint=..., api_version=...)` in:

- `graph/nodes/execution.py`
- `graph/nodes/query_intelligence.py`
- `graph/nodes/safety_gate.py`
- `metrics/faithfulness.py`
- `metrics/completeness.py`
