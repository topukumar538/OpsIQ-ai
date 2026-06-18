# OpsIQ — Intelligent Ops Assistant

An AI-powered operations assistant for DevOps and SRE teams. Chat with your runbooks, query uploaded documents via RAG, and run automated postmortem analysis on incident log files — all in one session-aware interface.

![OpsIQ](https://img.shields.io/badge/stack-FastAPI%20%7C%20LangGraph%20%7C%20FAISS%20%7C%20Groq-4f8ef7?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## What it does

| Mode | Trigger | What happens |
|------|---------|-------------|
| **Chat** | Send a message | Conversational Q&A with rolling memory summarisation |
| **RAG** | Upload PDF / DOCX / TXT | Document ingested into FAISS, answers grounded in content |
| **Postmortem** | Upload `.log` file | Parallel LangGraph pipeline analyzes errors, timeline, root cause, and remediation |

Sessions are persistent — switch sessions, restart the server, come back later. Everything restores from Postgres + FAISS disk stores.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Backend                     │
│                                                         │
│  ┌──────────┐   ┌────────────┐   ┌──────────────────┐  │
│  │   Auth   │   │  Session   │   │   Upload Route   │  │
│  │  Router  │   │  Manager   │   │  (SSE streaming) │  │
│  └──────────┘   └────────────┘   └──────────────────┘  │
│       │               │                   │             │
│  HMAC tokens    Write-through        LangGraph          │
│  bcrypt hash    cache + Postgres     Pipeline           │
│  Rate limited   FAISS disk store                        │
│                       │                   │             │
│              ┌────────┴────────┐          │             │
│              │   Three LLMs   │◄──────────┘             │
│              │ chat  0.7 temp │                         │
│              │ rag   0.3 temp │  Groq API               │
│              │ pm    0.1 temp │  llama-3.3-70b          │
│              └───────────────-┘                         │
└─────────────────────────────────────────────────────────┘
```

### Postmortem pipeline (LangGraph parallel execution)

```
         ┌─────────────┐
         │  Log Ingest │  chunk → embed → FAISS
         └──────┬──────┘
                │
        ┌───────┴────────┐
        ▼                ▼
  [log_analyzer]   [timeline]     ← run in parallel
        └───────┬────────┘
                ▼
          [root_cause]
                ▼
          [remediation]
                ▼
        [report_summarizer]       ← seeds pm_memory for Q&A
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.12 |
| AI orchestration | LangGraph, LangChain |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector store | FAISS (persisted to disk per session) |
| Database | PostgreSQL + SQLAlchemy async |
| Auth | HMAC-SHA256 stateless tokens, bcrypt |
| Rate limiting | slowapi |
| Frontend | Vanilla JS, SSE streaming |
| Hosting | Railway |

---

## Key engineering decisions

**Stateless HMAC tokens** — session cookies are signed with HMAC-SHA256 and verified without a DB lookup on every request. TTL enforced by timestamp in the token itself.

**Three LLM instances per session** — chat (0.7), RAG (0.3), and postmortem (0.1) each use a different temperature cached at session creation. Postmortem analysis is near-deterministic; chat is conversational.

**Write-through session cache** — in-memory `dict` for fast access, Postgres as source of truth. Server restarts are transparent — sessions restore from DB + FAISS disk stores automatically.

**FAISS path isolation** — stores are keyed by `user_id/session_token/kind` so two users with identical tokens (theoretically impossible but defensive) cannot cross-pollute.

**Parallel LangGraph nodes** — `log_analyzer` and `timeline` run concurrently since they have no shared inputs, reducing postmortem latency.

**Chat history carries into RAG** — when a user uploads a file mid-conversation, the existing `chat_memory` is copied into `rag_memory` so context (names, prior discussion) isn't lost on mode switch.

---

## Project structure

```
postmortem-ai/
├── backend/
│   ├── main.py               # FastAPI app, upload/chat routes
│   ├── config.py             # Pydantic settings with validation
│   ├── session.py            # Session cache + LangGraph runner
│   ├── router.py             # File type classifier
│   ├── auth/
│   │   ├── models.py         # SQLAlchemy models (cascade deletes)
│   │   ├── router.py         # Signup / login / logout / me
│   │   ├── tokens.py         # HMAC token sign + verify
│   │   └── dependencies.py   # FastAPI auth dependency
│   ├── graph/
│   │   ├── builder.py        # LangGraph graph definition
│   │   ├── state.py          # OpsState TypedDict
│   │   └── nodes/
│   │       ├── chat.py
│   │       ├── rag.py
│   │       └── postmortem.py
│   ├── postmortem/
│   │   ├── builder.py        # Parallel postmortem pipeline
│   │   ├── ingest.py         # Log chunking + error extraction
│   │   ├── report.py         # Report formatting
│   │   └── nodes/            # log_analyzer, timeline, root_cause, remediation
│   ├── rag/
│   │   └── ingest.py         # PDF/DOCX/TXT → FAISS
│   ├── core/
│   │   ├── llm.py            # LLM factory (3 temperatures)
│   │   ├── memory.py         # LangChain memory helpers
│   │   ├── retriever.py      # FAISS retrieval
│   │   └── faiss_store.py    # FAISS disk persistence
│   └── tests/
│       ├── test_tokens.py    # 16 token tests
│       ├── test_router.py    # 16 classifier tests
│       ├── test_ingest.py    # 17 log parsing tests
│       └── test_auth.py      # 15 auth endpoint tests
└── frontend/
    ├── index.html            # Main app (sidebar, chat, report panel)
    ├── login.html
    └── signup.html
```

---

## Local setup

### Prerequisites
- Python 3.12
- PostgreSQL 15+
- Groq API key ([get one free](https://console.groq.com))

### 1. Clone and install

```bash
git clone https://github.com/yourname/postmortem-ai
cd postmortem-ai/backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=generate_with_python_c_import_secrets_print_secrets_token_hex_32
DATABASE_URL=postgresql+asyncpg://myuser:mypassword@localhost:5432/opsiq
ALLOWED_ORIGINS=http://localhost:8000
```

Generate a secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Set up Postgres

```bash
sudo -u postgres psql
```

```sql
CREATE USER myuser WITH PASSWORD 'mypassword';
CREATE DATABASE opsiq OWNER myuser;
\q
```

### 4. Run

```bash
cd backend
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Running tests

```bash
cd backend
pytest tests/test_tokens.py tests/test_router.py tests/test_ingest.py -v   # no DB needed
pytest tests/test_auth.py -v                                                 # needs Postgres
pytest tests/ -v                                                             # all tests
```

---

## Deploying to Railway

1. Push to GitHub
2. Create a new Railway project → deploy from GitHub
3. Add a PostgreSQL plugin
4. Set environment variables:

```env
GROQ_API_KEY=...
SECRET_KEY=...
DATABASE_URL=<auto-filled by Railway Postgres plugin>
ALLOWED_ORIGINS=https://yourapp.up.railway.app
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
```

5. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## Security

- Passwords hashed with bcrypt (12 rounds)
- Session tokens signed with HMAC-SHA256, TTL enforced server-side
- Rate limiting on auth endpoints (5/min signup, 10/min login)
- CORS restricted to explicit origin list
- Input length limits on all endpoints
- Session ownership verified on every request
- Cascade deletes on all foreign keys
- Temp files cleaned up in `finally` blocks

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq API key |
| `SECRET_KEY` | required | Min 32 chars, used to sign session tokens |
| `DATABASE_URL` | required | PostgreSQL async connection string |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | Comma-separated CORS origins |
| `COOKIE_SECURE` | `false` | Set `true` in production (HTTPS only) |
| `COOKIE_SAMESITE` | `lax` | CSRF protection |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |
| `SESSION_TTL_SECONDS` | `7200` | In-memory session TTL (2 hours) |
| `FAISS_STORE_DIR` | `/tmp/opsiq_stores` | FAISS persistence directory |
| `DB_SCHEMA` | `opsiq` | Postgres schema name |