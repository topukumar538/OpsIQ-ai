# OpsIQ — Intelligent Ops Assistant

A full-stack AI-powered operations assistant built with FastAPI and LangGraph. Features a parallel postmortem analysis pipeline, RAG over uploaded documents, and session-aware conversational AI — all streamed in real time.

Built as a portfolio project to demonstrate backend engineering, AI integration, and production-readiness skills.

🔗 **[Live Demo](https://huggingface.co/spaces/topukumar/OpsIQ)**
---

## Tech Stack

| Layer            | Technology                            |
| ---------------- | ------------------------------------- |
| Backend          | FastAPI, Python 3.12                  |
| AI Orchestration | LangGraph, LangChain                  |
| LLM              | Groq API (llama-3.3-70b-versatile)    |
| Embeddings       | HuggingFace all-MiniLM-L6-v2          |
| Vector Store     | FAISS (persisted to disk per session) |
| Database         | PostgreSQL + SQLAlchemy async         |
| Authentication   | HMAC-SHA256 stateless tokens + bcrypt |
| Rate Limiting    | slowapi                               |
| Frontend         | HTML + Vanilla JS (SSE streaming)     |
| Server           | Uvicorn                               |

---

## Project Structure

```
OpsIQ-ai/
├── backend/
│   ├── main.py                  # FastAPI app, chat/upload routes, SSE streaming
│   ├── config.py                # Pydantic settings with validation
│   ├── session.py               # Write-through session cache + LangGraph runner
│   ├── router.py                # File type classifier
│   ├── auth/
│   │   ├── models.py            # SQLAlchemy models (cascade deletes)
│   │   ├── router.py            # Signup / login / logout / me
│   │   ├── tokens.py            # HMAC-SHA256 token sign + verify
│   │   ├── dependencies.py      # FastAPI auth dependency
│   │   └── database.py          # Async engine + session factory
│   ├── graph/
│   │   ├── builder.py           # LangGraph graph definition
│   │   ├── state.py             # OpsState TypedDict
│   │   └── nodes/
│   │       ├── chat.py          # Chat node (temperature 0.7)
│   │       ├── rag.py           # RAG node (temperature 0.3)
│   │       └── postmortem.py    # Postmortem Q&A node (temperature 0.1)
│   ├── postmortem/
│   │   ├── builder.py           # Parallel postmortem pipeline
│   │   ├── ingest.py            # Log chunking + error extraction + FAISS build
│   │   ├── report.py            # Report formatting
│   │   └── nodes/
│   │       ├── log_analyzer.py  # Error + service analysis
│   │       ├── timeline.py      # Chronological event reconstruction
│   │       ├── root_cause.py    # Root cause identification
│   │       ├── remediation.py   # Action plan generation
│   │       └── report_summarizer.py  # Final report + memory seeding
│   ├── rag/
│   │   └── ingest.py            # PDF/DOCX/TXT → FAISS
│   ├── core/
│   │   ├── llm.py               # LLM factory (3 temperatures)
│   │   ├── memory.py            # LangChain memory helpers + DB persistence
│   │   ├── retriever.py         # FAISS similarity search
│   │   └── faiss_store.py       # FAISS disk persistence helpers
│   └── tests/
│       ├── conftest.py
│       ├── test_tokens.py       # 16 HMAC token tests
│       ├── test_router.py       # 16 file classifier tests
│       ├── test_ingest.py       # 17 log parsing tests
│       └── test_auth.py         # 15 auth endpoint tests
├── frontend/
│   ├── index.html               # Main app (sidebar, chat, report panel)
│   ├── login.html               # Sign in
│   └── signup.html              # Create account
├── .env.example
├── docker-compose.yml
└── requirements.txt
```

---

## Features

### Three conversation modes

| Mode           | Trigger                 | What happens                                                            |
| -------------- | ----------------------- | ----------------------------------------------------------------------- |
| **Chat**       | Send a message          | Conversational Q&A with rolling memory summarisation                    |
| **RAG**        | Upload PDF / DOCX / TXT | Document ingested into FAISS, answers grounded in content               |
| **Postmortem** | Upload `.log` file      | Parallel LangGraph pipeline — errors, timeline, root cause, remediation |

### Postmortem pipeline (parallel LangGraph execution)

```
       ┌─────────────┐
       │  Log Ingest │  chunk → embed → FAISS
       └──────┬──────┘
              │
      ┌───────┴────────┐
      ▼                ▼
[log_analyzer]    [timeline]      ← run in parallel
      └───────┬────────┘
              ▼
        [root_cause]
              ▼
        [remediation]
              ▼
      [report_summarizer]         ← seeds pm_memory for Q&A
```

### Authentication

- Signup with email + password
- Passwords hashed with bcrypt (12 rounds)
- Stateless HMAC-SHA256 session tokens stored in httponly cookies
- TTL enforced by timestamp in the token — no DB lookup on every request
- Rate limited — 10/min on login, 5/min on signup

### Session management

- Multiple sessions per user, fully independent
- Write-through in-memory cache backed by Postgres
- Server restarts are transparent — sessions restore from DB + FAISS disk automatically
- Sessions evicted from memory after idle TTL; DB + FAISS kept for reconnection
- Postmortem report persisted to DB so it survives container restarts

### Memory

- Three separate memory objects per session — chat, RAG, postmortem
- LangChain `ConversationSummaryBufferMemory` compresses old messages into a rolling summary
- Summary + last 20 raw messages persisted to Postgres on every turn
- On reconnect: summary covers all old context, raw messages cover recent context
- Chat history carries into RAG memory when switching modes mid-conversation

### Frontend

- SSE streaming for real-time responses and postmortem progress
- Collapsible, resizable postmortem report panel
- Load more pagination for message history
- Session restore on page refresh
- Markdown rendering for AI responses
- Responsive — sidebar collapses on mobile
- Custom delete confirmation modal

---

## Validation

Postmortem accuracy validated using synthetic log files reconstructed from public incident reports:

| Incident | Root Cause | OpsIQ Result |
| -------- | ---------- | ------------ |
| GitLab 2017 database deletion | Human error — accidental `rm -rf` on primary DB | ✅ Identified operator error + database + backup gap |
| Cloudflare 2019 global outage | ReDoS in WAF regex rule → CPU exhaustion | ✅ Identified regex/WAF trigger + CPU exhaustion + rollback path |
| AWS 2020 us-east-1 outage | Kinesis thread exhaustion → cascading failure | ✅ Identified cascading failure + thread exhaustion + 5 affected services |

The Cloudflare case is the most non-trivial — a regex backtracking bug causing CPU starvation across the global network is not obvious from logs alone. OpsIQ correctly surfaced the WAF rule as the trigger and CPU exhaustion as the propagation mechanism.

---

## API Endpoints

### Auth — `/auth`

```
POST /auth/signup       register new account
POST /auth/login        login, sets httponly cookie
POST /auth/logout       clears cookie
GET  /auth/me           get current user info
```

### Sessions — `/session`

```
POST   /session              create new session
GET    /sessions             list all sessions for current user
DELETE /session              delete session (removes DB row + FAISS disk)
GET    /session/mode         get current mode (chat / rag / postmortem)
GET    /session/memory       get memory summary + recent messages + report
GET    /session/messages     paginated message history (?before=id&limit=20)
```

### Chat & Upload

```
POST /chat              send a message (SSE streaming response)
POST /upload            upload a file (SSE streaming for log files)
GET  /upload/extensions accepted file types + max size
```

---

## Database Schema

```
users
  id, username, email, password_hash, created_at

sessions
  id, token, user_id (FK cascade), name, mode, is_locked,
  report_str, faiss_store_path, created_at, last_accessed_at

session_files
  id, session_id (FK cascade), filename, file_hash, created_at

session_memory
  id, session_id (FK cascade), chat_summary, rag_summary, pm_summary, updated_at

session_messages
  id, session_id (FK cascade), role, content, mode, created_at
```

---

## Setup

### Prerequisites

- Python 3.12
- PostgreSQL 15+
- Groq API key — [get one free](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/topukumar538/OpsIQ-ai
cd OpsIQ-ai/backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create PostgreSQL database

```bash
sudo -u postgres psql
```

```sql
CREATE USER myuser WITH PASSWORD 'mypassword';
CREATE DATABASE opsiq OWNER myuser;
\q
```

### 3. Create `.env` at the project root

Copy `.env.example` and fill in the required values:

```bash
cp .env.example .env
```

```dotenv
# ── Required ───────────────────────────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
DATABASE_URL=postgresql+asyncpg://myuser:mypassword@localhost:5432/opsiq
ALLOWED_ORIGINS=http://localhost:8000

# ── LLM ────────────────────────────────────────────────────────────────────────
MODEL_NAME=llama-3.3-70b-versatile
CHAT_TEMPERATURE=0.7
RAG_TEMPERATURE=0.3
PM_TEMPERATURE=0.1
MAX_TOKEN_LIMIT=2000
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── RAG ────────────────────────────────────────────────────────────────────────
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=4

# ── Postmortem ─────────────────────────────────────────────────────────────────
PM_CHUNK_LINES=30
PM_OVERLAP_LINES=5
PM_TOP_K=4

# ── Upload ─────────────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB=50

# ── Cookie ─────────────────────────────────────────────────────────────────────
COOKIE_NAME=opsiq_session
COOKIE_MAX_AGE=604800
COOKIE_SECURE=false          # set true in production (HTTPS only)
COOKIE_SAMESITE=lax
BCRYPT_ROUNDS=12

# ── Session cache ──────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS=7200
SESSION_CLEANUP_INTERVAL_SECONDS=900

# ── FAISS ──────────────────────────────────────────────────────────────────────
# /tmp works for local dev. In production point this at a mounted volume so
# stores survive container restarts (e.g. /mnt/data/opsiq_stores).
FAISS_STORE_DIR=/tmp/opsiq_stores
```

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Run

```bash
cd backend
uvicorn main:app --reload
```

Open http://localhost:8000

---

## Docker

The `docker-compose.yml` at the project root runs the full stack — Postgres + app — with a single command. Both Postgres data and FAISS vector stores are persisted in named volumes so they survive container restarts.

### 1. Create `.env` at the project root

Use the same `.env` as above. The `DATABASE_URL` in your `.env` will be overridden automatically by Docker Compose to use the internal service name (`db`), so you can leave it pointing to `localhost` for local dev — it won't affect the Docker run.

Set `FAISS_STORE_DIR=/tmp/opsiq_stores` — this matches the volume mount in `docker-compose.yml`. If you change this value, update the volume mount path in `docker-compose.yml` to match.

For Docker, also set:

```dotenv
COOKIE_SECURE=false
ALLOWED_ORIGINS=http://localhost:8000
```

### 2. Build and run

```bash
docker compose up --build
```

Open http://localhost:8000

### 3. Stop

```bash
# Stop containers, keep volumes (data preserved)
docker compose down

# Stop and delete all data
docker compose down -v
```

---

## Running Tests

Tests are run manually inside the `backend/` directory.

```bash
cd backend

# Fast — no DB needed
pytest tests/test_tokens.py tests/test_router.py tests/test_ingest.py -v

# Auth integration tests — requires a running Postgres instance
pytest tests/test_auth.py -v

# All tests
pytest tests/ -v
```

---

## Key Design Decisions

**Stateless HMAC tokens** Session cookies are signed with HMAC-SHA256 and verified without a DB lookup on every request. TTL is enforced by a timestamp embedded in the token itself. No session table lookups on hot paths.

**Three LLM temperatures per session** Chat (0.7) for natural conversation, RAG (0.3) for accurate document Q&A, postmortem (0.1) for near-deterministic reproducible analysis. Each is a separate cached instance created at session startup.

**Write-through session cache** In-memory dict for fast access, Postgres as source of truth. Server restarts are transparent — sessions restore from DB + FAISS disk stores automatically. TTL eviction only drops from memory; data stays on disk for reconnection.

**Parallel LangGraph nodes** `log_analyzer` and `timeline` run concurrently since they have independent inputs, reducing postmortem latency. Root cause analysis then uses both outputs.

**FAISS path isolation** Stores keyed by `user_id/session_token/kind` — two users with identical tokens (impossible but defensive) cannot cross-pollute each other's vector stores.

**Chat history carries into RAG** When a user uploads a file mid-conversation, the existing chat memory (summary + raw messages) is copied into RAG memory. Context like names and prior discussion is not lost on mode switch.

**Sync graph runner in thread pool** LangGraph's `.invoke()` is blocking. `run_in_executor` moves it off the async event loop onto a thread pool worker so other users' requests are never frozen during a long postmortem run.

---

## Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `GROQ_API_KEY` | required | Groq API key |
| `SECRET_KEY` | required | Min 32 chars — signs session tokens |
| `DATABASE_URL` | required | PostgreSQL async connection string |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | Comma-separated CORS origins |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model |
| `CHAT_TEMPERATURE` | `0.7` | Chat LLM temperature |
| `RAG_TEMPERATURE` | `0.3` | RAG LLM temperature |
| `PM_TEMPERATURE` | `0.1` | Postmortem LLM temperature |
| `MAX_TOKEN_LIMIT` | `2000` | LangChain memory token limit before summarisation |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embeddings model |
| `RAG_CHUNK_SIZE` | `500` | Token chunk size for RAG ingestion |
| `RAG_CHUNK_OVERLAP` | `50` | Chunk overlap for RAG ingestion |
| `RAG_TOP_K` | `4` | Number of RAG chunks retrieved per query |
| `PM_CHUNK_LINES` | `30` | Log lines per postmortem chunk |
| `PM_OVERLAP_LINES` | `5` | Overlap lines between postmortem chunks |
| `PM_TOP_K` | `4` | Number of postmortem chunks retrieved per query |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |
| `COOKIE_NAME` | `opsiq_session` | Session cookie name |
| `COOKIE_MAX_AGE` | `604800` | Cookie lifetime in seconds (7 days) |
| `COOKIE_SECURE` | `false` | Set `true` in production (HTTPS) |
| `COOKIE_SAMESITE` | `lax` | CSRF protection |
| `BCRYPT_ROUNDS` | `12` | bcrypt work factor |
| `SESSION_TTL_SECONDS` | `7200` | Idle session eviction from memory |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `900` | Cleanup task interval |
| `FAISS_STORE_DIR` | `/tmp/opsiq_stores` | FAISS persistence directory — use a mounted volume in production |
| `DB_SCHEMA` | `opsiq` | Postgres schema name |
