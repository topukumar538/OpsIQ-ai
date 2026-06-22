# OpsIQ — Intelligent Ops Assistant

A full-stack AI-powered operations assistant built with FastAPI and LangGraph. Features a parallel postmortem analysis pipeline, RAG over uploaded documents, and session-aware conversational AI — all streamed in real time.

Built as a portfolio project to demonstrate backend engineering, AI integration, and production-readiness skills.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.12 |
| AI Orchestration | LangGraph, LangChain |
| LLM | Groq API (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (persisted to disk per session) |
| Database | PostgreSQL + SQLAlchemy async |
| Authentication | HMAC-SHA256 stateless tokens + bcrypt |
| Rate Limiting | slowapi |
| Frontend | HTML + Vanilla JS (SSE streaming) |
| Server | Uvicorn |

---

## Project Structure

```
postmortem-ai/
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
└── frontend/
    ├── index.html               # Main app (sidebar, chat, report panel)
    ├── login.html               # Sign in
    └── signup.html              # Create account
```

---

## Features

### Three conversation modes

| Mode | Trigger | What happens |
|------|---------|-------------|
| **Chat** | Send a message | Conversational Q&A with rolling memory summarisation |
| **RAG** | Upload PDF / DOCX / TXT | Document ingested into FAISS, answers grounded in content |
| **Postmortem** | Upload `.log` file | Parallel LangGraph pipeline — errors, timeline, root cause, remediation |

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
- Postmortem report persisted to DB so it survives Railway redeploys

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
- Session restore on page refresh (localStorage)
- Markdown rendering for AI responses
- Responsive — sidebar collapses on mobile
- Custom delete confirmation modal

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
git clone https://github.com/topukumar538/postmortem-ai
cd postmortem-ai/backend
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

### 3. Create `.env` file inside `backend/`

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

### 4. Run

```bash
cd backend
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Running Tests

```bash
cd backend

# Fast — no DB needed
pytest tests/test_tokens.py tests/test_router.py tests/test_ingest.py -v

# Auth integration tests — needs Postgres
pytest tests/test_auth.py -v

# All tests
pytest tests/ -v
```


```env
GROQ_API_KEY=...
SECRET_KEY=...
DATABASE_URL=<auto-filled by Railway Postgres plugin>
ALLOWED_ORIGINS=https://yourapp.up.railway.app
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
```

5. Set start command:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Key Design Decisions

**Stateless HMAC tokens**
Session cookies are signed with HMAC-SHA256 and verified without a DB lookup on every request. TTL is enforced by a timestamp embedded in the token itself. No session table lookups on hot paths.

**Three LLM temperatures per session**
Chat (0.7) for natural conversation, RAG (0.3) for accurate document Q&A, postmortem (0.1) for near-deterministic reproducible analysis. Each is a separate cached instance created at session startup.

**Write-through session cache**
In-memory dict for fast access, Postgres as source of truth. Server restarts are transparent — sessions restore from DB + FAISS disk stores automatically. TTL eviction only drops from memory; data stays on disk for reconnection.

**Parallel LangGraph nodes**
`log_analyzer` and `timeline` run concurrently since they have independent inputs, reducing postmortem latency. Root cause analysis then uses both outputs.

**FAISS path isolation**
Stores keyed by `user_id/session_token/kind` — two users with identical tokens (impossible but defensive) cannot cross-pollute each other's vector stores.

**Chat history carries into RAG**
When a user uploads a file mid-conversation, the existing chat memory (summary + raw messages) is copied into RAG memory. Context like names and prior discussion is not lost on mode switch.

**Sync graph runner in thread pool**
LangGraph's `.invoke()` is blocking. `run_in_executor` moves it off the async event loop onto a thread pool worker so other users' requests are never frozen during a long postmortem run.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq API key |
| `SECRET_KEY` | required | Min 32 chars — signs session tokens |
| `DATABASE_URL` | required | PostgreSQL async connection string |
| `ALLOWED_ORIGINS` | `http://localhost:8000` | Comma-separated CORS origins |
| `COOKIE_SECURE` | `false` | Set `true` in production (HTTPS) |
| `COOKIE_SAMESITE` | `lax` | CSRF protection |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model |
| `CHAT_TEMPERATURE` | `0.7` | Chat LLM temperature |
| `RAG_TEMPERATURE` | `0.3` | RAG LLM temperature |
| `PM_TEMPERATURE` | `0.1` | Postmortem LLM temperature |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |
| `SESSION_TTL_SECONDS` | `7200` | Idle session eviction from memory |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `900` | Cleanup task interval |
| `FAISS_STORE_DIR` | `/tmp/opsiq_stores` | FAISS persistence directory |
| `DB_SCHEMA` | `opsiq` | Postgres schema name |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embeddings model |
| `MAX_TOKEN_LIMIT` | `2000` | LangChain memory token limit before summarisation |