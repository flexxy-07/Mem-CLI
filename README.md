# MemCLI — Memory Management in Agents

> A hands-on, production-modelled study of how **persistent memory is built, managed, and governed** in real-world AI agents — using PostgreSQL, pgvector, LangGraph, and Google Gemini.

---

## Why This Project Exists

One of the hardest unsolved problems in applied LLM engineering is **memory**. Most tutorials show you how to call an LLM. Almost none show you how to make it remember — across sessions, at scale, with accuracy.

MemCLI is a ground-up implementation of a **production-grade memory system for LLM agents**, built as a learning exercise to explore and answer questions like:

- How do you store and retrieve memories **semantically**, not just by keyword?
- How do you prevent the same fact being saved 50 times in paraphrased form (**deduplication**)?
- What happens when a user changes their mind — how do you **resolve conflicts** between old and new facts without deleting history?
- How do you separate **in-session context** (short-term) from **persistent knowledge** (long-term) in a LangGraph agent?
- How do you scale vector search to thousands of memories efficiently using **HNSW indexing**?
- How do you give users **full control** over their stored data (GDPR-style: export, forget, wipe)?

The CLI interface and chat capability are just a vehicle to demonstrate these memory concepts in action — not the end goal.

---

## Architecture

```
                  ┌──────────────────────────────────┐
                  │         Incoming User Turn         │
                  └─────────────────┬────────────────-─┘
                                    │
          ┌─────────────────────────┴───────────────────────┐
          │                                                 │
  [Short-Term Memory]                             [Long-Term Memory]
  LangGraph MemorySaver                          PostgreSQL + pgvector
  Session buffer of last N turns                 HNSW-indexed 3072-dim vectors
  Resolves pronouns & context                    Retrieves relevant past facts
          │                                                 │
          └─────────────────────────┬───────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  generate_response  │  ← SystemPrompt(LTM) + Messages(STM)
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  extract_memories   │  ← LLM identifies new enduring facts
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │                                            │
   [Semantic Deduplication]                  [Contradiction Resolution]
   similarity ≥ 0.95 → refresh              0.75 ≤ sim < 0.95 → LLM arbiter
   existing row only                         Conflict? → supersede old memory
              │                                            │
              └─────────────────────┬──────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  PostgreSQL (HNSW)  │
                         └─────────────────────┘
```

---

## Features

### 🧠 Dual-Tier Memory Architecture
| Tier | Storage | Scope | What it solves |
| :--- | :--- | :--- | :--- |
| **Short-Term** | LangGraph `MemorySaver` (in-process) | Active session only | Multi-turn context, pronoun resolution, dialogue flow |
| **Long-Term** | PostgreSQL 17 + pgvector | Permanent across sessions | Persistent knowledge retrieval across any session |

### ⚡ HNSW Vector Indexing
- Embeddings are 3,072-dimensional vectors from `gemini-embedding-2`.
- Standard HNSW on pgvector is limited to 2000 dimensions. MemCLI works around this using **`halfvec(3072)`** (16-bit floats), enabling full HNSW (`m=16`, `ef_construction=64`) for **sub-millisecond ANN search** at scale.
- This is a real production consideration — naive `ORDER BY embedding <=> query LIMIT 5` becomes a sequential scan at scale without it.

### 🔁 Semantic Deduplication
- A naive memory system would store *"The user prefers Java"* and *"The user likes using Java in interviews"* as two separate rows.
- MemCLI computes cosine similarity before each write. If similarity ≥ 0.95, it **refreshes the existing row** rather than inserting a duplicate — keeping the memory store clean regardless of phrasing.

### ⚔️ Contradiction & Conflict Resolution
- Additive memory systems accumulate stale facts. When a user changes their mind, both the old and new memory coexist and the LLM gets confused.
- MemCLI uses an **LLM arbiter** (`arbiter.py`) to decide if a new fact supersedes an old one. Superseded memories are archived (`is_active = FALSE`, `superseded_by = new_id`) — auditable, never hard-deleted.

### 🔒 Data Governance (Production Requirement)
- Production memory systems must give users control over their data.
- Full GDPR-style controls via CLI: **`forget`** (by ID or semantic search), **`export`** (JSON or CSV), **`clear`** (full wipe).

---

## Project Structure

```text
memCLI/
├── docker-compose.yml          # PostgreSQL 17 + pgvector containerized stack
├── pyproject.toml              # Package config & entry points
├── requirements.txt            # Pinned dependencies
├── .env.example                # Environment template
├── ROADMAP.md                  # Architectural roadmap & milestone checklist
└── src/
    └── memcli/
        ├── __init__.py
        ├── agent.py            # LangGraph dual-tier memory workflow
        ├── arbiter.py          # LLM-based contradiction resolution
        ├── cli.py              # Full CLI: chat, list, search, forget, export, clear
        ├── database.py         # Engine, init_db(), HNSW migration
        ├── embeddings.py       # Gemini embedding service (3072-dim)
        └── memory.py           # Core memory CRUD: save, search, dedup, export, forget
```

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | Google Gemini (`langchain-google-genai`) |
| **Embeddings** | `gemini-embedding-2` — 3072 dimensions |
| **Vector Store** | PostgreSQL 17 + pgvector 0.8 |
| **Vector Index** | HNSW via `halfvec(3072)` |
| **Agent Framework** | LangGraph (`StateGraph` + `MemorySaver`) |
| **ORM / Driver** | SQLAlchemy 2.0 + psycopg3 |
| **Infra** | Docker Compose |
| **Python** | 3.10+ |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/) (running)
- Python **3.10+**
- A **Google Gemini API key**

---

## Quickstart

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd memCLI
```

### 2. Create and Activate a Virtual Environment
```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Package
```bash
pip install -e .
```

### 4. Configure Environment Variables
```powershell
Copy-Item .env.example .env
```
Edit `.env` with your credentials:
```env
POSTGRES_DB=memcli
POSTGRES_USER=memcli_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_PORT=5433
DATABASE_URL=postgresql://memcli_user:your_secure_password@127.0.0.1:5433/memcli

GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

> **Windows Note**: We use host port `5433` to avoid conflicts with local PostgreSQL. Use `127.0.0.1` instead of `localhost` to avoid IPv6 resolution delays.

### 5. Start the Database
```bash
docker compose up -d
```

### 6. Launch MemCLI
```bash
memcli
```
The database schema, indexes, and HNSW vector index are automatically initialized on first run.

---

## CLI Reference

### Chat

```bash
# Start an interactive multi-turn session (with short-term session buffer)
memcli chat

# Ask a single one-shot question
memcli chat "What programming language do I prefer for interviews?"

# Use a specific user namespace
memcli -u alice chat
```

### Memory Management

```bash
# List stored memories (active only)
memcli list

# List all memories including superseded history
memcli list --all

# Semantic search through memories
memcli search "what languages do I know"

# Keyword search (ILIKE)
memcli search java --keyword

# Manually save a memory
memcli remember "I am learning Rust for systems programming" --category fact
```

### User Control & Privacy

```bash
# Forget a specific memory by UUID
memcli forget 123e4567-e89b-12d3-a456-426614174000

# Semantically find and forget matching memories (with confirmation prompt)
memcli forget "Java for interviews"

# Skip confirmation with --yes
memcli forget "Java for interviews" --yes

# Export all memories as JSON (prints to stdout)
memcli export -u sumit --format json

# Export active-only memories to a CSV file
memcli export -u sumit --format csv --active-only -o backup.csv

# Permanently wipe ALL memories for a user
memcli clear -u sumit --confirm
```

---

## How Memory Works End-to-End

### Every Turn Pipeline (3 LangGraph Nodes)

1. **`retrieve_memories`** — Embeds the user's latest message and runs an HNSW vector search against PostgreSQL to pull the top-5 semantically relevant long-term memories (similarity ≥ 0.55).

2. **`generate_response`** — Injects long-term memories as a `SystemMessage` and the last 10 conversation turns as `HumanMessage`/`AIMessage` context. The LLM sees both tiers simultaneously and responds with full awareness.

3. **`extract_memories`** — A second LLM call analyzes the user's message for *enduring facts* (not chitchat or questions). Each extracted fact is saved to PostgreSQL after passing through the deduplication and conflict resolution pipeline.

### Deduplication Logic

```
New memory candidate
        │
        ▼
Cosine similarity against existing active memories
        │
   sim ≥ 0.95? ──→ YES → Refresh updated_at, return existing ID (no duplicate)
        │
       NO
        │
   0.75 ≤ sim < 0.95? ──→ YES → LLM Arbiter checks for contradiction
        │                           │
        │                      CONFLICT? → Mark old as inactive (superseded_by = new)
        │                      COMPATIBLE? → Both coexist
        │
       NO (sim < 0.75)
        │
        ▼
    Insert as new independent memory
```

---

## Database Schema

```sql
CREATE TABLE memories (
  id               UUID PRIMARY KEY,
  user_id          VARCHAR(128) NOT NULL,
  content          TEXT NOT NULL,
  category         VARCHAR(64) DEFAULT 'fact',       -- 'fact' | 'preference'
  embedding        vector(3072),                     -- Gemini embedding
  is_active        BOOLEAN DEFAULT TRUE,             -- FALSE = superseded
  superseded_by    UUID REFERENCES memories(id),     -- audit trail
  source_thread_id VARCHAR(128),                     -- session origin
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW vector index for sub-millisecond ANN search (3072-dim via halfvec)
CREATE INDEX idx_memories_embedding_hnsw
ON memories
USING hnsw (((embedding)::halfvec(3072)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## Docker Commands Cheat Sheet

| Action | Command |
| :--- | :--- |
| **Start database** | `docker compose up -d` |
| **Stop database** | `docker compose stop` |
| **Tear down** | `docker compose down` |
| **View logs** | `docker compose logs -f postgres` |
| **psql shell** | `docker exec -it memcli-postgres psql -U memcli_user -d memcli` |

---


