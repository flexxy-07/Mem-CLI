# memCLI

A CLI-based long-term memory store powered by **PostgreSQL** and **pgvector**.

---

## Features

- **PostgreSQL 17 with pgvector**: Containerized vector database for high-performance vector search and memory embeddings.
- **SQLAlchemy 2.0 & Psycopg 3**: Modern, typed, and efficient database interaction layer.
- **Ready for Packaging**: Clean `src/` layout configured with `pyproject.toml` for standard installation (`pip install -e .`).
- **Containerized Stack**: Simple one-command local database setup via Docker Compose.

---

## Project Structure

```text
memCLI/
├── docker-compose.yml       # Docker Compose setup for PostgreSQL + pgvector
├── pyproject.toml           # Modern package configuration & build system
├── requirements.txt         # Pinned project dependencies
├── .env.example             # Template for environment configuration
├── .gitignore               # Comprehensive gitignore for secrets & caches
├── README.md                # Documentation and setup guide
└── src/
    └── memcli/              # Main application package
        ├── __init__.py      # Package metadata
        ├── agent.py         # LangGraph memory & response agent
        ├── cli.py           # Interactive & command-line interface
        ├── database.py      # Database engine & connection testing
        └── memory.py        # Core memory storage & retrieval operations
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/) (running)
- Python **3.10+**

---

## Quickstart Guide

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd memCLI
```

### 2. Create and Activate a Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the Package and Dependencies
Install in editable mode so changes in `src/memcli` are immediately available:
```bash
pip install -e .
```
*(Or install via `requirements.txt`: `pip install -r requirements.txt`)*

### 4. Configure Environment Variables
Copy the example file to `.env`:
```powershell
# Windows PowerShell
Copy-Item .env.example .env
```
```bash
# macOS / Linux
cp .env.example .env
```

Ensure your `.env` contains valid credentials and port mappings:
```env
POSTGRES_DB=memcli
POSTGRES_USER=memcli_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_PORT=5433
DATABASE_URL=postgresql://memcli_user:your_secure_password@127.0.0.1:5433/memcli
```

> **Note for Windows Users**: We map host port `5433` by default to avoid conflicts with native PostgreSQL installations using port `5432`. We also recommend `127.0.0.1` instead of `localhost` in `DATABASE_URL` to prevent IPv6 DNS resolution delays.

### 5. Start the Database
Launch the PostgreSQL + pgvector container in the background:
```bash
docker compose up -d
```

Check the health status of the container:
```bash
docker compose ps
```

### 6. Verify Database Connection
Run the connection check script:
```bash
python -m memcli.database
```

Expected output:
```text
Connected to database: memcli
PostgreSQL version: PostgreSQL 17.x ...
```

### 7. Run MemCLI

Launch the interactive chat session:
```bash
memcli
# OR
python -m memcli.cli
```

Available subcommands:
```bash
# Start interactive chat (default)
memcli chat

# One-shot question
memcli chat "What programming language do I prefer for interviews?"

# List stored memories
memcli list

# Search memories by keyword
memcli search java

# Manually save a memory
memcli remember "I love building agentic AI applications" --category preference

# Switch user namespace
memcli -u alice chat
```

---

## Docker Commands Cheat Sheet

| Action | Command |
| :--- | :--- |
| **Start database** | `docker compose up -d` |
| **Stop database** | `docker compose stop` |
| **Tear down containers** | `docker compose down` |
| **View logs** | `docker compose logs -f postgres` |
| **Interactive psql shell** | `docker exec -it memcli-postgres psql -U memcli_user -d memcli` |

---

## Security

- Never commit `.env` or sensitive credentials to version control. The `.gitignore` file is configured to exclude all `.env` files automatically.
- Only `.env.example` with sanitized placeholder values should be committed to GitHub.

---

## License

MIT
