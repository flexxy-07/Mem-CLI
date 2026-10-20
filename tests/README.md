# MemCLI Test Suite

This directory contains all tests for the MemCLI memory system. Tests are
organized into two categories:

```
tests/
├── test_deduplication.py    # Semantic & exact-string deduplication
├── test_conflicts.py        # Contradiction detection & conflict resolution
├── test_privacy.py          # GDPR controls: forget, export, clear
├── test_extraction.py       # Memory extraction from chat messages
├── test_semantic_search.py  # Vector search correctness & ranking
├── test_embeddings.py       # Embedding model dimensions & similarity
├── test_dual_memory.py      # Dual-tier memory (short-term + long-term)
└── manual/
    ├── check_llm_connection.py   # Verify Gemini API key & connectivity
    └── check_db_search.py        # Interactive DB + semantic search check
```

---

## Prerequisites

Make sure the database is running before executing any tests:

```bash
docker compose up -d
```

And your `.env` is configured with valid credentials.

---

## Running the Full Test Suite

```bash
# From the project root
pytest tests/ -v
```

### Run a specific test file

```bash
pytest tests/test_conflicts.py -v
pytest tests/test_privacy.py -v
pytest tests/test_dual_memory.py -v
```

### Run a specific test function

```bash
pytest tests/test_conflicts.py::test_conflicting_memory_supersedes_old -v
```

---

## Test Categories

| File | What it tests | Requires DB | Requires LLM |
| :--- | :--- | :---: | :---: |
| `test_embeddings.py` | Embedding dimensions & similarity ordering | ✗ | ✅ |
| `test_deduplication.py` | Exact & semantic deduplication | ✅ | ✅ |
| `test_conflicts.py` | Conflict detection & supersession | ✅ | ✅ |
| `test_semantic_search.py` | Vector search ranking & thresholds | ✅ | ✅ |
| `test_privacy.py` | forget / export / clear operations | ✅ | ✅ |
| `test_extraction.py` | Memory extraction from chat turns | ✅ | ✅ |
| `test_dual_memory.py` | Short-term buffer + long-term retrieval | ✅ | ✅ |

---

## Manual Checks

These are **not pytest tests** — they are interactive scripts for quick
environment verification. Run them before the first test run:

```bash
# Check Gemini API is working
python -m tests.manual.check_llm_connection

# Interactively check DB connectivity and semantic search
python -m tests.manual.check_db_search
```

---

> **Note**: All integration tests use isolated `test-*` user IDs and clean up
> their data both before and after each test via pytest `autouse` fixtures.
> They will not interfere with production data.
