"""
Integration test: Contradiction & Conflict Resolution

Verifies that when a user changes a preference, the old conflicting
memory is superseded (is_active=False, superseded_by=<new_id>), and
the superseded memory no longer appears in semantic search results.
"""
import pytest
from sqlalchemy import text

from memcli.database import engine
from memcli.memory import (
    clear_memories,
    list_memories,
    save_memory,
    search_memories_semantic,
)

TEST_USER = "test-conflicts"


@pytest.fixture(autouse=True)
def clean_user():
    """Wipe the test user's memories before and after each test."""
    clear_memories(TEST_USER)
    yield
    clear_memories(TEST_USER)


def test_compatible_memories_both_stored():
    """Memories from different domains must coexist without conflict."""
    id1 = save_memory(
        user_id=TEST_USER,
        content="The user prefers Java for coding interview problems.",
        category="preference",
    )
    id2 = save_memory(
        user_id=TEST_USER,
        content="The user likes Python for machine learning.",
        category="preference",
    )
    assert id1 != id2, "Compatible memories were incorrectly merged."
    active = list_memories(TEST_USER)
    assert len(active) == 2, (
        f"Expected 2 active memories for compatible facts, got {len(active)}"
    )


def test_conflicting_memory_supersedes_old():
    """A direct preference reversal must deactivate the old memory and link it via superseded_by."""
    java_id = save_memory(
        user_id=TEST_USER,
        content="The user prefers Java for coding interview problems.",
        category="preference",
    )
    # Add an unrelated memory that should survive
    save_memory(
        user_id=TEST_USER,
        content="The user likes Python for machine learning.",
        category="preference",
    )
    python_id = save_memory(
        user_id=TEST_USER,
        content="The user prefers Python for coding interview problems.",
        category="preference",
        resolve_conflicts=True,
    )

    active = list_memories(TEST_USER)
    active_contents = [m["content"] for m in active]

    # Java memory must be gone from active set
    assert not any("Java" in c for c in active_contents), (
        "Superseded Java memory still appears in active memories."
    )
    # Python (replacement) and ML memories should remain
    assert len(active) == 2, (
        f"Expected 2 active memories after conflict resolution, got {len(active)}"
    )

    # Audit the database row directly
    with engine.connect() as conn:
        old_row = conn.execute(
            text("SELECT is_active, superseded_by FROM memories WHERE id = :id"),
            {"id": java_id},
        ).fetchone()

    assert old_row is not None, "Old Java memory row not found in database."
    assert old_row[0] is False, "Old Java memory should be is_active=False."
    assert str(old_row[1]) == str(python_id), (
        "superseded_by should point to the new Python memory."
    )


def test_superseded_memory_absent_from_semantic_search():
    """Superseded memories must not surface in semantic search results."""
    save_memory(
        user_id=TEST_USER,
        content="The user prefers Java for coding interview problems.",
        category="preference",
    )
    save_memory(
        user_id=TEST_USER,
        content="The user prefers Python for coding interview problems.",
        category="preference",
        resolve_conflicts=True,
    )

    results = search_memories_semantic(TEST_USER, "What do I prefer for coding interviews?")
    contents = [r["content"] for r in results]

    assert not any("Java" in c for c in contents), (
        "Superseded Java memory appeared in semantic search results."
    )
    assert any("Python for coding interview" in c for c in contents), (
        "New Python memory is missing from semantic search results."
    )
