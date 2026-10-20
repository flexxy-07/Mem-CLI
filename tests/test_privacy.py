"""
Integration test: User Control & Data Governance (GDPR)

Verifies all data management primitives: forget by ID, semantic forget,
export, and full clear (right to be forgotten).
"""
import pytest
from memcli.memory import (
    save_memory,
    list_memories,
    delete_memory_by_id,
    delete_memories_semantic,
    clear_memories,
    export_memories,
)


TEST_USER = "test-privacy"


@pytest.fixture(autouse=True)
def clean_user():
    """Wipe the test user's memories before and after each test."""
    clear_memories(TEST_USER)
    yield
    clear_memories(TEST_USER)


@pytest.fixture()
def three_memories():
    """Seed three distinct memories and return their IDs."""
    id1 = save_memory(
        user_id=TEST_USER,
        content="The user loves building agentic workflows with LangGraph.",
        category="preference",
    )
    id2 = save_memory(
        user_id=TEST_USER,
        content="The user prefers TypeScript for building frontend applications.",
        category="preference",
    )
    id3 = save_memory(
        user_id=TEST_USER,
        content="The user lives in San Francisco, California.",
        category="fact",
    )
    return id1, id2, id3


def test_export_returns_all_memories(three_memories):
    """export_memories() must return all stored records with expected fields."""
    exported = export_memories(TEST_USER)
    assert len(exported) == 3, f"Expected 3 exported memories, got {len(exported)}"
    for record in exported:
        assert "id" in record
        assert "content" in record
        assert "category" in record
        assert "created_at" in record


def test_forget_by_id(three_memories):
    """delete_memory_by_id() must remove exactly the targeted memory."""
    id1, id2, id3 = three_memories
    deleted = delete_memory_by_id(TEST_USER, id1)
    assert deleted is True, "Expected True for a successful deletion."
    remaining = list_memories(TEST_USER)
    assert len(remaining) == 2, f"Expected 2 memories after ID deletion, got {len(remaining)}"
    remaining_ids = [str(m["id"]) for m in remaining]
    assert str(id1) not in remaining_ids, "Deleted memory still appears in list."


def test_forget_nonexistent_id_returns_false():
    """delete_memory_by_id() must return False for an ID that does not exist."""
    import uuid
    fake_id = str(uuid.uuid4())
    result = delete_memory_by_id(TEST_USER, fake_id)
    assert result is False, "Expected False when deleting a non-existent memory."


def test_semantic_forget(three_memories):
    """delete_memories_semantic() must delete semantically matching memories."""
    removed = delete_memories_semantic(
        user_id=TEST_USER,
        query="What frontend language do I use?",
        similarity_threshold=0.60,
    )
    assert len(removed) >= 1, "Expected at least 1 memory removed by semantic forget."
    assert any("TypeScript" in r["content"] for r in removed), (
        "Expected TypeScript memory to be removed by semantic forget."
    )


def test_clear_all_memories(three_memories):
    """clear_memories() must wipe every memory for the user."""
    count = clear_memories(TEST_USER)
    assert count >= 3, f"Expected at least 3 memories cleared, got {count}."
    remaining = list_memories(TEST_USER)
    assert len(remaining) == 0, (
        f"Expected 0 memories after clear, got {len(remaining)}."
    )
