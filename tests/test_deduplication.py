"""
Integration test: Semantic Deduplication

Verifies that saving two semantically identical (or exactly matching)
memories results in only ONE database row — the existing row is refreshed,
not duplicated.
"""
import pytest
from memcli.memory import save_memory, list_memories, clear_memories


TEST_USER = "test-deduplication"


@pytest.fixture(autouse=True)
def clean_user():
    """Wipe the test user's memories before and after each test."""
    clear_memories(TEST_USER)
    yield
    clear_memories(TEST_USER)


def test_exact_string_deduplication():
    """Exact same string (case/whitespace variants) must reuse the same row."""
    first_id = save_memory(
        user_id=TEST_USER,
        content="I prefer Java for coding interview problems.",
        category="preference",
    )
    second_id = save_memory(
        user_id=TEST_USER,
        content="  i prefer java for coding interview problems.  ",
        category="preference",
    )
    assert first_id == second_id, (
        f"Expected same memory ID for exact duplicate, got {first_id} vs {second_id}"
    )
    memories = list_memories(TEST_USER)
    assert len(memories) == 1, (
        f"Expected 1 memory after exact duplicate, found {len(memories)}"
    )


def test_semantic_deduplication_paraphrase():
    """Two paraphrases of the same fact with high cosine similarity must not create a duplicate."""
    first_id = save_memory(
        user_id=TEST_USER,
        content="The user prefers Java for coding interview problems.",
        category="preference",
    )
    second_id = save_memory(
        user_id=TEST_USER,
        content="The user likes using Java in coding interviews.",
        category="preference",
    )
    # Paraphrases should be deduplicated (similarity >= 0.95)
    assert first_id == second_id, (
        "Paraphrase was stored as a separate row — semantic deduplication failed."
    )
    memories = list_memories(TEST_USER)
    assert len(memories) == 1, (
        f"Expected 1 memory after paraphrase duplicate, found {len(memories)}"
    )


def test_distinct_memories_stored_separately():
    """Two clearly unrelated memories must each get their own row."""
    id1 = save_memory(
        user_id=TEST_USER,
        content="The user prefers Java for coding interviews.",
        category="preference",
    )
    id2 = save_memory(
        user_id=TEST_USER,
        content="The user enjoys hiking in the mountains on weekends.",
        category="preference",
    )
    assert id1 != id2, "Two distinct memories were incorrectly deduplicated."
    memories = list_memories(TEST_USER)
    assert len(memories) == 2, (
        f"Expected 2 separate memories, found {len(memories)}"
    )
