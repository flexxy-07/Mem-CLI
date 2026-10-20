"""
Integration test: Memory Extraction Pipeline

Verifies that the LangGraph agent correctly extracts enduring facts and
preferences from user messages, and does NOT extract ephemeral content
like questions, chitchat, or one-off requests.
"""
import pytest
from memcli.agent import interact
from memcli.memory import list_memories, clear_memories


TEST_USER = "test-extraction"


@pytest.fixture(autouse=True)
def clean_user():
    """Wipe the test user's memories before and after each test."""
    clear_memories(TEST_USER)
    yield
    clear_memories(TEST_USER)


def test_extracts_fact_and_preference_from_message():
    """A message containing both a fact and a preference must produce at least one extracted memory."""
    msg = "I'm learning LangGraph, and I prefer Java for coding interviews."
    result = interact(TEST_USER, msg)
    extracted = result.get("extracted_memories", [])
    assert len(extracted) >= 1, (
        f"Expected at least 1 extracted memory from '{msg}', got {len(extracted)}"
    )
    # Should be saved to DB
    db_memories = list_memories(TEST_USER)
    assert len(db_memories) >= 1, "Extracted memories were not persisted to the database."


def test_deduplication_on_repeated_message():
    """Sending the same message twice must not grow the memory count."""
    msg = "I'm learning LangGraph, and I prefer Java for coding interviews."
    interact(TEST_USER, msg)
    count_after_first = len(list_memories(TEST_USER))

    interact(TEST_USER, msg)
    count_after_second = len(list_memories(TEST_USER))

    assert count_after_first == count_after_second, (
        f"Memory count grew on duplicate message: {count_after_first} -> {count_after_second}. "
        "Deduplication failed."
    )


def test_no_memory_extracted_from_question():
    """A simple question must not produce any extracted memories."""
    result = interact(TEST_USER, "What's the weather like today?")
    extracted = result.get("extracted_memories", [])
    assert len(extracted) == 0, (
        f"Expected 0 extracted memories from a weather question, got {len(extracted)}: {extracted}"
    )


def test_no_memory_extracted_from_chitchat():
    """Chitchat and technical requests must not produce extracted memories."""
    result = interact(TEST_USER, "Can you explain how quicksort works?")
    extracted = result.get("extracted_memories", [])
    assert len(extracted) == 0, (
        f"Expected 0 extracted memories from a chitchat message, got {len(extracted)}: {extracted}"
    )


def test_response_is_non_empty():
    """The agent must always return a non-empty response string."""
    result = interact(TEST_USER, "Hello!")
    assert result.get("response", "").strip(), "Agent returned an empty response."
