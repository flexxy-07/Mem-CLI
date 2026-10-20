"""
Integration test: Dual-Tier Memory Architecture

Verifies that the LangGraph agent correctly maintains:
  1. Short-term session buffer (multi-turn context, pronoun resolution)
  2. Long-term memory influence (retrieved facts shape the response)

This is the core architectural test for the dual-tier memory system.
"""
import uuid
import pytest
from memcli.agent import chat, interact
from memcli.memory import save_memory, list_memories, clear_memories


TEST_USER = "test-dual-memory"


@pytest.fixture(autouse=True)
def clean_user():
    """Wipe the test user's memories before and after each test."""
    clear_memories(TEST_USER)
    yield
    clear_memories(TEST_USER)


def test_long_term_memory_influences_response():
    """
    A preference stored in long-term memory must be respected in the
    agent's response even in a new session.
    """
    # Plant a strong dietary preference in long-term memory
    save_memory(
        user_id=TEST_USER,
        content="The user is strictly vegetarian and does not eat meat or fish.",
        category="preference",
    )
    assert len(list_memories(TEST_USER)) == 1

    session_id = uuid.uuid4().hex[:8]
    response = chat(
        TEST_USER,
        "Plan a 3-course dinner menu for my anniversary.",
        session_id=session_id,
    )
    assert response.strip(), "Agent returned an empty response."

    # The response should not suggest meat items
    meat_words = ["beef", "chicken", "pork", "steak", "bacon", "salmon", "shrimp"]
    found = [w for w in meat_words if w in response.lower()]
    assert not found, (
        f"Long-term vegetarian memory was ignored. Found: {found}\nResponse: {response}"
    )


def test_short_term_session_buffer_accumulates_turns():
    """
    Multiple messages in the same session must accumulate in the
    session buffer (MemorySaver), growing the message count each turn.
    """
    session_id = uuid.uuid4().hex[:8]

    state1 = interact(TEST_USER, "What is a binary search tree?", session_id=session_id)
    count1 = len(state1.get("messages", []))

    state2 = interact(TEST_USER, "Can you give me a Python implementation of it?", session_id=session_id)
    count2 = len(state2.get("messages", []))

    assert count2 > count1, (
        f"Session buffer did not grow across turns: turn1={count1}, turn2={count2}. "
        "MemorySaver checkpointing may not be working."
    )


def test_different_sessions_are_isolated():
    """
    Two different session IDs for the same user must have independent
    short-term buffers — messages from session A must not bleed into session B.
    """
    session_a = uuid.uuid4().hex[:8]
    session_b = uuid.uuid4().hex[:8]

    interact(TEST_USER, "I am working on a FastAPI project.", session_id=session_a)
    state_b = interact(TEST_USER, "What have we been discussing?", session_id=session_b)

    # Session B should have only 2 messages (1 human + 1 AI), not 4
    count_b = len(state_b.get("messages", []))
    assert count_b <= 2, (
        f"Session B has {count_b} messages — it may have inherited Session A's context."
    )
