"""
Integration test: Semantic Vector Search

Verifies that search_memories_semantic() returns relevant results ordered
by cosine similarity, and that results respect the similarity threshold.
"""
import pytest
from memcli.memory import save_memory, search_memories_semantic, clear_memories


TEST_USER = "test-semantic-search"


@pytest.fixture(autouse=True)
def clean_user():
    """Wipe the test user's memories before and after each test."""
    clear_memories(TEST_USER)
    yield
    clear_memories(TEST_USER)


@pytest.fixture()
def seeded_memories():
    """Seed a set of memories with distinct topics."""
    save_memory(TEST_USER, "The user prefers Java for coding interviews.", "preference")
    save_memory(TEST_USER, "The user likes Python for machine learning.", "preference")
    save_memory(TEST_USER, "The user lives in San Francisco.", "fact")
    save_memory(TEST_USER, "The user is learning LangGraph.", "fact")


def test_semantic_search_returns_relevant_result(seeded_memories):
    """Querying about coding language should return the coding interview memory on top."""
    results = search_memories_semantic(
        user_id=TEST_USER,
        query="What programming language do I use for interviews?",
        limit=3,
    )
    assert len(results) > 0, "Expected at least one semantic search result."
    top_content = results[0]["content"]
    assert "Java" in top_content or "coding interview" in top_content.lower(), (
        f"Expected coding interview memory at top, got: '{top_content}'"
    )


def test_similarity_scores_are_descending(seeded_memories):
    """Results must be ordered from highest to lowest similarity."""
    results = search_memories_semantic(
        user_id=TEST_USER,
        query="programming language for tech interviews",
        limit=4,
    )
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True), (
        "Results are not sorted by descending similarity."
    )


def test_similarity_threshold_filters_low_matches(seeded_memories):
    """A high threshold must exclude weakly related memories."""
    results_high = search_memories_semantic(
        user_id=TEST_USER,
        query="programming language for tech interviews",
        limit=5,
        similarity_threshold=0.90,
    )
    results_low = search_memories_semantic(
        user_id=TEST_USER,
        query="programming language for tech interviews",
        limit=5,
        similarity_threshold=0.30,
    )
    assert len(results_high) <= len(results_low), (
        "High threshold should return fewer or equal results than low threshold."
    )


def test_empty_query_returns_no_results(seeded_memories):
    """An empty or whitespace-only query must return an empty list without error."""
    results = search_memories_semantic(user_id=TEST_USER, query="   ")
    assert results == [], f"Expected empty list for blank query, got: {results}"


def test_similarity_field_in_results(seeded_memories):
    """Each result must include both 'similarity' and 'distance' fields."""
    results = search_memories_semantic(
        user_id=TEST_USER,
        query="What do I like?",
        limit=2,
    )
    for r in results:
        assert "similarity" in r, "Result missing 'similarity' field."
        assert "distance" in r, "Result missing 'distance' field."
        assert 0.0 <= r["similarity"] <= 1.0, (
            f"Similarity {r['similarity']} out of expected [0, 1] range."
        )
