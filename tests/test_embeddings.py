"""
Integration test: Embeddings Module

Verifies that the Gemini embedding service returns valid 3072-dimensional
vectors and that cosine similarity between semantically related texts
is higher than between unrelated texts.
"""
import math
import pytest
from memcli.embeddings import get_embedding, get_model_name


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def test_embedding_returns_correct_dimension():
    """get_embedding() must return a 3072-dimensional vector."""
    vector = get_embedding("The user prefers Java for coding interviews.")
    assert isinstance(vector, list), "Embedding must be a list."
    assert len(vector) == 3072, (
        f"Expected 3072 dimensions, got {len(vector)}."
    )


def test_embedding_values_are_floats():
    """All values in the embedding vector must be floats."""
    vector = get_embedding("Hello world.")
    assert all(isinstance(v, float) for v in vector), (
        "Embedding vector contains non-float values."
    )


def test_empty_string_raises_error():
    """get_embedding() must raise a ValueError for empty input."""
    with pytest.raises(ValueError):
        get_embedding("   ")


def test_semantic_similarity_ordering():
    """
    'Java for interviews' should have higher cosine similarity to
    'preferred coding language' than an unrelated text like 'cloudy weather'.
    """
    query = "What is my preferred programming language?"
    related = "The user prefers Java for coding interviews."
    unrelated = "The weather today is cloudy with rain."

    q_vec = get_embedding(query)
    r_vec = get_embedding(related)
    u_vec = get_embedding(unrelated)

    sim_related = cosine_similarity(q_vec, r_vec)
    sim_unrelated = cosine_similarity(q_vec, u_vec)

    assert sim_related > sim_unrelated, (
        f"Expected related text (sim={sim_related:.4f}) to score higher than "
        f"unrelated text (sim={sim_unrelated:.4f})."
    )


def test_model_name_is_configured():
    """get_model_name() must return a non-empty string."""
    name = get_model_name()
    assert isinstance(name, str) and name.strip(), (
        "Embedding model name is not configured."
    )
