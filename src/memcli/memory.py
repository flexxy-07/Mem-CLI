from uuid import uuid4
from sqlalchemy import text
from .database import engine
from .embeddings import get_embedding


def save_memory(
  user_id: str,
  content: str,
  category: str = 'fact',
  source_thread_id: str | None = None
) -> str:
  """Save a memory with vector embedding and return its ID. Deduplicates if the same memory already exists for the user."""
  clean_content = content.strip()
  embedding = get_embedding(clean_content)
  embedding_str = str(embedding)

  find_query = text("""
    SELECT id
    FROM memories
    WHERE user_id = :user_id
      AND LOWER(TRIM(content)) = LOWER(:content)
    LIMIT 1
  """)

  update_query = text("""
    UPDATE memories
    SET updated_at = NOW(),
        category = :category,
        embedding = COALESCE(embedding, :embedding),
        source_thread_id = COALESCE(:source_thread_id, source_thread_id)
    WHERE id = :id
    RETURNING id
  """)

  insert_query = text("""
    INSERT INTO memories(
      id, user_id, content, category, embedding, source_thread_id
    )
    VALUES(
      :id, :user_id, :content, :category, :embedding, :source_thread_id
    )
    RETURNING id
  """)

  with engine.begin() as connection:
    existing = connection.execute(
      find_query,
      {"user_id": user_id, "content": clean_content}
    ).fetchone()

    if existing:
      existing_id = existing[0]
      connection.execute(
        update_query,
        {
          "id": existing_id,
          "category": category,
          "embedding": embedding_str,
          "source_thread_id": source_thread_id,
        }
      )
      return str(existing_id)

    memory_id = uuid4()
    connection.execute(
      insert_query,
      {
        "id": memory_id,
        "user_id": user_id,
        "content": clean_content,
        "category": category,
        "embedding": embedding_str,
        "source_thread_id": source_thread_id,
      }
    )
    return str(memory_id)


def list_memories(user_id: str, limit: int = 10) -> list[dict]:
  """Return the most recently updated memories for one user."""
  query = text("""
    SELECT id, content, category, created_at, updated_at
    FROM memories
    WHERE user_id = :user_id
    ORDER BY updated_at DESC
    LIMIT :limit
  """)

  with engine.connect() as connection:
    result = connection.execute(
      query,
      {"user_id": user_id, "limit": limit}
    )
    return [
      dict(row)
      for row in result.mappings().all()
    ]


def search_memories(
  user_id: str,
  keyword: str,
  limit: int = 10,
) -> list[dict]:
  """Search a user's memories by exact keyword (substring ILIKE)."""
  query = text("""
    SELECT id, content, category, created_at, updated_at
    FROM memories
    WHERE user_id = :user_id
      AND content ILIKE :keyword
    ORDER BY updated_at DESC
    LIMIT :limit
  """)

  with engine.connect() as connection:
    result = connection.execute(
      query,
      {
        "user_id": user_id,
        "keyword": f"%{keyword}%",
        "limit": limit
      }
    )
    return [dict(row) for row in result.mappings().all()]


def search_memories_semantic(
  user_id: str,
  query: str,
  limit: int = 5,
  similarity_threshold: float | None = None,
) -> list[dict]:
  """Search memories using vector cosine similarity via pgvector."""
  if not query or not query.strip():
    return []

  query_vector = get_embedding(query.strip())

  sql = text("""
    SELECT id, content, category, created_at, updated_at,
           (embedding <=> :query_vector) AS distance
    FROM memories
    WHERE user_id = :user_id
      AND embedding IS NOT NULL
    ORDER BY distance ASC
    LIMIT :limit
  """)

  with engine.connect() as connection:
    result = connection.execute(
      sql,
      {
        "user_id": user_id,
        "query_vector": str(query_vector),
        "limit": limit,
      }
    )
    memories = []
    for row in result.mappings().all():
      row_dict = dict(row)
      distance = float(row_dict.get("distance", 0.0))
      similarity = max(0.0, 1.0 - distance)
      row_dict["distance"] = distance
      row_dict["similarity"] = round(similarity, 4)

      if similarity_threshold is not None and similarity < similarity_threshold:
        continue

      memories.append(row_dict)

    return memories