from uuid import uuid4
from sqlalchemy import text
from .database import engine
from .embeddings import get_embedding
from .arbiter import detect_conflict


def save_memory(
  user_id: str,
  content: str,
  category: str = 'fact',
  source_thread_id: str | None = None,
  resolve_conflicts: bool = True,
) -> str:
  """Save a memory with vector embedding, semantic deduplication, and contradiction resolution.

  1. Checks for semantic duplicates (similarity >= 0.95 or exact text match).
  2. If resolve_conflicts=True, detects if the new memory supersedes/contradicts any existing memory.
     Contradicted memories are marked is_active = False with superseded_by pointing to the new memory.
  3. Inserts the new active memory.
  """
  clean_content = content.strip()
  embedding = get_embedding(clean_content)
  embedding_str = str(embedding)
  new_memory_id = uuid4()

  # 1. Semantic Deduplication Check
  find_duplicate_query = text("""
    SELECT id
    FROM memories
    WHERE user_id = :user_id
      AND is_active = TRUE
      AND (
        LOWER(TRIM(content)) = LOWER(:content)
        OR (embedding <=> :embedding) <= 0.05
      )
    ORDER BY (embedding <=> :embedding) ASC
    LIMIT 1
  """)

  update_duplicate_query = text("""
    UPDATE memories
    SET updated_at = NOW(),
        category = :category,
        embedding = COALESCE(embedding, :embedding),
        source_thread_id = COALESCE(:source_thread_id, source_thread_id)
    WHERE id = :id
    RETURNING id
  """)

  find_conflicts_query = text("""
    SELECT id, content, category
    FROM memories
    WHERE user_id = :user_id
      AND is_active = TRUE
      AND id != :new_id
      AND (embedding <=> :embedding) BETWEEN 0.05 AND 0.38
    ORDER BY (embedding <=> :embedding) ASC
    LIMIT 3
  """)

  supersede_query = text("""
    UPDATE memories
    SET is_active = FALSE,
        superseded_by = :new_id,
        updated_at = NOW()
    WHERE id = :old_id
  """)

  insert_query = text("""
    INSERT INTO memories(
      id, user_id, content, category, embedding, is_active, source_thread_id
    )
    VALUES(
      :id, :user_id, :content, :category, :embedding, TRUE, :source_thread_id
    )
    RETURNING id
  """)

  with engine.begin() as connection:
    # 1. Check for near-identical duplicate
    duplicate = connection.execute(
      find_duplicate_query,
      {
        "user_id": user_id,
        "content": clean_content,
        "embedding": embedding_str,
      }
    ).fetchone()

    if duplicate:
      existing_id = duplicate[0]
      connection.execute(
        update_duplicate_query,
        {
          "id": existing_id,
          "category": category,
          "embedding": embedding_str,
          "source_thread_id": source_thread_id,
        }
      )
      return str(existing_id)

    # 2. Insert new memory first (so foreign key constraint is satisfied)
    connection.execute(
      insert_query,
      {
        "id": new_memory_id,
        "user_id": user_id,
        "content": clean_content,
        "category": category,
        "embedding": embedding_str,
        "source_thread_id": source_thread_id,
      }
    )

    # 3. Contradiction & Conflict Resolution: find and supersede outdated memories
    if resolve_conflicts:
      candidates = connection.execute(
        find_conflicts_query,
        {
          "user_id": user_id,
          "embedding": embedding_str,
          "new_id": new_memory_id,
        }
      ).fetchall()

      for cand_id, cand_content, _ in candidates:
        decision = detect_conflict(existing_memory=cand_content, new_memory=clean_content)
        if decision.get("conflict"):
          connection.execute(
            supersede_query,
            {
              "old_id": cand_id,
              "new_id": new_memory_id,
            }
          )

  return str(new_memory_id)


def list_memories(
  user_id: str,
  limit: int = 10,
  include_superseded: bool = False,
) -> list[dict]:
  """Return recently updated memories for a user. By default returns only active memories."""
  active_clause = "" if include_superseded else "AND is_active = TRUE"

  query = text(f"""
    SELECT id, content, category, is_active, superseded_by, created_at, updated_at
    FROM memories
    WHERE user_id = :user_id
      {active_clause}
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
  """Search a user's active memories by exact keyword (substring ILIKE)."""
  query = text("""
    SELECT id, content, category, created_at, updated_at
    FROM memories
    WHERE user_id = :user_id
      AND is_active = TRUE
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
  """Search active memories using vector cosine similarity via pgvector."""
  if not query or not query.strip():
    return []

  query_vector = get_embedding(query.strip())

  sql = text("""
    SELECT id, content, category, created_at, updated_at,
           (embedding <=> :query_vector) AS distance
    FROM memories
    WHERE user_id = :user_id
      AND is_active = TRUE
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