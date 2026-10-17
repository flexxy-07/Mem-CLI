from uuid import uuid4
from sqlalchemy import text
from .database import engine

def save_memory(
  user_id: str,
  content: str,
  category: str = 'fact',
  source_thread_id: str | None = None
) -> str:
  """Save a memory and return its ID. Deduplicates if the same memory already exists for the user."""
  clean_content = content.strip()

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
        source_thread_id = COALESCE(:source_thread_id, source_thread_id)
    WHERE id = :id
    RETURNING id
  """)

  insert_query = text("""
    INSERT INTO memories(
      id, user_id, content, category, source_thread_id
    )
    VALUES(
      :id, :user_id, :content, :category, :source_thread_id
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
        "source_thread_id": source_thread_id,
      }
    )
    return str(memory_id)

def list_memories(user_id: str, limit: int = 10) -> list[dict]:
  """Return the most recently updated memories for one user."""

  query = text("""
    SELECT id, content, category,created_at, updated_at
    from memories
    where user_id = :user_id
    order by updated_at desc
    limit :limit

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
  """Search a user's memories by keyword"""
  query = text(
    """
    SELECT id, content, category, created_at, updated_at
    FROM memories
    WHERE user_id = :user_id
    AND content ILIKE :keyword
    ORDER BY updated_at DESC
    LIMIT :limit  
    """
  )

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