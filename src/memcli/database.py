import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
  raise ValueError("DATABASE_URL missing in .env file")
  
engine = create_engine(database_url, pool_pre_ping=True)

def init_db():
  """Ensure pgvector extension, table schemas, and HNSW vector index are set up."""
  with engine.begin() as connection:
    # 1. Enable pgvector extension
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # 2. Ensure memories table exists
    connection.execute(text("""
      CREATE TABLE IF NOT EXISTS memories (
        id UUID PRIMARY KEY,
        user_id VARCHAR(128) NOT NULL,
        content TEXT NOT NULL,
        category VARCHAR(64) DEFAULT 'fact',
        embedding vector(3072),
        is_active BOOLEAN DEFAULT TRUE,
        superseded_by UUID REFERENCES memories(id),
        source_thread_id VARCHAR(128),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      );
    """))

    # 3. Ensure embedding column is sized for 3072 dimensions
    connection.execute(text("""
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1 FROM information_schema.columns 
          WHERE table_name = 'memories' AND column_name = 'embedding'
        ) THEN
          ALTER TABLE memories ALTER COLUMN embedding TYPE vector(3072);
        END IF;
      END $$;
    """))

    # 4. Standard B-tree indexes for fast user queries
    connection.execute(text("""
      CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
    """))
    connection.execute(text("""
      CREATE INDEX IF NOT EXISTS idx_memories_user_active ON memories(user_id, is_active);
    """))

    # 5. HNSW vector index on 3072 dimensions using halfvec for sub-millisecond ANN search
    connection.execute(text("""
      CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
      ON memories 
      USING hnsw (((embedding)::halfvec(3072)) halfvec_cosine_ops)
      WITH (m = 16, ef_construction = 64);
    """))


def test_connection():
  init_db()
  with engine.connect() as connection:
    result = connection.execute(
      text("SELECT current_database(), version()")
    )
    database_name, version = result.fetchone()
    print(f"Connected to database: {database_name}")
    print(f"PostgreSQL version: {version}")
    print("Database schema and HNSW vector index verified.")


if __name__ == "__main__":
  test_connection()