import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
  raise ValueError("DATABASE_URL missing in .env file")
  
engine = create_engine(database_url, pool_pre_ping=True)

def test_connection():
  with engine.connect() as connection:
    result = connection.execute(
      text("SELECT current_database(), version()")
    )
    database_name, version = result.fetchone()
    print(f"Connected to database: {database_name}")
    print(f"PostgreSQL version: {version}")

if __name__ == "__main__":
  test_connection()