"""
Manual sanity check: Database connectivity & quick semantic search

NOT a pytest test — run this to visually confirm the database is running
and semantic search is working for a given user.

Usage:
    python -m tests.manual.check_db_search
"""
from memcli.database import test_connection
from memcli.memory import search_memories_semantic


def main():
    print("=== Database Connection ===")
    test_connection()

    user_id = input("\nEnter a user_id to search (or press Enter for 'sumit'): ").strip() or "sumit"
    query = input("Enter a search query: ").strip()

    if not query:
        print("No query provided. Exiting.")
        return

    print(f"\n=== Semantic Search: '{query}' for user '{user_id}' ===")
    results = search_memories_semantic(user_id=user_id, query=query, limit=5)

    if not results:
        print("No results found.")
        return

    for idx, r in enumerate(results, 1):
        print(f"  {idx}. (sim: {r['similarity']:.4f}) [{r['category']}] {r['content']}")


if __name__ == "__main__":
    main()
