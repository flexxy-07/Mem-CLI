import argparse
import os
import sys

from .agent import chat
from .memory import (
    list_memories,
    save_memory,
    search_memories,
    search_memories_semantic,
)


def run_interactive(user_id: str):
    """Run a continuous interactive chat session in the terminal."""
    print("=" * 55)
    print("  MemCLI - AI Assistant with Long-Term Memory")
    print(f"  User: {user_id}")
    print("  Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 55 + "\n")

    while True:
        try:
            message = input(f"[{user_id}] You: ").strip()
            if not message:
                continue

            if message.lower() in ("exit", "quit", "q"):
                print("\nGoodbye! See you next time.")
                break

            response = chat(user_id, message)
            print(f"\nMemCLI: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        prog="memcli",
        description="MemCLI: Long-term memory AI assistant powered by PostgreSQL and pgvector.",
    )
    parser.add_argument(
        "-u",
        "--user",
        default=os.getenv("MEMCLI_USER", "sumit"),
        help="User ID for memory namespace (default: sumit or $MEMCLI_USER)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Optional single message to ask immediately",
    )

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List stored memories for the user")
    list_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="Maximum number of memories to display (default: 10)",
    )

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Search memories semantically or by keyword")
    search_parser.add_argument("query", help="Query or keyword to search for")
    search_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        help="Maximum search results (default: 5)",
    )
    search_parser.add_argument(
        "--keyword",
        action="store_true",
        help="Use literal keyword search (ILIKE) instead of semantic vector search",
    )

    # remember subcommand
    remember_parser = subparsers.add_parser("remember", help="Manually save a memory")
    remember_parser.add_argument("content", help="Memory content to save")
    remember_parser.add_argument(
        "-c",
        "--category",
        default="fact",
        help="Category for the memory (default: fact)",
    )

    args = parser.parse_args()
    user_id = args.user

    if args.command == "list":
        memories = list_memories(user_id=user_id, limit=args.limit)
        if not memories:
            print(f"\nNo memories found for user '{user_id}'.\n")
            return
        print(f"\nStored memories for user '{user_id}' ({len(memories)} total):")
        print("-" * 55)
        for m in memories:
            created = str(m.get("created_at", ""))[:19]
            print(f"  - [{m.get('category', 'fact')}] {m['content']} ({created})")
        print()

    elif args.command == "search":
        if args.keyword:
            results = search_memories(user_id=user_id, keyword=args.query, limit=args.limit)
            if not results:
                print(f"\nNo memories found matching keyword '{args.query}'.\n")
                return
            print(f"\nKeyword search results for '{args.query}':")
            print("-" * 55)
            for m in results:
                print(f"  - [{m.get('category', 'fact')}] {m['content']}")
            print()
        else:
            results = search_memories_semantic(user_id=user_id, query=args.query, limit=args.limit)
            if not results:
                print(f"\nNo semantic matches found for '{args.query}'.\n")
                return
            print(f"\nSemantic search results for '{args.query}':")
            print("-" * 55)
            for m in results:
                sim = m.get("similarity", 0.0)
                print(f"  - (similarity: {sim:.2f}) [{m.get('category', 'fact')}] {m['content']}")
            print()

    elif args.command == "remember":
        memory_id = save_memory(user_id=user_id, content=args.content, category=args.category)
        print(f"\n[OK] Memory saved successfully! (ID: {memory_id})\n")

    elif args.command == "chat":
        if args.message:
            response = chat(user_id, args.message)
            print(f"\nMemCLI: {response}\n")
        else:
            run_interactive(user_id)

    else:
        # Default action when no subcommand is specified: start interactive chat
        run_interactive(user_id)


if __name__ == "__main__":
    main()
