import argparse
import json
import os
from uuid import uuid4

from .agent import chat
from .database import init_db
from .memory import (
    clear_memories,
    delete_memories_semantic,
    delete_memory_by_id,
    export_memories,
    is_valid_uuid,
    list_memories,
    save_memory,
    search_memories,
    search_memories_semantic,
)


def run_interactive(user_id: str):
    """Run a continuous interactive chat session in the terminal with short-term context continuity."""
    session_id = uuid4().hex[:8]
    print("=" * 55)
    print("  MemCLI - AI Assistant with Dual-Tier Memory")
    print(f"  User: {user_id}  |  Session: {session_id}")
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

            response = chat(user_id, message, session_id=session_id)
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
    list_parser.add_argument(
        "--all",
        action="store_true",
        help="Include inactive and superseded memories for audit/history",
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

    # forget subcommand
    forget_parser = subparsers.add_parser(
        "forget",
        help="Delete a memory by UUID or semantic content match",
    )
    forget_parser.add_argument(
        "target",
        help="Memory ID (UUID) or search phrase to semantically match and delete",
    )
    forget_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for semantic forget",
    )
    forget_parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Similarity threshold for semantic matching (default: 0.65)",
    )
    forget_parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        help="Maximum number of matches to delete (default: 5)",
    )

    # export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export user memories for backup or data portability (GDPR compliance)",
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format (json or csv, default: json)",
    )
    export_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional file path to write exported data (prints to stdout if omitted)",
    )
    export_parser.add_argument(
        "--active-only",
        action="store_true",
        help="Export only active memories, excluding superseded ones",
    )

    # clear subcommand
    clear_parser = subparsers.add_parser(
        "clear",
        help="Permanently wipe all memories for the user (GDPR Right to be Forgotten)",
    )
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion non-interactively without prompt",
    )

    args = parser.parse_args()
    user_id = args.user
    init_db()

    if args.command == "list":
        memories = list_memories(user_id=user_id, limit=args.limit, include_superseded=args.all)
        if not memories:
            print(f"\nNo memories found for user '{user_id}'.\n")
            return
        header = f"Stored memories for user '{user_id}' ({len(memories)} total"
        header += ", including superseded):" if args.all else "):"
        print(f"\n{header}")
        print("-" * 65)
        for m in memories:
            created = str(m.get("created_at", ""))[:19]
            status_tag = ""
            if not m.get("is_active", True):
                status_tag = " [SUPERSEDED]"
            print(f"  - [{m.get('category', 'fact')}]{status_tag} {m['content']} ({created})")
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

    elif args.command == "forget":
        target = args.target.strip()
        if is_valid_uuid(target):
            deleted = delete_memory_by_id(user_id=user_id, memory_id=target)
            if deleted:
                print(f"\n[OK] Successfully forgotten memory ID: {target}\n")
            else:
                print(f"\n[ERROR] Memory ID '{target}' not found for user '{user_id}'.\n")
        else:
            matches = search_memories_semantic(
                user_id=user_id,
                query=target,
                limit=args.limit,
                similarity_threshold=args.threshold,
            )
            if not matches:
                print(f"\nNo memories found semantically matching '{target}' (threshold: {args.threshold}).\n")
                return

            print(f"\nFound {len(matches)} matching memory(ies) to forget:")
            print("-" * 65)
            for m in matches:
                sim = m.get("similarity", 0.0)
                print(f"  - [{m['id']}] (sim: {sim:.2f}) [{m.get('category', 'fact')}] {m['content']}")
            print()

            if not args.yes:
                try:
                    confirm = input(f"Permanently delete these {len(matches)} memory(ies)? [y/N]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\nOperation cancelled.\n")
                    return
                if confirm not in ("y", "yes"):
                    print("\nOperation cancelled.\n")
                    return

            deleted_list = delete_memories_semantic(
                user_id=user_id,
                query=target,
                similarity_threshold=args.threshold,
                limit=args.limit,
            )
            print(f"\n[OK] Successfully forgotten {len(deleted_list)} memory(ies).\n")

    elif args.command == "export":
        data = export_memories(
            user_id=user_id,
            include_superseded=not args.active_only,
        )
        if args.format == "json":
            output_content = json.dumps(data, indent=2, ensure_ascii=False)
        elif args.format == "csv":
            import csv
            import io
            if not data:
                output_content = ""
            else:
                fieldnames = [
                    "id",
                    "user_id",
                    "content",
                    "category",
                    "is_active",
                    "superseded_by",
                    "source_thread_id",
                    "created_at",
                    "updated_at",
                ]
                string_io = io.StringIO()
                writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
                output_content = string_io.getvalue()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_content)
            print(f"\n[OK] Exported {len(data)} memory(ies) for user '{user_id}' to: {args.output}\n")
        else:
            print(output_content)

    elif args.command == "clear":
        if not args.confirm:
            try:
                confirm = input(
                    f"WARNING: This will permanently delete ALL memories for user '{user_id}'. "
                    "This action cannot be undone.\nProceed? [y/N]: "
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.\n")
                return

            if confirm not in ("y", "yes"):
                print("\nOperation cancelled.\n")
                return

        deleted_count = clear_memories(user_id=user_id)
        print(f"\n[OK] Successfully wiped all memories for user '{user_id}' ({deleted_count} records deleted).\n")

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
