import os
import sys
import argparse
from rag_engine import PDFRAGEngine
import config


def print_banner():
    print("=" * 60)
    print("      PDF RAG Assistant with Gemini & ChromaDB")
    print("  (Semantic Chunking | Re-Ranking | Memory | Citations)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Advanced PDF RAG with Gemini & ChromaDB")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF file into ChromaDB")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    ingest_parser.add_argument("--collection", type=str, default=config.DEFAULT_COLLECTION_NAME, help="Collection name")

    # Ask single query command
    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--collection", type=str, default=config.DEFAULT_COLLECTION_NAME, help="Collection name")

    # Interactive chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session with conversation memory")
    chat_parser.add_argument("--collection", type=str, default=config.DEFAULT_COLLECTION_NAME, help="Collection name")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show vector database collection statistics")
    stats_parser.add_argument("--collection", type=str, default=config.DEFAULT_COLLECTION_NAME, help="Collection name")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the vector database collection")
    clear_parser.add_argument("--collection", type=str, default=config.DEFAULT_COLLECTION_NAME, help="Collection name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        engine = PDFRAGEngine()
    except Exception as e:
        print(f"\n[Error Initializing Engine]: {e}")
        print("Please verify GEMINI_API_KEY in your .env file or environment.")
        sys.exit(1)

    if args.command == "ingest":
        print(f"\n[Ingesting]: {args.pdf_path}")
        result = engine.ingest_pdf(args.pdf_path, collection_name=args.collection)
        print(f"[Done] Created {result['chunks_created']} semantic chunks and saved to ChromaDB collection '{result['collection']}'.")

    elif args.command == "stats":
        stats = engine.vector_store.get_collection_stats(args.collection)
        print(f"\n[Collection]: {stats['name']}")
        print(f"[Total Chunks]: {stats['total_chunks']}")
        print(f"[Indexed Documents]: {', '.join(stats['documents']) if stats['documents'] else 'None'}")

    elif args.command == "clear":
        engine.vector_store.clear_collection(args.collection)
        print(f"\n[Done] Collection '{args.collection}' cleared.")

    elif args.command == "query":
        print(f"\n[Question]: {args.question}")
        response = engine.ask(args.question, collection_name=args.collection, use_memory=False)
        print("\n=== Answer ===")
        print(response.answer)
        if response.sources:
            print("\n=== Citations ===")
            for s in response.sources:
                score_info = f" (Rerank Score: {s['rerank_score']:.1f}/10)" if s['rerank_score'] is not None else ""
                print(f"- {s['source']} [Page {s['page']}]{score_info}")

    elif args.command == "chat":
        print_banner()
        stats = engine.vector_store.get_collection_stats(args.collection)
        print(f"Active Collection: {stats['name']} ({stats['total_chunks']} chunks stored)")
        print("Type 'exit' or 'quit' to end session. Type 'reset' to clear conversation memory.\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
                if user_input.lower() == "reset":
                    engine.reset_conversation()
                    print("[Memory reset: started fresh conversation]")
                    continue

                response = engine.ask(user_input, collection_name=args.collection, use_memory=True)
                
                if response.standalone_query != user_input:
                    print(f"🔍 [Query Reformulated]: \"{response.standalone_query}\"")

                print(f"\nGemini:\n{response.answer}")

                if response.sources:
                    print("\nSources:")
                    for s in response.sources:
                        score_info = f" (Score: {s['rerank_score']:.1f}/10)" if s['rerank_score'] is not None else ""
                        print(f"  • {s['source']} - Page {s['page']}{score_info}")

            except KeyboardInterrupt:
                print("\nSession ended.")
                break


if __name__ == "__main__":
    main()
