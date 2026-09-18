"""
Database Inspector for ChromaDB.
Run: python inspect_db.py
"""
import sys
from pathlib import Path
import chromadb

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import config

def inspect_database():
    db_path = config.CHROMA_PERSIST_DIR
    print("=" * 60)
    print(f" Connecting to ChromaDB at:\n {db_path}")
    print("=" * 60)

    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Error opening ChromaDB: {e}")
        return

    collections = client.list_collections()
    if not collections:
        print("No collections found in database.")
        return

    print(f"\nFound {len(collections)} collection(s):")
    for col in collections:
        print(f" • Collection Name: '{col.name}' (Count: {col.count()})")

    # Inspect the default collection
    try:
        target_col = client.get_collection(config.DEFAULT_COLLECTION_NAME)
    except Exception:
        target_col = collections[0]

    count = target_col.count()
    print(f"\n--- Detailed View for '{target_col.name}' ---")
    print(f"Total Chunks Stored: {count}")

    if count == 0:
        print("The collection is currently empty. Ingest a PDF to populate it.")
        return

    # Retrieve all documents, metadata, and IDs (omit raw high-dim embeddings for clean display)
    data = target_col.get(include=["documents", "metadatas"])

    # Group by source document
    doc_groups = {}
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        source = meta.get("source", "Unknown")
        page = meta.get("page", 1)
        if source not in doc_groups:
            doc_groups[source] = []
        doc_groups[source].append({"id": cid, "page": page, "text": text, "meta": meta})

    print(f"\nIndexed Files ({len(doc_groups)}):")
    for source, chunks in doc_groups.items():
        pages = set(c["page"] for c in chunks)
        print(f" • File: {source}")
        print(f"   - Chunks: {len(chunks)}")
        print(f"   - Page count covered: {len(pages)} (Pages: {sorted(list(pages))})")

    # Show first 3 chunks as sample
    print("\n--- Sample Chunks (First 3) ---")
    for i, (cid, text, meta) in enumerate(zip(data["ids"][:3], data["documents"][:3], data["metadatas"][:3])):
        print(f"\n[Chunk #{i + 1}] ID: {cid}")
        print(f"Source: {meta.get('source')} | Page: {meta.get('page')}")
        snippet = text.strip()[:200].replace("\n", " ")
        print(f"Text Snippet: \"{snippet}...\"")

    print("\n" + "=" * 60)
    print("Tip: You can query, view, or filter specific chunks in Python:")
    print("   import chromadb")
    print("   client = chromadb.PersistentClient(path='./chroma_data')")
    print("   col = client.get_collection('pdf_documents')")
    print("   records = col.get(where={'page': 1})")
    print("=" * 60)


if __name__ == "__main__":
    inspect_database()
