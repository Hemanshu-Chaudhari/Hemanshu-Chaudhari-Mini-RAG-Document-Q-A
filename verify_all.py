import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def test_imports():
    print("Testing core module imports...")
    import config
    from chunking import SemanticStructureChunker
    from vector_store import VectorStoreManager, GeminiEmbeddingFunction
    from memory import ConversationMemory
    from reranker import LLMReranker
    from rag_engine import PDFRAGEngine
    import app
    print("[OK] All modules imported successfully.")

def test_pdf_chunking():
    print("\nTesting semantic PDF chunking...")
    from chunking import SemanticStructureChunker
    pdf_path = Path(__file__).resolve().parent / "sample_rag_report.pdf"
    if not pdf_path.exists():
        import create_sample_pdf
        create_sample_pdf.create_minimal_pdf(str(pdf_path))

    chunker = SemanticStructureChunker(target_chars=600, overlap_chars=100)
    chunks = chunker.process_pdf(str(pdf_path))
    assert len(chunks) > 0, "No chunks generated!"
    
    pages = set(c.metadata["page"] for c in chunks)
    print(f"[OK] Generated {len(chunks)} chunks across {len(pages)} pages (pages: {sorted(list(pages))}).")
    for i, c in enumerate(chunks[:2]):
        print(f"  Chunk {i+1} [Page {c.metadata['page']}]: {c.text[:70]}...")

def test_flask_routes():
    print("\nTesting Flask web application endpoints...")
    from app import app
    client = app.test_client()

    # Test Homepage
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / failed with {resp.status_code}"
    assert b"Gemini RAG" in resp.data, "Brand not found in HTML"
    print("[OK] GET / returned 200 with chat interface HTML.")

    # Test Validation on Missing Chat Message
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 400, f"Expected 400 for empty chat payload, got {resp.status_code}"
    print("[OK] POST /api/chat correctly validated missing message payload.")

    # Test Validation on Upload Without File
    resp = client.post("/api/upload")
    assert resp.status_code == 400, f"Expected 400 for empty upload, got {resp.status_code}"
    print("[OK] POST /api/upload correctly validated missing file upload.")

def test_memory_logic():
    print("\nTesting conversation memory structure...")
    from memory import ConversationMemory
    # Test with dummy client since API key may or may not be provided in offline test
    class DummyClient:
        pass
    mem = ConversationMemory(client=DummyClient())
    mem.add_user_message("What is RAG?")
    mem.add_assistant_message("RAG stands for Retrieval-Augmented Generation.")
    mem.add_user_message("What is its second stage?")
    assert len(mem.get_history()) == 3
    print(f"[OK] Conversation memory stored turns properly:\n  {mem.get_formatted_history().replace(chr(10), ' | ')}")

if __name__ == "__main__":
    try:
        test_imports()
        test_pdf_chunking()
        test_flask_routes()
        test_memory_logic()
        print("\n==========================================")
        print(">>> ALL AUTOMATED VERIFICATION TESTS PASSED!")
        print("==========================================")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

