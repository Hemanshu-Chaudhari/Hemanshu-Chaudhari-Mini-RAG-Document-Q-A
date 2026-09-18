import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Models
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemini-3.5-flash-lite")

# Storage
CHROMA_PERSIST_DIR = str(BASE_DIR / os.environ.get("CHROMA_PERSIST_DIR", "chroma_data"))
UPLOAD_DIR = str(BASE_DIR / "uploads")
DEFAULT_COLLECTION_NAME = os.environ.get("DEFAULT_COLLECTION_NAME", "pdf_documents")

# Advanced RAG Parameters
CHUNK_TARGET_CHARS = int(os.environ.get("CHUNK_TARGET_CHARS", 800))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", 150))
FIRST_STAGE_RETRIEVAL_K = int(os.environ.get("FIRST_STAGE_RETRIEVAL_K", 8))
FINAL_RERANK_K = int(os.environ.get("FINAL_RERANK_K", 3))

# Ensure directories exist
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
