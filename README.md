# Advanced PDF RAG with Gemini & ChromaDB

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** pipeline powered by **Google Gemini** (`gemini-2.5-flash` & `text-embedding-004`) and **ChromaDB**.

---

## 🌟 Key Features

1. **Semantic & Document-Structure Chunking**:
   - Preserves natural paragraph and sentence boundaries.
   - Retains page numbers, document names, and token/character metrics for accurate citations.
   - Configurable target size and overlap window.

2. **ChromaDB Vector Store with Gemini Embeddings**:
   - Uses Google's `text-embedding-004`.
   - Distinguishes between `task_type="RETRIEVAL_DOCUMENT"` (for indexing) and `task_type="RETRIEVAL_QUERY"` (for searching) for optimal vector projection.
   - Persistent on-disk vector storage with cosine distance.

3. **Conversational Memory & Query Reformulation**:
   - Multi-turn conversational history.
   - Automatically detects follow-up questions containing pronouns or ambiguous references (e.g., *"Can you elaborate on its second point?"*) and reformulates them into complete standalone search queries.

4. **Two-Stage Retrieval with LLM Reranker**:
   - First stage: ChromaDB retrieves the top-N candidate passages.
   - Second stage: Gemini reranker evaluates candidate relevance (0–10 score) to eliminate noise and elevate the highest-signal context chunks.

5. **Grounded Responses with Page Citations**:
   - System prompts enforce strict grounding in the provided context.
   - Every factual assertion cites the source page (e.g. `[Page 4]`).

6. **Dual User Interfaces**:
   - **Modern Web UI (Flask)**: Drag-and-drop PDF upload, progress tracking, live chat, query reformulation badges, and collapsible citations.
   - **Terminal CLI**: Fast scriptable CLI for batch ingestion and terminal chat sessions.

---

## 🚀 Quick Start

### 1. Configure Gemini API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Add your Gemini API Key in `.env`:
```env
GEMINI_API_KEY=AIzaSy...
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

- Drag and drop any PDF into the sidebar.
- Start asking questions in the chat!

---

## 💻 Command-Line Interface (CLI) Usage

### Ingest a PDF document:
```bash
python cli.py ingest path/to/document.pdf
```

### Start an interactive chat session:
```bash
python cli.py chat
```

### Run a single query:
```bash
python cli.py query "What are the primary conclusions of the report?"
```

### Inspect collection statistics:
```bash
python cli.py stats
```

### Clear the collection:
```bash
python cli.py clear
```

---

## ⚙️ Configuration (`config.py`)

You can customize behavior via `.env` or in `config.py`:
- `CHUNK_TARGET_CHARS`: Target character size per chunk (default: `800`)
- `CHUNK_OVERLAP_CHARS`: Overlap between adjacent chunks (default: `150`)
- `FIRST_STAGE_RETRIEVAL_K`: Number of candidates retrieved from vector search (default: `8`)
- `FINAL_RERANK_K`: Number of top reranked chunks passed to Gemini (default: `3`)
- `EMBEDDING_MODEL`: Gemini embedding model (default: `text-embedding-004`)
- `GENERATION_MODEL`: Gemini generative model (default: `gemini-2.5-flash`)
