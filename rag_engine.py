from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from google import genai

import config
from chunking import SemanticStructureChunker
from vector_store import VectorStoreManager
from memory import ConversationMemory
from reranker import LLMReranker


@dataclass
class RAGResponse:
    answer: str
    standalone_query: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]


class PDFRAGEngine:
    """
    Main orchestrator for the Advanced PDF RAG System.
    Connects ingestion, vector search, conversational memory, reranking, and generation.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable "
                "or pass it to PDFRAGEngine."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.chunker = SemanticStructureChunker(
            target_chars=config.CHUNK_TARGET_CHARS,
            overlap_chars=config.CHUNK_OVERLAP_CHARS
        )
        self.vector_store = VectorStoreManager(client=self.client)
        self.memory = ConversationMemory(client=self.client)
        self.reranker = LLMReranker(client=self.client)

    def ingest_pdf(self, pdf_path: str, collection_name: str = config.DEFAULT_COLLECTION_NAME) -> Dict[str, Any]:
        """
        Parses PDF, applies semantic chunking, and stores vectors in ChromaDB.
        """
        chunks = self.chunker.process_pdf(pdf_path)
        count = self.vector_store.add_chunks(chunks, collection_name=collection_name)
        return {
            "file": pdf_path,
            "chunks_created": len(chunks),
            "chunks_stored": count,
            "collection": collection_name
        }

    def ask(
        self,
        question: str,
        collection_name: str = config.DEFAULT_COLLECTION_NAME,
        use_reranking: bool = True,
        use_memory: bool = True
    ) -> RAGResponse:
        """
        Executes the full advanced retrieval and generation workflow.
        """
        # Step 1: Query Reformulation (Conversational Memory)
        if use_memory:
            standalone_query = self.memory.reformulate_query(question)
        else:
            standalone_query = question.strip()

        # Step 2: First-Stage Vector Retrieval from ChromaDB
        candidates = self.vector_store.query(
            query_text=standalone_query,
            top_k=config.FIRST_STAGE_RETRIEVAL_K,
            collection_name=collection_name
        )

        if not candidates:
            no_docs_msg = "No documents found in the database. Please upload or ingest a PDF first."
            return RAGResponse(
                answer=no_docs_msg,
                standalone_query=standalone_query,
                sources=[],
                retrieved_chunks=[]
            )

        # Step 3: Second-Stage Reranking
        if use_reranking and len(candidates) > config.FINAL_RERANK_K:
            relevant_chunks = self.reranker.rerank(
                query=standalone_query,
                candidate_chunks=candidates,
                top_k=config.FINAL_RERANK_K
            )
        else:
            relevant_chunks = candidates[:config.FINAL_RERANK_K]

        # Step 4: Build Context and Page Citations
        context_blocks = []
        sources = []
        seen_pages = set()

        for chunk in relevant_chunks:
            meta = chunk["metadata"]
            source_file = meta.get("source", "Document")
            page = meta.get("page", 1)
            
            context_blocks.append(f"--- Excerpt from {source_file} (Page {page}) ---\n{chunk['text']}")
            
            page_key = (source_file, page)
            if page_key not in seen_pages:
                sources.append({
                    "source": source_file,
                    "page": page,
                    "similarity": chunk.get("similarity", 0.0),
                    "rerank_score": chunk.get("rerank_score", None)
                })
                seen_pages.add(page_key)

        context_str = "\n\n".join(context_blocks)
        history_str = self.memory.get_formatted_history() if use_memory else "None"

        # Step 5: Grounded Prompting for Gemini
        system_instruction = f"""You are an expert AI research assistant answering questions using document context.

DOCUMENT CONTEXT:
==================================================
{context_str}
==================================================

CONVERSATION HISTORY:
{history_str}

LATEST QUESTION:
{question}

CRITICAL RULES:
1. Answer the question thoroughly, accurately, and directly using ONLY information present in the DOCUMENT CONTEXT above.
2. If the context does not contain enough information to answer the question, clearly state:
   "I cannot find this information in the uploaded document." Do NOT fabricate facts or extrapolate without basis.
3. Every factual claim MUST include a page citation in brackets, e.g., "[Page 3]" or "[Page 2, 5]".
4. Maintain a professional, clear, and structured format (using bullet points where helpful).
"""

        response = self.client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=system_instruction
        )
        answer_text = response.text.strip()

        # Step 6: Update Conversational Memory
        if use_memory:
            self.memory.add_user_message(question)
            self.memory.add_assistant_message(answer_text)

        return RAGResponse(
            answer=answer_text,
            standalone_query=standalone_query,
            sources=sources,
            retrieved_chunks=relevant_chunks
        )

    def reset_conversation(self):
        self.memory.clear()
