import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import types

import config
from chunking import DocumentChunk


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible EmbeddingFunction backed by Google Gemini text-embedding-004.
    Batches document embeddings with task_type='RETRIEVAL_DOCUMENT'.
    """

    def __init__(self, client: genai.Client, model: str = config.EMBEDDING_MODEL, batch_size: int = 50):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings: Embeddings = []
        doc_list = list(input)

        for i in range(0, len(doc_list), self.batch_size):
            batch = doc_list[i:i + self.batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                for emb in response.embeddings:
                    all_embeddings.append(emb.values)
            except Exception as e:
                # If single-call batch fails, fallback to item-by-item
                for item in batch:
                    resp = self.client.models.embed_content(
                        model=self.model,
                        contents=item,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT"
                        )
                    )
                    all_embeddings.append(resp.embeddings[0].values)

        return all_embeddings


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector storage and semantic retrieval with Gemini.
    """

    def __init__(self, client: genai.Client, persist_dir: str = config.CHROMA_PERSIST_DIR):
        self.client = client
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = GeminiEmbeddingFunction(client=self.client)

    def get_or_create_collection(self, collection_name: str = config.DEFAULT_COLLECTION_NAME):
        return self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk], collection_name: str = config.DEFAULT_COLLECTION_NAME) -> int:
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)
        
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Ingest in batches of 100 to stay well within limits
        batch_size = 100
        total_added = 0
        for i in range(0, len(ids), batch_size):
            b_ids = ids[i:i + batch_size]
            b_texts = texts[i:i + batch_size]
            b_meta = metadatas[i:i + batch_size]

            # Upsert will insert new chunks or update existing chunks if re-ingested
            collection.upsert(
                ids=b_ids,
                documents=b_texts,
                metadatas=b_meta
            )
            total_added += len(b_ids)

        return total_added

    def embed_query(self, query: str) -> List[float]:
        """
        Embed query using task_type='RETRIEVAL_QUERY' for optimal semantic search alignment.
        """
        response = self.client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        return response.embeddings[0].values

    def query(self, query_text: str, top_k: int = config.FIRST_STAGE_RETRIEVAL_K, collection_name: str = config.DEFAULT_COLLECTION_NAME) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        query_vector = self.embed_query(query_text)
        actual_k = min(top_k, count)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, cid, dist in zip(docs, metas, ids, distances):
                # Cosine distance: similarity = 1.0 - distance
                similarity = round(max(0.0, 1.0 - dist), 4)
                retrieved.append({
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity
                })

        return retrieved

    def get_collection_stats(self, collection_name: str = config.DEFAULT_COLLECTION_NAME) -> Dict[str, Any]:
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()
        
        # Get unique source documents
        sample = collection.get(include=["metadatas"])
        sources = set()
        if sample and "metadatas" in sample:
            for m in sample["metadatas"]:
                if m and "source" in m:
                    sources.add(m["source"])

        return {
            "name": collection_name,
            "total_chunks": count,
            "documents": sorted(list(sources))
        }

    def clear_collection(self, collection_name: str = config.DEFAULT_COLLECTION_NAME):
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass
