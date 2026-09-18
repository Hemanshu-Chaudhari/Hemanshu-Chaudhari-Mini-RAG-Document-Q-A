import json
import re
from typing import List, Dict, Any
from google import genai
import config


class LLMReranker:
    """
    Two-stage RAG reranker: Evaluates candidate chunks retrieved from vector search
    and scores their direct relevance to answering the user's question.
    """

    def __init__(self, client: genai.Client, model: str = config.GENERATION_MODEL):
        self.client = client
        self.model = model

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = config.FINAL_RERANK_K) -> List[Dict[str, Any]]:
        """
        Takes candidates from vector search and re-ranks them based on relevance score (0-10).
        Returns top_k chunks.
        """
        if not candidate_chunks:
            return []

        if len(candidate_chunks) <= top_k:
            return candidate_chunks

        # Format candidates for scoring
        chunks_repr = []
        for idx, item in enumerate(candidate_chunks):
            preview = item["text"][:300].replace("\n", " ")
            page = item["metadata"].get("page", "?")
            chunks_repr.append(f"[{idx}] (Page {page}): {preview}")

        candidates_text = "\n".join(chunks_repr)

        prompt = f"""You are a high-precision document relevance evaluator for a RAG system.
Evaluate how relevant each passage is for directly answering the given user query.

User Query: "{query}"

Candidate Passages:
{candidates_text}

Task:
Score each passage index from 0 to 10 (10 = directly answers the query with high relevance, 0 = completely irrelevant).
Return your evaluation strictly as a valid JSON array of objects with keys "index" (integer) and "score" (number 0-10), sorted in descending order of score.

Example format:
[
  {{"index": 2, "score": 9.5}},
  {{"index": 0, "score": 7.0}}
]
"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            raw_text = response.text.strip()
            
            # Extract JSON from code fences or raw text
            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                scores_data = json.loads(json_match.group(0))
                
                reranked = []
                seen_indices = set()
                for entry in scores_data:
                    idx = entry.get("index")
                    score = entry.get("score", 0.0)
                    if isinstance(idx, int) and 0 <= idx < len(candidate_chunks) and idx not in seen_indices:
                        chunk = dict(candidate_chunks[idx])
                        chunk["rerank_score"] = float(score)
                        reranked.append(chunk)
                        seen_indices.add(idx)

                # Append any missed candidates as fallback
                for i, c in enumerate(candidate_chunks):
                    if i not in seen_indices:
                        c_copy = dict(c)
                        c_copy["rerank_score"] = 0.0
                        reranked.append(c_copy)

                return reranked[:top_k]
        except Exception:
            # Fallback to vector search order if reranking encounters an issue
            pass

        return candidate_chunks[:top_k]
