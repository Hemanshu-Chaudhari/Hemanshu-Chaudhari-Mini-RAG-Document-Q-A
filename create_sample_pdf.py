"""
Generates a multi-page sample PDF for testing the RAG pipeline.
"""
from pathlib import Path
from pypdf import PdfWriter
from io import BytesIO

# Simple raw PDF template creator
PAGE_1_TEXT = """
Retrieval-Augmented Generation (RAG) Architecture Report
Author: AI Systems Engineering Group
Date: September 2026

1. Executive Summary
Retrieval-Augmented Generation (RAG) is a technique that augments large language model (LLM) prompts with verified external knowledge retrieved from vector databases. By doing so, RAG dramatically reduces hallucinations, guarantees factual accuracy, and allows AI models to answer questions using proprietary or specialized internal documentation.

2. Document Ingestion and Chunking
Document ingestion begins by extracting raw text from document formats such as PDFs, Markdown, and HTML. In our implementation, text is split using a SemanticStructureChunker. The chunker preserves natural paragraph and sentence boundaries, avoiding the mid-thought truncations common to fixed-character splitters. A target chunk size of 800 characters with an overlap window of 150 characters is maintained to preserve semantic continuity across chunk boundaries.
"""

PAGE_2_TEXT = """
3. Vector Embeddings and ChromaDB Storage
Each document chunk is transformed into a dense vector embedding using Google Gemini's text-embedding-004 model. Crucially, document chunks are indexed using the task_type RETRIEVAL_DOCUMENT, which trains the embedding projection for optimal storage. 

Embeddings are saved to ChromaDB, a lightweight, persistent vector database. ChromaDB calculates cosine distance to determine semantic similarity between vectors. Metadata including source file name, page numbers, and chunk indices are attached to each vector record.

4. Two-Stage Retrieval and LLM Reranking
When a user submits a query, two retrieval stages execute:
First Stage: ChromaDB executes vector search using a RETRIEVAL_QUERY embedding to retrieve the top 8 candidates.
Second Stage: A Gemini-powered LLM reranker examines the candidates and scores their direct relevance to the query on a scale of 0 to 10. The top 3 highest-scoring chunks are selected for context synthesis.
"""

PAGE_3_TEXT = """
5. Conversational Memory and Query Reformulation
In interactive chat scenarios, users frequently ask contextual follow-up questions such as "What was the second stage?" or "Can you explain its benefits?". These questions fail in standard vector search because pronouns lack self-contained meaning. 

To resolve this, our Conversational Memory module maintains multi-turn history and uses Gemini to rewrite contextual questions into standalone search queries before querying ChromaDB.

6. Conclusion and Key Findings
The combination of semantic chunking, dual-task Gemini embeddings, ChromaDB vector indexing, LLM reranking, and conversational query rewriting produces a robust, production-grade PDF Q&A system with verified page citations.
"""


def create_minimal_pdf(filename: str = "sample_rag_report.pdf"):
    # Create raw PDF content with 3 pages
    # Using clean standard PDF objects
    p1 = PAGE_1_TEXT.strip().replace("\n", " ")
    p2 = PAGE_2_TEXT.strip().replace("\n", " ")
    p3 = PAGE_3_TEXT.strip().replace("\n", " ")

    def make_stream(txt):
        # Escape parenthesis
        escaped = txt.replace("(", "\\(").replace(")", "\\)")
        # Wrap words to fit lines
        words = escaped.split()
        lines = []
        cur_line = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) > 75:
                lines.append(" ".join(cur_line))
                cur_line = [w]
                cur_len = len(w)
            else:
                cur_line.append(w)
                cur_len += len(w) + 1
        if cur_line:
            lines.append(" ".join(cur_line))

        stream_body = "BT\n/F1 10 Tf\n50 740 Td\n14 TL\n"
        for line in lines:
            stream_body += f"({line}) '\n"
        stream_body += "ET"
        return stream_body

    s1 = make_stream(p1)
    s2 = make_stream(p2)
    s3 = make_stream(p3)

    pdf_template = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R 5 0 R] /Count 3 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >> endobj
4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 8 0 R >> endobj
5 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 9 0 R >> endobj
6 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
7 0 obj << /Length {len(s1.encode('latin1'))} >>
stream
{s1}
endstream
endobj
8 0 obj << /Length {len(s2.encode('latin1'))} >>
stream
{s2}
endstream
endobj
9 0 obj << /Length {len(s3.encode('latin1'))} >>
stream
{s3}
endstream
endobj
xref
0 10
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000130 00000 n 
0000000247 00000 n 
0000000364 00000 n 
0000000481 00000 n 
0000000562 00000 n 
0000000750 00000 n 
0000000950 00000 n 
trailer << /Size 10 /Root 1 0 R >>
startxref
1200
%%EOF
"""
    out_path = Path(__file__).resolve().parent / filename
    with open(out_path, "wb") as f:
        f.write(pdf_template.encode("latin1"))

    print(f"Sample 3-page test PDF created at: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    create_minimal_pdf()
