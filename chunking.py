import re
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


class SemanticStructureChunker:
    """
    Splits document text while preserving paragraph boundaries, 
    heading structures, and sentences to ensure high semantic coherence.
    """

    def __init__(self, target_chars: int = 800, overlap_chars: int = 150):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def split_into_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries (. ! ?) followed by whitespace or quotes
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_page_text(self, page_text: str, source_name: str, page_num: int, start_idx: int = 0) -> List[DocumentChunk]:
        # Normalize carriage returns and non-breaking spaces
        normalized = page_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        
        # Split by paragraph / double-newlines first
        raw_paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        if not raw_paragraphs:
            raw_paragraphs = [normalized.strip()] if normalized.strip() else []

        chunks: List[DocumentChunk] = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        chunk_counter = start_idx

        # Decompose paragraphs into coherent sentence units
        sentence_units = []
        for p in raw_paragraphs:
            sentences = self.split_into_sentences(p)
            if sentences:
                sentence_units.extend(sentences)
            else:
                sentence_units.append(p)

        for sentence in sentence_units:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds target and we already have content, finalize current chunk
            if current_length + sentence_len > self.target_chars and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunk_id = f"{source_name}_p{page_num}_{chunk_counter}"
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        metadata={
                            "source": source_name,
                            "page": page_num,
                            "chunk_index": chunk_counter,
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split())
                        },
                        chunk_id=chunk_id
                    ))
                    chunk_counter += 1

                # Calculate overlap: take trailing sentences from current chunk until overlap length is met
                overlap_sentences = []
                accumulated_overlap = 0
                for s in reversed(current_chunk_sentences):
                    if accumulated_overlap + len(s) <= self.overlap_chars:
                        overlap_sentences.insert(0, s)
                        accumulated_overlap += len(s)
                    else:
                        break

                current_chunk_sentences = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_len + 1

        # Emit any remaining text
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                chunk_id = f"{source_name}_p{page_num}_{chunk_counter}"
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": chunk_counter,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    },
                    chunk_id=chunk_id
                ))

        return chunks

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Extracts text from all pages of a PDF and returns semantic chunks with page metadata.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        reader = PdfReader(str(path))
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            page_chunks = self.chunk_page_text(
                page_text=page_text,
                source_name=path.name,
                page_num=page_idx + 1,
                start_idx=global_chunk_idx
            )
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks
