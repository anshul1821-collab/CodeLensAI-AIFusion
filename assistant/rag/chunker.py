import re
from typing import List, Dict, Any

DEFAULT_CHUNK_SIZE = 550
DEFAULT_CHUNK_OVERLAP = 120


class TextChunker:
    """
    Transparent, boundary-aware text chunking strategy.
    Maintains page references and splits on semantic boundaries (paragraphs, sentences).
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_page_text(self, text: str, page_number: int, start_chunk_idx: int = 1) -> List[Dict[str, Any]]:
        """
        Chunks text from a single page, attaching page number and char positions.
        """
        text = text.strip()
        if not text:
            return []

        chunks = []
        text_len = len(text)
        current_start = 0
        chunk_idx = start_chunk_idx

        # Splitting delimiters in order of preference
        split_patterns = [
            r'\n\n+',       # Paragraph boundary
            r'\n',          # Line break
            r'(?<=[.!?])\s+', # Sentence boundary
            r';\s+',        # Statement break
            r'\s+'          # Word boundary
        ]

        while current_start < text_len:
            target_end = min(current_start + self.chunk_size, text_len)

            if target_end >= text_len:
                chunk_slice = text[current_start:text_len].strip()
                if chunk_slice:
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "page_number": page_number,
                        "content": chunk_slice,
                        "char_start": current_start,
                        "char_end": text_len,
                        "char_count": len(chunk_slice),
                        "word_count": len(chunk_slice.split())
                    })
                break

            # Find the best split point near target_end
            best_split = -1
            search_window_start = max(current_start + self.chunk_size - self.chunk_overlap, current_start)
            window = text[search_window_start:target_end]

            for pat in split_patterns:
                matches = list(re.finditer(pat, window))
                if matches:
                    last_match = matches[-1]
                    best_split = search_window_start + last_match.end()
                    break

            if best_split == -1 or best_split <= current_start:
                # If no boundary found, hard break at target_end
                best_split = target_end

            chunk_slice = text[current_start:best_split].strip()
            if chunk_slice:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "page_number": page_number,
                    "content": chunk_slice,
                    "char_start": current_start,
                    "char_end": best_split,
                    "char_count": len(chunk_slice),
                    "word_count": len(chunk_slice.split())
                })
                chunk_idx += 1

            # Advance by step taking overlap into account
            step = max(1, (best_split - current_start) - self.chunk_overlap)
            current_start += step

        return chunks

    def chunk_document_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks an entire document given its extracted pages list.
        """
        all_chunks = []
        global_chunk_idx = 1

        for page in pages:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")
            page_chunks = self.split_page_text(page_text, page_number=page_num, start_chunk_idx=global_chunk_idx)
            all_chunks.extend(page_chunks)
            global_chunk_idx += len(page_chunks)

        return all_chunks


def chunk_document(pages: List[Dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_document_pages(pages)
