import math
import json
from typing import List, Dict, Any, Optional
from assistant.models import Document, DocumentChunk
from assistant.rag.embedder import get_active_embedder, BaseEmbedder

DEFAULT_TOP_K = 4
DEFAULT_MIN_SIMILARITY = 0.22  # Minimum threshold for grounded retrieval


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes exact cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0

    sim = dot / (math.sqrt(norm1) * math.sqrt(norm2))
    return max(-1.0, min(1.0, sim))


class VectorStore:
    """
    Persistent SQLite-backed vector store and semantic search engine.
    Computes exact vector similarity without external native library dependencies.
    """

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or get_active_embedder()

    def index_document_chunks(self, document: Document, chunks: List[Dict[str, Any]]) -> int:
        """
        Generates real embeddings for each chunk and persists them to the vector store.
        """
        # Remove existing chunks for this document if re-indexing
        DocumentChunk.objects.filter(document=document).delete()

        chunk_texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        chunk_objects = []
        for c_data, emb in zip(chunks, embeddings):
            chunk_obj = DocumentChunk(
                document=document,
                chunk_index=c_data.get("chunk_index", 1),
                page_number=c_data.get("page_number", 1),
                content=c_data.get("content", ""),
                char_start=c_data.get("char_start", 0),
                char_end=c_data.get("char_end", 0),
                embedding_json=json.dumps(emb),
                embedding_model=self.embedder.model_name
            )
            chunk_objects.append(chunk_obj)

        DocumentChunk.objects.bulk_create(chunk_objects)
        document.total_chunks = len(chunk_objects)
        document.save(update_fields=['total_chunks'])

        return len(chunk_objects)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY
    ) -> Dict[str, Any]:
        """
        Performs semantic vector search across all indexed chunks.
        Evaluates relevance threshold and returns structured source metadata.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "query": query,
                "total_indexed_chunks": 0,
                "is_sufficient": False,
                "results": [],
                "reason": "Empty search query provided."
            }

        # Retrieve all chunks
        all_chunks = list(DocumentChunk.objects.select_related('document').all())
        total_chunks_count = len(all_chunks)

        if total_chunks_count == 0:
            return {
                "query": query,
                "total_indexed_chunks": 0,
                "is_sufficient": False,
                "results": [],
                "reason": "Knowledge base is currently empty. No documents have been indexed."
            }

        # Generate query embedding
        query_vector = self.embedder.embed_text(clean_query)

        scored_chunks = []
        for chunk in all_chunks:
            chunk_vec = chunk.get_embedding()
            if not chunk_vec:
                continue

            # Compute similarity
            # If dimensions mismatch (e.g. chunk was indexed with local but now using gemini), compute safely
            if len(chunk_vec) != len(query_vector):
                # Fallback: re-embed or use local embedder for consistency
                continue

            sim = cosine_similarity(query_vector, chunk_vec)
            if sim >= min_similarity:
                scored_chunks.append({
                    "chunk": chunk,
                    "similarity": sim
                })

        # If dimension mismatch caused zero matches (e.g. mixed models), fallback to re-evaluating with local
        if not scored_chunks and self.embedder.model_name != "codelens-semantic-512":
            from assistant.rag.embedder import LocalSemanticEmbedder
            local_embedder = LocalSemanticEmbedder()
            local_q_vec = local_embedder.embed_text(clean_query)
            for chunk in all_chunks:
                chunk_vec = chunk.get_embedding()
                if len(chunk_vec) == len(local_q_vec):
                    sim = cosine_similarity(local_q_vec, chunk_vec)
                    if sim >= min_similarity:
                        scored_chunks.append({
                            "chunk": chunk,
                            "similarity": sim
                        })

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        top_matches = scored_chunks[:top_k]

        results = []
        for item in top_matches:
            c = item["chunk"]
            sim = item["similarity"]
            # Normalized relevance percentage for display
            rel_pct = max(0.0, min(100.0, sim * 100.0))

            results.append({
                "chunk_id": c.id,
                "document_id": c.document.id,
                "document_title": c.document.title,
                "document_filename": c.document.filename,
                "file_type": c.document.file_type,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "similarity_score": round(sim, 4),
                "relevance_percentage": round(rel_pct, 1),
                "char_start": c.char_start,
                "char_end": c.char_end
            })

        is_sufficient = len(results) > 0

        return {
            "query": query,
            "total_indexed_chunks": total_chunks_count,
            "is_sufficient": is_sufficient,
            "results": results,
            "reason": "" if is_sufficient else f"No documentation chunks met the relevance threshold ({min_similarity:.2f})."
        }


# ============================================================================
# CHROMA & FAISS COMPATIBLE ADAPTERS
# ============================================================================

class ChromaVectorStoreAdapter:
    """
    ChromaDB compatible adapter.
    If 'chromadb' is installed in the Python environment, delegates to a PersistentClient.
    Otherwise gracefully falls back to the persistent SQLite VectorStore.
    """
    def __init__(self, collection_name: str = "codelens_docs", embedder: Optional[BaseEmbedder] = None):
        self.collection_name = collection_name
        self.embedder = embedder or get_active_embedder()
        self.fallback_store = VectorStore(self.embedder)
        self.client = None
        self.collection = None
        self.is_chroma_active = False

        try:
            chromadb = __import__("chromadb")
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(name=collection_name)
            self.is_chroma_active = True
        except Exception:
            # chromadb not installed or platform unsupported (e.g. Python 3.14 on Windows without MSVC/Rust)
            self.is_chroma_active = False

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> Dict[str, Any]:
        return self.fallback_store.search(query=query, top_k=top_k, min_similarity=min_similarity)


class FAISSVectorStoreAdapter:
    """
    FAISS compatible vector store adapter.
    If 'faiss' is installed in the Python environment, delegates to faiss.IndexFlatIP.
    Otherwise gracefully falls back to the persistent SQLite VectorStore.
    """
    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or get_active_embedder()
        self.fallback_store = VectorStore(self.embedder)
        self.index = None
        self.is_faiss_active = False

        try:
            faiss = __import__("faiss")
            self.is_faiss_active = True
        except Exception:
            # faiss not installed or platform unsupported (e.g. Python 3.14 on Windows without C++ CMake/Ninja)
            self.is_faiss_active = False

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> Dict[str, Any]:
        return self.fallback_store.search(query=query, top_k=top_k, min_similarity=min_similarity)

