import re
import os
from pathlib import Path
from typing import Dict, Any, Optional
from django.core.files.base import ContentFile
from assistant.models import Document, DocumentChunk, AnalysisRecord
from assistant.rag.extractor import extract_document
from assistant.rag.chunker import chunk_document
from assistant.rag.embedder import get_active_embedder
from assistant.rag.vector_store import VectorStore
from assistant.rag.generator import ContextGroundedGenerator


def create_retrieval_query(language: str, code: str, problem_description: str) -> str:
    """
    Constructs an optimized semantic search query by extracting problem context
    and salient keywords (function signatures, default parameters, data types, errors).
    """
    query_parts = []
    
    if problem_description.strip():
        query_parts.append(problem_description.strip())

    # Extract function definitions
    fn_defs = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)', code)
    for fn_name, params in fn_defs:
        query_parts.append(f"function {fn_name} parameter {params}")
        if "=" in params:
            query_parts.append("default argument value parameter evaluation")

    # Extract common error patterns
    if "[]" in code or "{}" in code:
        query_parts.append("mutable default argument list dictionary parameter")
    if "range(" in code:
        query_parts.append("range len sequence index out of range bounds indexerror")
    if "except" in code:
        query_parts.append("exception handling try except runtime error")
    if "/" in code:
        query_parts.append("zerodivisionerror division by zero")

    # Include language tag
    query_parts.append(f"{language} programming documentation")

    return " ".join(query_parts)


class RAGPipeline:
    """
    End-to-end RAG orchestrator for CodeLens AI.
    Handles ingestion, chunking, embedding, retrieval, and grounded response generation.
    """

    def __init__(self):
        self.embedder = get_active_embedder()
        self.vector_store = VectorStore(self.embedder)
        self.generator = ContextGroundedGenerator()

    def ingest_document(
        self,
        file_source,
        filename: str,
        file_size: int,
        is_sample: bool = False
    ) -> Document:
        """
        Executes full document ingestion lifecycle:
        1. Extract text & preserve page structure
        2. Create Document database record
        3. Split text into boundary-aware overlapping chunks
        4. Generate embeddings and persist in vector store
        """
        extracted = extract_document(file_source, filename, file_size)

        doc = Document.objects.create(
            title=extracted["title"],
            filename=extracted["filename"],
            file_type=extracted["file_type"],
            file_size=extracted["file_size"],
            total_pages=extracted["total_pages"],
            is_sample=is_sample
        )

        # Chunk pages
        chunks = chunk_document(extracted["pages"])

        # Index in vector store
        indexed_count = self.vector_store.index_document_chunks(doc, chunks)

        return doc

    def analyze_code(
        self,
        language: str,
        code: str,
        problem_description: str = "",
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Full RAG code explanation and debugging workflow:
        1. Formulate semantic query
        2. Retrieve real matching document chunks from vector index
        3. Validate grounding threshold
        4. Generate context-grounded analysis with source citations
        5. Save record and return structured results
        """
        if not code.strip():
            raise ValueError("No code was provided for analysis.")

        # 1. Query construction
        retrieval_query = create_retrieval_query(language, code, problem_description)

        # 2. Semantic retrieval
        retrieval_result = self.vector_store.search(
            query=retrieval_query,
            top_k=top_k,
            min_similarity=0.18
        )

        # 3. Grounded generation
        analysis = self.generator.generate_analysis(
            language=language,
            code=code,
            problem_description=problem_description,
            retrieval_result=retrieval_result
        )

        # 4. Persist analysis record
        import json
        record = AnalysisRecord.objects.create(
            language=language,
            submitted_code=code,
            problem_description=problem_description,
            summary=analysis.get("summary", ""),
            code_explanation=analysis.get("code_explanation", ""),
            potential_issues_json=json.dumps(analysis.get("potential_issues", [])),
            suggested_fix=analysis.get("suggested_fix", ""),
            corrected_code=analysis.get("corrected_code", ""),
            retrieved_sources_json=json.dumps(analysis.get("retrieved_sources", [])),
            is_grounded=analysis.get("is_grounded", True),
            grounding_message=analysis.get("grounding_message", ""),
            model_used=analysis.get("model_used", "CodeLens Engine")
        )

        analysis["record_id"] = record.id
        analysis["created_at"] = record.created_at.strftime("%Y-%m-%d %H:%M:%S")
        analysis["retrieval_query"] = retrieval_query

        return analysis
