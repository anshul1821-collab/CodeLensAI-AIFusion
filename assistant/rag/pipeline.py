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
    and salient keywords (function signatures, default parameters, data types, errors)
    across all supported programming languages (Python, JavaScript, Java, C/C++, SQL).
    """
    query_parts = []
    
    if problem_description.strip():
        query_parts.append(problem_description.strip())

    lang_lower = (language or "").lower()

    # --- Python Specific Patterns ---
    if lang_lower in ("python", "py"):
        fn_defs = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)', code)
        for fn_name, params in fn_defs:
            query_parts.append(f"function {fn_name} parameter {params}")
            if "=" in params:
                query_parts.append("default argument value parameter evaluation")
        if "[]" in code or "{}" in code:
            query_parts.append("mutable default argument list dictionary parameter")
        if "range(" in code:
            query_parts.append("range len sequence index out of range bounds indexerror")
        if "except" in code:
            query_parts.append("exception handling try except runtime error")

    # --- JavaScript / TypeScript Specific Patterns ---
    elif lang_lower in ("javascript", "typescript", "js", "ts"):
        if "undefined" in code or "null" in code or "?." in code:
            query_parts.append("TypeError Cannot read properties of undefined null optional chaining")
        if "async" in code or "await" in code or "Promise" in code:
            query_parts.append("async await Promise unhandled rejection asynchronous event loop")
        if "var " in code or "let " in code or "const " in code:
            query_parts.append("scoping temporal dead zone hoisting closure var let const")
        if "==" in code:
            query_parts.append("equality type coercion loose equality strict equality")
        if ".sort(" in code or ".splice(" in code or ".map(" in code:
            query_parts.append("array mutability shallow copy deep copy structuredClone")

    # --- Java Specific Patterns ---
    elif lang_lower == "java":
        if "null" in code or "Optional" in code:
            query_parts.append("NullPointerException NPE null reference Optional defensive check")
        if "==" in code and ("String" in code or '"' in code):
            query_parts.append("String comparison equals string pool memory reference")
        if "ConcurrentModificationException" in code or ".remove(" in code:
            query_parts.append("ConcurrentModificationException collection iteration iterator remove")
        if "try (" in code or "Connection" in code or "Stream" in code:
            query_parts.append("try-with-resources AutoCloseable resource leak connection pool")
        if "[]" in code or "length" in code:
            query_parts.append("ArrayIndexOutOfBoundsException array index zero-based bounds")

    # --- C / C++ Specific Patterns ---
    elif lang_lower in ("cpp", "c", "c++"):
        if "*" in code or "->" in code or "nullptr" in code or "NULL" in code:
            query_parts.append("Segmentation fault SIGSEGV pointer dereference dangling wild pointer")
        if "malloc" in code or "free" in code or "new" in code or "delete" in code:
            query_parts.append("memory leak double free use after free smart pointer unique_ptr RAII")
        if "char[" in code or "strcpy" in code or "buffer" in code:
            query_parts.append("buffer overflow stack smashing array bounds bounds violation")
        if "virtual" in code or "destructor" in code or "~" in code:
            query_parts.append("virtual destructor polymorphic base class memory leak slicing")

    # --- SQL / Database Specific Patterns ---
    elif lang_lower in ("sql", "database"):
        if "SELECT" in code.upper() or "WHERE" in code.upper():
            query_parts.append("SQL query execution injection parameterized query prepared statement")
        if "NULL" in code.upper():
            query_parts.append("IS NULL three-valued logic UNKNOWN COALESCE")
        if "JOIN" in code.upper() or "all()" in code:
            query_parts.append("N+1 query problem select_related prefetch_related eager loading")
        if "TRANSACTION" in code.upper() or "commit" in code.lower():
            query_parts.append("transaction deadlock ACID isolation connection pool exhaustion")

    # Universal error indicators
    if "/" in code:
        query_parts.append("zerodivisionerror division by zero")
    if "index" in problem_description.lower() or "bounds" in problem_description.lower():
        query_parts.append("index bounds off-by-one boundary checking")

    # Include language tag
    query_parts.append(f"{language} programming documentation error handling")

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
