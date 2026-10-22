import json
import io
from pathlib import Path
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from assistant.models import Document, DocumentChunk, AnalysisRecord, SystemSetting
from assistant.rag.extractor import extract_document, DocumentExtractionError
from assistant.rag.chunker import chunk_document, TextChunker
from assistant.rag.embedder import LocalSemanticEmbedder
from assistant.rag.vector_store import VectorStore, cosine_similarity
from assistant.rag.generator import ContextGroundedGenerator, GROUNDING_LIMITATION_MESSAGE
from assistant.rag.pipeline import RAGPipeline


class DocumentProcessingTests(TestCase):
    """Verifies ingestion, validation, and extraction for PDF and TXT files."""

    def test_valid_txt_extraction(self):
        txt_content = b"Python functions have default argument values evaluated at definition time.\n" * 10
        doc = extract_document(txt_content, "python_tutorial.txt", len(txt_content))
        self.assertEqual(doc["file_type"], "txt")
        self.assertGreater(doc["total_pages"], 0)
        self.assertIn("Python functions", doc["full_text"])

    def test_valid_pdf_extraction(self):
        pdf_path = settings.BASE_DIR / 'sample_docs' / 'python_data_structures_and_memory.pdf'
        if pdf_path.exists():
            with open(pdf_path, 'rb') as f:
                content = f.read()
            doc = extract_document(content, "python_data_structures_and_memory.pdf", len(content))
            self.assertEqual(doc["file_type"], "pdf")
            self.assertEqual(doc["total_pages"], 2)
            self.assertIn("Mutable Defaults", doc["full_text"])

    def test_empty_document_rejection(self):
        with self.assertRaises(DocumentExtractionError):
            extract_document(b"", "empty.txt", 0)

    def test_unsupported_file_extension(self):
        with self.assertRaises(DocumentExtractionError):
            extract_document(b"alert('hello')", "script.js", 14)

    def test_malformed_pdf_handling(self):
        malformed_bytes = b"%PDF-1.4 corrupt header without catalog or xref %%EOF"
        with self.assertRaises(DocumentExtractionError):
            extract_document(malformed_bytes, "corrupt.pdf", len(malformed_bytes))


class ChunkingAndEmbeddingTests(TestCase):
    """Verifies chunking metadata and deterministic semantic embedding behavior."""

    def test_chunking_metadata_and_boundaries(self):
        pages = [
            {"page_number": 1, "text": "Paragraph 1 about python lists.\n\nParagraph 2 about mutable arguments."},
            {"page_number": 2, "text": "Paragraph 3 about exceptions and index error."}
        ]
        chunks = chunk_document(pages, chunk_size=100, chunk_overlap=20)
        self.assertGreaterEqual(len(chunks), 2)
        # Check metadata
        for chunk in chunks:
            self.assertIn("chunk_index", chunk)
            self.assertIn("page_number", chunk)
            self.assertIn("content", chunk)
            self.assertIn("char_start", chunk)
            self.assertIn("char_end", chunk)

    def test_deterministic_semantic_embeddings(self):
        embedder = LocalSemanticEmbedder()
        v1 = embedder.embed_text("mutable default argument in python function")
        v2 = embedder.embed_text("mutable default argument in python function")
        # Exact reproducibility
        self.assertEqual(v1, v2)
        self.assertEqual(len(v1), 512)

        # Semantic similarity check: related concept vs unrelated concept
        v_related = embedder.embed_text("default parameter evaluation list append bug")
        v_unrelated = embedder.embed_text("cooking pasta with olive oil and tomato sauce")

        sim_related = cosine_similarity(v1, v_related)
        sim_unrelated = cosine_similarity(v1, v_unrelated)

        self.assertGreater(sim_related, sim_unrelated)
        self.assertGreater(sim_related, 0.30)
        self.assertLess(sim_unrelated, 0.15)


class VectorStoreAndRetrievalTests(TestCase):
    """Verifies indexing, persistent storage, and real vector retrieval."""

    def setUp(self):
        self.embedder = LocalSemanticEmbedder()
        self.store = VectorStore(self.embedder)
        
        # Create a test document
        self.doc = Document.objects.create(
            title="Python Functions Reference",
            filename="functions.txt",
            file_type="txt",
            file_size=500,
            total_pages=1
        )
        test_chunks = [
            {
                "chunk_index": 1,
                "page_number": 1,
                "content": "Default parameter values are evaluated once at function definition time. Using mutable defaults like [] causes shared state across calls.",
                "char_start": 0,
                "char_end": 120
            },
            {
                "chunk_index": 2,
                "page_number": 1,
                "content": "Python sequences are 0-indexed. An IndexError occurs if an index is outside sequence length.",
                "char_start": 121,
                "char_end": 220
            }
        ]
        self.store.index_document_chunks(self.doc, test_chunks)

    def test_chunks_indexed_in_database(self):
        chunks_count = DocumentChunk.objects.filter(document=self.doc).count()
        self.assertEqual(chunks_count, 2)

    def test_semantic_query_retrieval(self):
        search_res = self.store.search("mutable default list parameter", top_k=2)
        self.assertTrue(search_res["is_sufficient"])
        self.assertGreater(len(search_res["results"]), 0)
        
        # Top result should be the mutable default chunk
        top_match = search_res["results"][0]
        self.assertEqual(top_match["chunk_index"], 1)
        self.assertIn("Default parameter values are evaluated", top_match["content"])
        self.assertEqual(top_match["document_title"], "Python Functions Reference")

    def test_irrelevant_query_grounding_limitation(self):
        search_res = self.store.search("astronomy telescope celestial orbits astrophysics", top_k=2, min_similarity=0.45)
        # Should flag insufficient results
        self.assertFalse(search_res["is_sufficient"])
        self.assertEqual(len(search_res["results"]), 0)


class RAGAnalysisPipelineTests(TestCase):
    """Verifies the complete end-to-end RAG analysis and static safety."""

    def setUp(self):
        self.pipeline = RAGPipeline()
        # Ingest sample docs
        sample_path = settings.BASE_DIR / 'sample_docs' / 'python_default_arguments_and_functions.txt'
        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                self.pipeline.ingest_document(f, sample_path.name, sample_path.stat().st_size, is_sample=True)

    def test_mutable_default_argument_analysis(self):
        code = "def append_val(val, my_list=[]):\n    my_list.append(val)\n    return my_list"
        result = self.pipeline.analyze_code(
            language="python",
            code=code,
            problem_description="Subsequent calls preserve previous items in my_list."
        )

        self.assertTrue(result["is_grounded"])
        self.assertIn("summary", result)
        self.assertIn("code_explanation", result)
        self.assertGreater(len(result["potential_issues"]), 0)

        # Verify identified issue
        top_issue = result["potential_issues"][0]
        self.assertIn("Mutable Default Argument", top_issue["title"])
        self.assertEqual(top_issue["severity"], "Critical")
        self.assertIn("None", result["corrected_code"])

        # Verify source citations exist and are populated
        self.assertGreater(len(result["retrieved_sources"]), 0)
        top_src = result["retrieved_sources"][0]
        self.assertIn("page_number", top_src)
        self.assertIn("relevance_percentage", top_src)

    def test_untrusted_code_is_never_executed(self):
        # Pass dangerous code that would create a file if executed
        canary_file = settings.BASE_DIR / 'test_canary_should_not_exist.txt'
        if canary_file.exists():
            canary_file.unlink()

        malicious_code = f"import os\nopen('{canary_file.name}', 'w').write('HACKED')\n"
        result = self.pipeline.analyze_code("python", malicious_code)

        # The file MUST NOT exist!
        self.assertFalse(canary_file.exists(), "SECURITY VIOLATION: Submitted code was executed!")
        self.assertIn("validation_notice", result)


class APIEndpointTests(TestCase):
    """Verifies HTTP API endpoints and responses."""

    def setUp(self):
        self.client = Client()

    def test_status_api(self):
        resp = self.client.get('/api/status/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("total_documents", data)
        self.assertIn("active_embedder", data)

    def test_load_samples_api(self):
        resp = self.client.post('/api/documents/load-samples/', content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_documents"], 0)

    def test_analyze_api_empty_code_rejection(self):
        resp = self.client.post('/api/analyze/', data=json.dumps({"code": ""}), content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_search_preview_api(self):
        resp = self.client.post(
            '/api/search-preview/',
            data=json.dumps({"query": "python mutable default", "top_k": 3}),
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("results", data["data"])
