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
        self.assertIn("Mutable Default", top_issue["title"])
        self.assertIn(top_issue["severity"].capitalize(), ["Critical", "Warning"])
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

    def test_history_api(self):
        resp = self.client.get('/api/history/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("history", data)

    def test_demos_api(self):
        resp = self.client.get('/api/demos/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(len(data["demos"]), 10)

    def test_verify_api(self):
        resp = self.client.post(
            '/api/verify/',
            data=json.dumps({"language": "python", "code": "x = 10\nprint(x)"}),
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("verification", data)
        self.assertEqual(data["verification"]["syntax_check"]["status"], "Passed")


class JudgeDemosAnalysisTests(TestCase):
    """Verifies that all 10 official competition judge demos are accurately detected and safely fixed."""

    def setUp(self):
        self.generator = ContextGroundedGenerator()

    def test_demo_1_index_error(self):
        code = "numbers = [10, 20, 30]\nfor i in range(len(numbers) + 1):\n    print(numbers[i])"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("IndexError", top["title"])
        self.assertEqual(top["severity"], "Critical")
        self.assertIn("range(len(numbers))", res["corrected_code"])
        self.assertIn("verification", res)

    def test_demo_2_name_error_typo(self):
        code = "def calculate_total():\n    price = 100\n    quantity = 3\n    return price * quntity\nprint(calculate_total())"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("NameError", top["title"])
        self.assertIn("quntity", top["title"])
        self.assertIn("quantity", top["explanation"])
        self.assertIn("quantity", res["corrected_code"])

    def test_demo_3_type_error(self):
        code = "age = 20\nmessage = \"Age: \" + age\nprint(message)"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("TypeError", top["title"])
        self.assertIn("str(age)", res["corrected_code"])

    def test_demo_4_zero_division(self):
        code = "def divide(a, b):\n    return a / b\nprint(divide(10, 0))"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("ZeroDivisionError", top["title"])
        self.assertEqual(top["severity"], "Critical")

    def test_demo_5_logic_error(self):
        code = "def is_adult(age):\n    return age > 18"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("Logic Error", top["title"])
        self.assertIn("business rule", top["explanation"].lower())
        self.assertIn(">= 18", res["corrected_code"])

    def test_demo_6_syntax_error(self):
        code = "def greet(name)\n    print(\"Hello\", name)"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("SyntaxError", top["title"])
        self.assertIn("def greet(name):", res["corrected_code"])

    def test_demo_7_key_error(self):
        code = "student = {\"name\": \"Aarav\", \"age\": 20}\nprint(student[\"grade\"])"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("KeyError", top["title"])
        self.assertIn("get(\"grade\"", res["corrected_code"])

    def test_demo_8_attribute_error(self):
        code = "text = \"hello\"\ntext.append(\"!\")"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("AttributeError", top["title"])
        self.assertIn("+=", res["corrected_code"])

    def test_demo_9_value_error(self):
        code = "number = int(\"hello\")\nprint(number)"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("ValueError", top["title"])

    def test_demo_10_import_error(self):
        code = "import module_that_does_not_exist"
        res = self.generator.generate_analysis("python", code, "", {"is_sufficient": False, "results": []})
        self.assertGreater(len(res["potential_issues"]), 0)
        top = res["potential_issues"][0]
        self.assertIn("ModuleNotFoundError", top["title"])
        self.assertIn("not be automatically installed", top["explanation"].lower())

