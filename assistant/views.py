import json
import os
from pathlib import Path
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from django.conf import settings

from assistant.models import Document, DocumentChunk, AnalysisRecord, SystemSetting
from assistant.rag.pipeline import RAGPipeline
from assistant.rag.embedder import get_active_embedder
from assistant.rag.extractor import DocumentExtractionError


@ensure_csrf_cookie
def index_view(request):
    """
    Renders the primary CodeLens AI interface.
    Provides initial context including knowledge base metrics and recent records.
    """
    documents = Document.objects.all().order_by('-created_at')
    recent_analyses = AnalysisRecord.objects.all().order_by('-created_at')[:10]
    total_chunks = DocumentChunk.objects.count()
    
    embedder = get_active_embedder()
    gemini_key = SystemSetting.get_setting("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    has_gemini = bool(gemini_key.strip())

    context = {
        "documents": documents,
        "recent_analyses": recent_analyses,
        "total_chunks": total_chunks,
        "active_embedder": embedder.model_name,
        "has_gemini": has_gemini,
        "gemini_masked_key": f"{gemini_key[:4]}...{gemini_key[-4:]}" if len(gemini_key) > 8 else ("Configured" if has_gemini else "Not Configured"),
    }
    return render(request, "assistant/index.html", context)


@require_http_methods(["POST"])
def api_analyze_code(request):
    """
    Asynchronous API endpoint to analyze code snippets with context grounding.
    """
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({"success": False, "error": "Invalid JSON payload."}, status=400)

    language = data.get("language", "python").strip()
    code = data.get("code", "").strip()
    problem_description = data.get("problem_description", "").strip()
    top_k = int(data.get("top_k", 4))

    if not code:
        return JsonResponse({"success": False, "error": "Code snippet cannot be empty."}, status=400)

    try:
        pipeline = RAGPipeline()
        analysis = pipeline.analyze_code(
            language=language,
            code=code,
            problem_description=problem_description,
            top_k=top_k
        )
        return JsonResponse({"success": True, "data": analysis})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@require_http_methods(["POST"])
def api_upload_document(request):
    """
    Uploads, extracts, chunks, and indexes a PDF or TXT document into the vector store.
    """
    if 'document_file' not in request.FILES:
        return JsonResponse({"success": False, "error": "No file was uploaded."}, status=400)

    uploaded_file = request.FILES['document_file']
    filename = uploaded_file.name
    file_size = uploaded_file.size

    try:
        pipeline = RAGPipeline()
        doc = pipeline.ingest_document(
            file_source=uploaded_file,
            filename=filename,
            file_size=file_size,
            is_sample=False
        )

        return JsonResponse({
            "success": True,
            "message": f"Successfully indexed '{doc.title}' ({doc.total_chunks} chunks).",
            "document": {
                "id": doc.id,
                "title": doc.title,
                "filename": doc.filename,
                "file_type": doc.file_type.upper(),
                "file_size": doc.formatted_size,
                "total_pages": doc.total_pages,
                "total_chunks": doc.total_chunks,
                "created_at": doc.created_at.strftime("%Y-%m-%d %H:%M")
            }
        })
    except DocumentExtractionError as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Indexing failure: {str(e)}"}, status=500)


@require_http_methods(["POST"])
def api_delete_document(request, doc_id):
    """
    Deletes a document and all its associated vector embeddings.
    """
    doc = get_object_or_404(Document, id=doc_id)
    title = doc.title
    doc.delete()  # Cascade deletes DocumentChunk records
    return JsonResponse({"success": True, "message": f"Deleted document '{title}' and removed its vector chunks."})


@require_http_methods(["POST"])
def api_load_samples(request):
    """
    Populates knowledge base with the authentic Python documentation sample pack.
    """
    sample_dir = settings.BASE_DIR / 'sample_docs'
    if not sample_dir.exists():
        return JsonResponse({"success": False, "error": "Sample docs directory not found."}, status=404)

    pipeline = RAGPipeline()
    loaded_docs = []
    total_new_chunks = 0

    for sample_path in sample_dir.iterdir():
        if sample_path.suffix.lower() in ['.pdf', '.txt']:
            # Check if already loaded
            if Document.objects.filter(filename=sample_path.name).exists():
                continue

            file_size = sample_path.stat().st_size
            try:
                with open(sample_path, 'rb') as f:
                    doc = pipeline.ingest_document(
                        file_source=f,
                        filename=sample_path.name,
                        file_size=file_size,
                        is_sample=True
                    )
                    loaded_docs.append(doc.title)
                    total_new_chunks += doc.total_chunks
            except Exception as e:
                pass

    return JsonResponse({
        "success": True,
        "message": f"Loaded {len(loaded_docs)} sample reference documents ({total_new_chunks} chunks).",
        "loaded_titles": loaded_docs,
        "total_documents": Document.objects.count(),
        "total_chunks": DocumentChunk.objects.count()
    })


@require_http_methods(["POST"])
def api_update_settings(request):
    """
    Configures API keys and embedder preferences.
    """
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)

    gemini_key = data.get("gemini_api_key", "").strip()
    embedder_pref = data.get("embedder_preference", "auto").strip()

    if gemini_key:
        SystemSetting.set_setting("GEMINI_API_KEY", gemini_key)
    SystemSetting.set_setting("EMBEDDER_PREFERENCE", embedder_pref)

    embedder = get_active_embedder()

    return JsonResponse({
        "success": True,
        "message": "Settings updated successfully.",
        "active_embedder": embedder.model_name,
        "has_gemini": bool(gemini_key)
    })


@require_http_methods(["GET"])
def api_get_status(request):
    """
    Returns current system and knowledge base metrics.
    """
    documents = Document.objects.all().order_by('-created_at')
    doc_list = [{
        "id": d.id,
        "title": d.title,
        "filename": d.filename,
        "file_type": d.file_type.upper(),
        "file_size": d.formatted_size,
        "total_pages": d.total_pages,
        "total_chunks": d.total_chunks,
        "is_sample": d.is_sample,
        "created_at": d.created_at.strftime("%Y-%m-%d %H:%M")
    } for d in documents]

    embedder = get_active_embedder()
    gemini_key = SystemSetting.get_setting("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

    return JsonResponse({
        "success": True,
        "total_documents": len(doc_list),
        "total_chunks": DocumentChunk.objects.count(),
        "total_analyses": AnalysisRecord.objects.count(),
        "active_embedder": embedder.model_name,
        "has_gemini": bool(gemini_key.strip()),
        "documents": doc_list
    })


@require_http_methods(["POST"])
def api_search_preview(request):
    """
    Demonstrates semantic search retrieval in isolation.
    Returns matched chunks and relevance scores directly.
    """
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)

    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 4))

    pipeline = RAGPipeline()
    result = pipeline.vector_store.search(query=query, top_k=top_k)

    return JsonResponse({"success": True, "data": result})


@require_http_methods(["GET"])
def api_get_history(request):
    """
    Returns recent analysis records for the Analysis History panel.
    """
    records = AnalysisRecord.objects.all().order_by('-created_at')[:30]
    history_data = []
    for r in records:
        issues = r.issues
        crit_count = sum(1 for i in issues if str(i.get("severity", "")).lower() == "critical")
        warn_count = sum(1 for i in issues if str(i.get("severity", "")).lower() == "warning")
        history_data.append({
            "id": r.id,
            "language": r.language,
            "submitted_code": r.submitted_code,
            "problem_description": r.problem_description,
            "summary": r.summary,
            "code_explanation": r.code_explanation,
            "potential_issues": issues,
            "total_issues": len(issues),
            "critical_count": crit_count,
            "warning_count": warn_count,
            "suggested_fix": r.suggested_fix,
            "corrected_code": r.corrected_code,
            "model_used": r.model_used,
            "is_grounded": r.is_grounded,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M")
        })
    return JsonResponse({"success": True, "history": history_data})


@require_http_methods(["GET"])
def api_get_demos(request):
    """
    Returns all 10 judge demos and multi-language test scenarios.
    """
    demos = [
        {
            "id": "demo_1_index",
            "number": 1,
            "name": "Index Error",
            "category": "Runtime Error (IndexError)",
            "language": "python",
            "code": "numbers = [10, 20, 30]\n\nfor i in range(len(numbers) + 1):\n    print(numbers[i])",
            "problem": "Why does this loop crash with IndexError: list index out of range on the last iteration?",
            "expected_finding": "The loop attempts to access an index outside the list using range(len(numbers) + 1).",
            "suggested_fix": "Use range(len(numbers)) or iterate directly over list items with for num in numbers:"
        },
        {
            "id": "demo_2_name",
            "number": 2,
            "name": "Name Error",
            "category": "Runtime Error (NameError)",
            "language": "python",
            "code": "def calculate_total():\n    price = 100\n    quantity = 3\n    return price * quntity\n\nprint(calculate_total())",
            "problem": "NameError: name 'quntity' is not defined when calculating total price.",
            "expected_finding": "The variable quntity is not defined; quantity was likely intended (typographical error).",
            "suggested_fix": "Change quntity to the defined variable quantity: return price * quantity"
        },
        {
            "id": "demo_3_type",
            "number": 3,
            "name": "Type Error",
            "category": "Runtime Error (TypeError)",
            "language": "python",
            "code": "age = 20\nmessage = \"Age: \" + age\nprint(message)",
            "problem": "TypeError: can only concatenate str (not 'int') to str when building message.",
            "expected_finding": "A string literal is concatenated with an integer without conversion.",
            "suggested_fix": "Convert integer to string with str(age) or use an f-string: f'Age: {age}'"
        },
        {
            "id": "demo_4_zero_div",
            "number": 4,
            "name": "Zero Division Error",
            "category": "Runtime Error (ZeroDivision)",
            "language": "python",
            "code": "def divide(a, b):\n    return a / b\n\nprint(divide(10, 0))",
            "problem": "ZeroDivisionError: division by zero crash.",
            "expected_finding": "The call passes 0 as divisor, performing division by zero which is undefined.",
            "suggested_fix": "Add guard clause if b == 0: raise ValueError(...) or supply non-zero divisor."
        },
        {
            "id": "demo_5_logic",
            "number": 5,
            "name": "Logic Error",
            "category": "Logic Error",
            "language": "python",
            "code": "def is_adult(age):\n    return age > 18",
            "problem": "Should 18-year-olds be considered adults? Function returns False for age 18.",
            "expected_finding": "If intended rule is age 18 or older, comparison should use >= instead of >. (Finding depends on intended business rule).",
            "suggested_fix": "Change > 18 to >= 18 if 18-year-olds are considered adults."
        },
        {
            "id": "demo_6_syntax",
            "number": 6,
            "name": "Syntax Error",
            "category": "Syntax Error",
            "language": "python",
            "code": "def greet(name)\n    print(\"Hello\", name)",
            "problem": "SyntaxError: expected ':' at end of function declaration.",
            "expected_finding": "Function declaration def greet(name) is missing a colon ':' at the end of the line.",
            "suggested_fix": "Add a colon ':' at the end of def greet(name):"
        },
        {
            "id": "demo_7_key",
            "number": 7,
            "name": "Key Error",
            "category": "Data Structure Error (KeyError)",
            "language": "python",
            "code": "student = {\"name\": \"Aarav\", \"age\": 20}\nprint(student[\"grade\"])",
            "problem": "KeyError: 'grade' when accessing student profile data.",
            "expected_finding": "The dictionary does not contain the requested 'grade' key.",
            "suggested_fix": "Use student.get('grade', 'Not Assigned') or initialize 'grade' in dictionary."
        },
        {
            "id": "demo_8_attribute",
            "number": 8,
            "name": "Attribute Error",
            "category": "Runtime Error (AttributeError)",
            "language": "python",
            "code": "text = \"hello\"\ntext.append(\"!\")",
            "problem": "AttributeError: 'str' object has no attribute 'append'.",
            "expected_finding": "Strings are immutable sequence types and do not support the list append method.",
            "suggested_fix": "Use string concatenation with += (e.g. text += '!')"
        },
        {
            "id": "demo_9_value",
            "number": 9,
            "name": "Value Error",
            "category": "Runtime Error (ValueError)",
            "language": "python",
            "code": "number = int(\"hello\")\nprint(number)",
            "problem": "ValueError: invalid literal for int() with base 10: 'hello'.",
            "expected_finding": "The non-numeric string 'hello' cannot be parsed into an integer.",
            "suggested_fix": "Supply a valid numeric string (e.g. '42') or wrap in try...except ValueError."
        },
        {
            "id": "demo_10_import",
            "number": 10,
            "name": "Import Error",
            "category": "Runtime Error (ImportError)",
            "language": "python",
            "code": "import module_that_does_not_exist",
            "problem": "ModuleNotFoundError: No module named 'module_that_does_not_exist'.",
            "expected_finding": "Requested module is unavailable in the environment. Per security rules, it must not be auto-installed.",
            "suggested_fix": "Import an installed module or run pip install <module_name> in your virtual environment."
        },
        {
            "id": "demo_extra_mutable",
            "number": 11,
            "name": "Mutable Default Argument",
            "category": "Data Structure Error",
            "language": "python",
            "code": "def add_student_grade(student_name, grade_list=[]):\n    grade_list.append(student_name)\n    return grade_list\n\nclass_a = add_student_grade(\"Alice\")\nclass_b = add_student_grade(\"Bob\")\nprint(\"Class B:\", class_b)",
            "problem": "Why does class_b contain ['Alice', 'Bob'] instead of ['Bob']?",
            "expected_finding": "Default parameter grade_list=[] is evaluated only once at definition, accumulating state across calls.",
            "suggested_fix": "Use grade_list=None and initialize if grade_list is None: grade_list = [] inside function."
        },
        {
            "id": "demo_extra_js",
            "number": 12,
            "name": "JavaScript Undefined Chain",
            "category": "Runtime Error (JS TypeError)",
            "language": "javascript",
            "code": "function renderUserProfile(apiResponse) {\n    const email = apiResponse.user.profile.email;\n    return `<div class=\"user-card\">${email}</div>`;\n}\n\nrenderUserProfile({ user: { id: \"u99\" } });",
            "problem": "Uncaught TypeError: Cannot read properties of undefined (reading email).",
            "expected_finding": "Direct nested property access crashes when intermediate objects are undefined.",
            "suggested_fix": "Use optional chaining: apiResponse.user?.profile?.email"
        },
        {
            "id": "demo_extra_java",
            "number": 13,
            "name": "Java String Equality",
            "category": "Logic / Reference Error",
            "language": "java",
            "code": "public class AuthService {\n    public boolean check(String status) {\n        return status == \"ACTIVE\";\n    }\n}",
            "problem": "Comparison returns false even when string content is ACTIVE.",
            "expected_finding": "Using == compares memory addresses instead of string text content.",
            "suggested_fix": "Use \"ACTIVE\".equals(status) instead of =="
        }
    ]
    return JsonResponse({"success": True, "demos": demos})


@require_http_methods(["POST"])
def api_verify_code(request):
    """
    Runs Level 4 Safe Code Validation on submitted code without executing it.
    """
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)

    language = data.get("language", "python").strip()
    code = data.get("code", "").strip()

    if not code:
        return JsonResponse({"success": False, "error": "Code cannot be empty."}, status=400)

    from assistant.rag.generator import ContextGroundedGenerator
    generator = ContextGroundedGenerator()
    verification = generator._run_safe_verification(language, code, code, [])
    return JsonResponse({"success": True, "verification": verification})

