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
