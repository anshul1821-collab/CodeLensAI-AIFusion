import os
import io
from pathlib import Path
from typing import Dict, List, Any
import pypdf

ALLOWED_EXTENSIONS = {'.pdf', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class DocumentExtractionError(Exception):
    """Custom exception raised when document extraction or validation fails."""
    pass


def validate_file(filename: str, file_size: int):
    """Validates file extension and size constraints."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentExtractionError(
            f"Unsupported file format '{ext}'. Only PDF (.pdf) and Text (.txt) files are accepted."
        )

    if file_size <= 0:
        raise DocumentExtractionError(f"Uploaded file '{filename}' is empty (0 bytes).")

    if file_size > MAX_FILE_SIZE:
        raise DocumentExtractionError(
            f"File size exceeds maximum limit of {MAX_FILE_SIZE // (1024 * 1024)}MB. Provided: {file_size / (1024*1024):.2f}MB."
        )


def extract_text_from_pdf(file_source) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file using pypdf.
    Preserves page numbers (1-indexed) and character metrics.
    """
    pages_data = []
    try:
        if isinstance(file_source, (str, Path)):
            reader = pypdf.PdfReader(str(file_source))
        else:
            if hasattr(file_source, 'read'):
                content = file_source.read()
                reader = pypdf.PdfReader(io.BytesIO(content))
            else:
                reader = pypdf.PdfReader(io.BytesIO(file_source))

        if len(reader.pages) == 0:
            raise DocumentExtractionError("The PDF document does not contain any pages.")

        for idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            # Strip control characters except standard whitespace
            cleaned_text = page_text.strip()
            if cleaned_text:
                pages_data.append({
                    "page_number": idx,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                })

        if not pages_data:
            raise DocumentExtractionError(
                "No readable text could be extracted from this PDF. It may be scanned, image-only, or encrypted."
            )

        return pages_data

    except DocumentExtractionError:
        raise
    except Exception as e:
        raise DocumentExtractionError(f"Failed to parse PDF document: {str(e)}")


def extract_text_from_txt(file_source) -> List[Dict[str, Any]]:
    """
    Extracts text from a plain text document.
    Attempts UTF-8 decoding, falling back to Latin-1/CP1252.
    Groups lines into logical sections or pages.
    """
    raw_bytes = b""
    try:
        if isinstance(file_source, (str, Path)):
            with open(file_source, 'rb') as f:
                raw_bytes = f.read()
        elif hasattr(file_source, 'read'):
            raw_bytes = file_source.read()
        else:
            raw_bytes = file_source

        # Decode with fallback
        text = None
        for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise DocumentExtractionError("Could not decode text file with standard encodings.")

        text = text.strip()
        if not text:
            raise DocumentExtractionError("Text file is empty or contains only whitespace.")

        # For text documents, treat 100 lines or ~2500 chars as a 'page' to retain line/section references
        lines = text.splitlines()
        pages_data = []
        page_size_lines = 80
        total_pages = max(1, (len(lines) + page_size_lines - 1) // page_size_lines)

        for p_idx in range(total_pages):
            start_l = p_idx * page_size_lines
            end_l = min((p_idx + 1) * page_size_lines, len(lines))
            page_content = "\n".join(lines[start_l:end_l]).strip()
            if page_content:
                pages_data.append({
                    "page_number": p_idx + 1,
                    "text": page_content,
                    "char_count": len(page_content),
                    "line_range": f"L{start_l + 1}-L{end_l}"
                })

        return pages_data

    except DocumentExtractionError:
        raise
    except Exception as e:
        raise DocumentExtractionError(f"Failed to process text file: {str(e)}")


def extract_document(file_source, filename: str, file_size: int) -> Dict[str, Any]:
    """
    Main extraction interface. Validates constraints and extracts structured page data.
    """
    validate_file(filename, file_size)
    ext = Path(filename).suffix.lower()
    
    if ext == '.pdf':
        pages = extract_text_from_pdf(file_source)
        file_type = 'pdf'
    elif ext == '.txt':
        pages = extract_text_from_txt(file_source)
        file_type = 'txt'
    else:
        raise DocumentExtractionError(f"Unsupported extension: {ext}")

    full_text = "\n\n".join([p["text"] for p in pages])
    clean_title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()

    return {
        "title": clean_title,
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "total_pages": len(pages),
        "pages": pages,
        "full_text": full_text
    }
