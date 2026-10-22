import os
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from assistant.models import Document, DocumentChunk
from assistant.rag.pipeline import RAGPipeline
from assistant.rag.extractor import extract_document

class Command(BaseCommand):
    help = "Seeds and indexes the sample documentation reference pack into the knowledge base."

    def handle(self, *args, **options):
        sample_dir = settings.BASE_DIR / 'sample_docs'
        if not sample_dir.exists():
            self.stdout.write(self.style.ERROR(f"sample_docs directory not found at {sample_dir}"))
            return

        pipeline = RAGPipeline()
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.pdf', '.txt'))]

        if not sample_files:
            self.stdout.write(self.style.WARNING("No sample files found in sample_docs/"))
            return

        total_new = 0
        for fname in sample_files:
            fpath = sample_dir / fname
            title = fname.replace('_', ' ').replace('-', ' ').title()
            title = title.replace('.Pdf', '').replace('.Txt', '')

            # Check if document already exists with chunks
            existing_doc = Document.objects.filter(title=title).first()
            if existing_doc and existing_doc.total_chunks > 0:
                self.stdout.write(f"Skipping existing document: {title}")
                continue
            elif existing_doc:
                existing_doc.delete()

            try:
                with open(fpath, 'rb') as f:
                    content_bytes = f.read()

                doc = pipeline.ingest_document(
                    file_source=content_bytes,
                    filename=fname,
                    file_size=len(content_bytes),
                    is_sample=True
                )
                total_new += 1
                self.stdout.write(self.style.SUCCESS(f"Successfully indexed: {doc.title} ({doc.total_chunks} chunks)"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Failed to index {fname}: {e}"))


        self.stdout.write(self.style.SUCCESS(f"Done! Knowledge Base has {Document.objects.count()} docs and {DocumentChunk.objects.count()} chunks."))
