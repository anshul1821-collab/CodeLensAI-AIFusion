import json
from django.db import models


class Document(models.Model):
    FILE_TYPE_CHOICES = [
        ('pdf', 'PDF Document'),
        ('txt', 'Text File'),
    ]

    title = models.CharField(max_length=255)
    filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/', blank=True, null=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    file_size = models.IntegerField(default=0, help_text="Size in bytes")
    total_pages = models.IntegerField(default=1)
    total_chunks = models.IntegerField(default=0)
    is_sample = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} ({self.file_type.upper()})"

    @property
    def formatted_size(self):
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        return f"{self.file_size / (1024 * 1024):.1f} MB"


class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    chunk_index = models.IntegerField()
    page_number = models.IntegerField(default=1)
    content = models.TextField()
    char_start = models.IntegerField(default=0)
    char_end = models.IntegerField(default=0)
    embedding_json = models.TextField(blank=True, default='[]')
    embedding_model = models.CharField(max_length=100, default='semantic_vector')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['document', 'chunk_index']

    def __str__(self):
        return f"Chunk #{self.chunk_index} of {self.document.title} (p.{self.page_number})"

    def get_embedding(self):
        try:
            return json.loads(self.embedding_json) if self.embedding_json else []
        except Exception:
            return []

    def set_embedding(self, vector):
        self.embedding_json = json.dumps(vector)


class AnalysisRecord(models.Model):
    language = models.CharField(max_length=30, default='python')
    submitted_code = models.TextField()
    problem_description = models.TextField(blank=True, default='')
    
    summary = models.TextField(blank=True, default='')
    code_explanation = models.TextField(blank=True, default='')
    potential_issues_json = models.TextField(blank=True, default='[]')
    suggested_fix = models.TextField(blank=True, default='')
    corrected_code = models.TextField(blank=True, default='')
    retrieved_sources_json = models.TextField(blank=True, default='[]')
    
    is_grounded = models.BooleanField(default=True)
    grounding_message = models.TextField(blank=True, default='')
    model_used = models.CharField(max_length=100, default='CodeLens Engine')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Analysis ({self.language}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    @property
    def issues(self):
        try:
            return json.loads(self.potential_issues_json) if self.potential_issues_json else []
        except Exception:
            return []

    @property
    def sources(self):
        try:
            return json.loads(self.retrieved_sources_json) if self.retrieved_sources_json else []
        except Exception:
            return []


class SystemSetting(models.Model):
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField(blank=True, default='')
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.key}"

    @classmethod
    def get_setting(cls, key, default=''):
        obj = cls.objects.filter(key=key).first()
        return obj.value if obj else default

    @classmethod
    def set_setting(cls, key, value):
        obj, _ = cls.objects.get_or_create(key=key)
        obj.value = value
        obj.save()
        return obj
