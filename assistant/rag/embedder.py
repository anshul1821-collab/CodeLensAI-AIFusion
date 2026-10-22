import os
import math
import hashlib
import json
import re
from typing import List, Dict, Any, Optional
import requests
from django.conf import settings

LOCAL_VECTOR_DIM = 512
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
FALLBACK_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"


def l2_normalize(vector: List[float]) -> List[float]:
    """Applies L2 normalization to ensure dot product equals cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0] * len(vector)
    return [round(x / norm, 6) for x in vector]


class BaseEmbedder:
    model_name: str = "base"
    dimension: int = 512

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class LocalSemanticEmbedder(BaseEmbedder):
    """
    High-dimensional deterministic semantic feature vector embedder in pure Python.
    Constructs a 512-dimensional normalized vector using subword n-grams,
    programming domain keyword weights, and hashed token representations.
    Zero external dependencies, 100% reproducible and verifiable offline.
    """
    model_name = "codelens-semantic-512"
    dimension = LOCAL_VECTOR_DIM

    # High-relevance programming concept weights
    KEYWORD_WEIGHTS: Dict[str, float] = {
        "mutable": 4.5, "immutable": 4.5, "default": 4.0, "argument": 3.8, "parameter": 3.8,
        "list": 3.2, "dict": 3.2, "dictionary": 3.2, "tuple": 3.2, "append": 3.5, "extend": 3.2,
        "function": 3.0, "def": 3.0, "definition": 3.0, "evaluate": 3.5, "evaluated": 3.5,
        "persistent": 3.5, "shared": 3.5, "state": 2.8, "reference": 3.5, "binding": 3.2,
        "exception": 4.0, "error": 3.5, "indexerror": 4.8, "keyerror": 4.8, "typeerror": 4.8,
        "zerodivisionerror": 4.8, "attributeerror": 4.8, "valueerror": 4.8, "syntaxerror": 4.8,
        "raise": 3.0, "try": 3.2, "except": 3.5, "finally": 3.2, "catch": 3.2, "throw": 3.2,
        "index": 3.2, "range": 3.0, "bounds": 3.5, "loop": 2.8, "for": 2.2, "while": 2.2,
        "none": 3.5, "null": 3.5, "undefined": 3.5, "pointer": 3.2, "leak": 3.8,
        "copy": 3.5, "deepcopy": 4.0, "shallow": 3.5, "slice": 3.0, "id": 2.8,
        "call": 2.5, "caller": 2.5, "return": 2.5, "yield": 3.0, "class": 2.8,
        "object": 2.5, "instance": 2.8, "method": 2.8, "variable": 2.5, "scope": 3.2,
        "global": 3.0, "nonlocal": 3.2, "closure": 3.5, "memory": 3.0, "garbage": 3.0
    }

    # Concept synsets: maps synonyms to shared semantic concept dimensions
    CONCEPT_SYNSETS: List[tuple[str, List[str], float]] = [
        ("topic_parameters", ["default", "defaults", "argument", "arguments", "parameter", "parameters", "param", "params", "evaluated"], 3.5),
        ("topic_mutability", ["mutable", "mutability", "list", "lists", "dict", "dictionary", "append", "extend", "mutate", "in_place"], 3.5),
        ("topic_functions", ["function", "functions", "def", "definition", "callable", "caller", "call", "invoke"], 2.5),
        ("topic_exceptions", ["exception", "exceptions", "error", "errors", "raise", "raised", "try", "except", "crash"], 3.5),
        ("topic_indexing", ["index", "indexerror", "range", "subscript", "bounds", "sequence", "len", "slice"], 3.5),
        ("topic_memory", ["reference", "binding", "identity", "id", "shallow", "deep", "copy", "deepcopy", "shared", "state"], 3.0),
        ("topic_math_division", ["zerodivision", "zerodivisionerror", "division", "zero", "divide", "modulo", "by_zero"], 4.0),
        ("topic_mapping_keys", ["keyerror", "mapping", "key", "keys", "lookup", "dict", "get"], 3.5)
    ]

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'[a-zA-Z0-9_]+', text.lower())
        return tokens

    def _hash_token(self, token: str, seed: int = 0) -> int:
        data = f"{seed}:{token}".encode('utf-8')
        return int(hashlib.md5(data).hexdigest(), 16)

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        # 1. Unigram and weighted keyword features
        for token in tokens:
            weight = self.KEYWORD_WEIGHTS.get(token, 1.2)
            bucket = self._hash_token(token, 42) % self.dimension
            vec[bucket] += weight

        # 2. Bigrams for phrases (e.g. "default argument", "index error")
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bucket = self._hash_token(bigram, 101) % self.dimension
            weight = 2.0
            if tokens[i] in self.KEYWORD_WEIGHTS or tokens[i+1] in self.KEYWORD_WEIGHTS:
                weight = 3.5
            vec[bucket] += weight

        # 3. Semantic Concept Synsets (bridges synonyms like argument <-> parameter)
        token_set = set(tokens)
        for concept_name, synset_words, concept_weight in self.CONCEPT_SYNSETS:
            matching_count = sum(1 for w in synset_words if w in token_set)
            if matching_count > 0:
                bucket = self._hash_token(concept_name, 2026) % self.dimension
                vec[bucket] += concept_weight * min(3.0, 1.0 + (matching_count - 1) * 0.5)

        # 4. Subword character 3-grams & 4-grams (handles typos and morphology)
        clean_text = " ".join(tokens)
        for n in (3, 4):
            for i in range(len(clean_text) - n + 1):
                ngram = clean_text[i:i+n]
                bucket = self._hash_token(ngram, n * 17) % self.dimension
                vec[bucket] += 0.35

        return l2_normalize(vec)


class GoogleGeminiEmbedder(BaseEmbedder):
    """
    Google Gemini embedding integration via direct REST API.
    Zero C++ build dependencies, high semantic retrieval precision.
    """
    model_name = "models/gemini-embedding-001"
    dimension = 768

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("Gemini API key cannot be empty.")

    def embed_text(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return [0.0] * self.dimension

        url = f"{GEMINI_EMBED_URL}?key={self.api_key}"
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [{"text": text[:2000]}]
            },
            "outputDimensionality": 768
        }
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=12)
            if resp.status_code != 200:
                # Try fallback model: models/gemini-embedding-2
                fallback_url = f"{FALLBACK_EMBED_URL}?key={self.api_key}"
                fallback_payload = {
                    "model": "models/gemini-embedding-2",
                    "content": {"parts": [{"text": text[:2000]}]},
                    "outputDimensionality": 768
                }
                resp = requests.post(fallback_url, headers=headers, json=fallback_payload, timeout=12)

            if resp.status_code == 200:
                result = resp.json()
                embedding_vals = result.get("embedding", {}).get("values", [])
                if embedding_vals:
                    return l2_normalize(embedding_vals)

            # If Gemini returned non-200, gracefully fall back to local semantic vector
            print(f"[CodeLens Warning] Gemini embedding returned status {resp.status_code}. Falling back to LocalSemanticEmbedder.")
            return LocalSemanticEmbedder().embed_text(text)

        except Exception as e:
            # Fall back to offline deterministic semantic vector
            print(f"[CodeLens Warning] Gemini embedding network error: {e}. Falling back to LocalSemanticEmbedder.")
            return LocalSemanticEmbedder().embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # For batch, sequentially
        results = []
        for text in texts:
            results.append(self.embed_text(text))
        return results


def get_active_embedder(force_provider: Optional[str] = None) -> BaseEmbedder:
    """
    Resolves the active embedder.
    If GEMINI_API_KEY is configured in env or settings, and not forced to local, uses Gemini.
    Otherwise uses LocalSemanticEmbedder.
    """
    from assistant.models import SystemSetting
    
    if force_provider == 'local':
        return LocalSemanticEmbedder()

    # Check database setting first, then environment
    gemini_key = SystemSetting.get_setting("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    embedder_preference = SystemSetting.get_setting("EMBEDDER_PREFERENCE", "auto").strip().lower()

    if force_provider == 'gemini' or (embedder_preference != 'local' and gemini_key):
        try:
            return GoogleGeminiEmbedder(api_key=gemini_key)
        except Exception:
            return LocalSemanticEmbedder()

    return LocalSemanticEmbedder()
