"""Local sentence-transformers wrapper.

Single module-level model. Lazy-loaded on first call to avoid a 1.5s import-time
hit (and to keep test discovery snappy when tests stub embeddings entirely).
"""
from __future__ import annotations

from threading import Lock
from typing import Sequence

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_lock = Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_one(text: str) -> list[float]:
    vec = _get_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_many(texts: Sequence[str]) -> list[list[float]]:
    vecs = _get_model().encode(list(texts), normalize_embeddings=True)
    return [v.tolist() for v in vecs]
