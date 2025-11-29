"""
Knowledge base module that powers PDF embedding, storage, and retrieval.
"""

from .builder import build_knowledge_base, load_kb_config  # noqa: F401
from .chunker import split_markdown  # noqa: F401
from .embedder import Embedder  # noqa: F401
from .qa import answer_query  # noqa: F401
from .retriever import retrieve  # noqa: F401
from .vectorstore import VectorStore  # noqa: F401

__all__ = [
    "Embedder",
    "VectorStore",
    "split_markdown",
    "build_knowledge_base",
    "retrieve",
    "answer_query",
    "load_kb_config",
]
