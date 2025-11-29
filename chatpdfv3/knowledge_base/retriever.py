from __future__ import annotations

import logging
from typing import List

from .builder import load_kb_config
from .embedder import Embedder
from .vectorstore import VectorStore

logger = logging.getLogger("chatpdf")


def retrieve(query: str, kb_path: str | None = "./kb_store", top_k: int = 5) -> List[dict]:
    """
    Embed the query, perform vector search, and return the top_k chunks.
    """
    if not query.strip():
        return []

    config = load_kb_config(kb_path or "./kb_store")
    model_name = config.get("model_name", "text-embedding-3-large")
    provider = config.get("provider")

    embedder = Embedder(model_name=model_name, provider=provider)
    vector_store = VectorStore(kb_path or "./kb_store")
    query_embedding = embedder.embed_text(query)
    if not query_embedding:
        logger.warning("查询内容为空，无法进行检索")
        return []

    return vector_store.query(query_embedding, top_k=top_k)


__all__ = ["retrieve"]
