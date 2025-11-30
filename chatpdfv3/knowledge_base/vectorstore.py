from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("chatpdf")

try:  # pragma: no cover - import guarded for helpful error messages
    import chromadb  # type: ignore
except ImportError as exc:  # pragma: no cover
    chromadb = None
    _chromadb_import_error = exc
else:
    _chromadb_import_error = None


class VectorStore:
    """
    Minimal wrapper around ChromaDB for storing and querying chunk embeddings.
    """

    def __init__(
        self,
        persist_dir: str | Path = "./kb_store",
        *,
        collection_name: str = "chatpdf_documents",
    ) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is required for the knowledge base. "
                "Install it with `pip install chromadb`."
            ) from _chromadb_import_error

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict],
    ) -> None:
        if not chunks:
            return
        if not (len(chunks) == len(embeddings) == len(metadata)):
            raise ValueError("chunks, embeddings, and metadata lists must have the same length")

        ids = []
        for meta in metadata:
            chunk_id = meta.get("chunk_id") or uuid.uuid4().hex
            meta["chunk_id"] = chunk_id
            ids.append(chunk_id)

        self._collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids,
        )

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if not query_embedding:
            return []
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        payload: list[dict] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            payload.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "score": distance,
                }
            )
        return payload

    def get_all_documents(self, *, include_embeddings: bool = False) -> List[Dict]:
        """Return every chunk stored in the collection.

        This is primarily useful for offline processing steps such as
        clustering, where we need the raw embeddings and metadata rather than
        similarity search results.
        """
        if self._collection.count() == 0:
            return []

        include = ["metadatas", "documents"]
        if include_embeddings:
            include.append("embeddings")

        records = self._collection.get(include=include)
        documents = records.get("documents", []) or []
        metadatas = records.get("metadatas", []) or []
        ids = records.get("ids", []) or []
        embeddings = records.get("embeddings") if include_embeddings else None

        payload: list[dict] = []
        for idx, chunk_id in enumerate(ids):
            embedding_value = None
            if embeddings is not None and idx < len(embeddings):
                embedding_value = embeddings[idx]

            payload.append(
                {
                    "id": chunk_id,
                    "text": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "embedding": embedding_value,
                }
            )

        return payload

    def stats(self) -> Dict[str, Optional[int | str]]:
        return {
            "persist_dir": str(self.persist_dir),
            "collection": self.collection_name,
            "chunks": self._collection.count(),
        }


__all__ = ["VectorStore"]
