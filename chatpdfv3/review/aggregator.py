from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .clustering import TopicCluster


@dataclass(frozen=True)
class ChunkRecord:
    """Representation of a stored chunk inside the knowledge base."""

    chunk_id: str
    paper_id: str
    text: str
    metadata: dict
    embedding: List[float] | None = None


@dataclass
class SourceStats:
    paper_id: str
    chunk_count: int
    source_path: str | None = None


@dataclass
class ClusterAggregation:
    cluster_id: int
    paper_ids: list[str]
    centroid: List[float] | None
    combined_text: str
    chunks: list[ChunkRecord]
    sources: list[SourceStats]


def build_chunk_records(records: Iterable[dict]) -> list[ChunkRecord]:
    """Convert raw VectorStore rows into rich chunk records."""

    payload: list[ChunkRecord] = []
    for idx, row in enumerate(records):
        metadata = row.get("metadata") or {}
        chunk_id = row.get("id") or metadata.get("chunk_id") or f"chunk_{idx}"
        paper_id = metadata.get("source_name") or metadata.get("source_path") or chunk_id
        embedding = row.get("embedding")
        if embedding is not None:
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            else:
                embedding = [float(x) for x in embedding]

        payload.append(
            ChunkRecord(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=row.get("text") or row.get("document") or "",
                metadata=metadata,
                embedding=embedding,
            )
        )
    return payload


def aggregate_clusters(
    topics: Sequence[TopicCluster],
    chunks: Sequence[ChunkRecord],
) -> list[ClusterAggregation]:
    """Group original chunks by cluster so LLM prompts can reference them."""

    if not topics:
        return []

    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    aggregations: list[ClusterAggregation] = []

    for topic in topics:
        members: list[ChunkRecord] = []
        for chunk_id in topic.chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk:
                members.append(chunk)

        if not members:
            continue

        combined_text = "\n\n".join(chunk.text for chunk in members if chunk.text)
        paper_ids = topic.paper_ids or sorted({chunk.paper_id for chunk in members})

        counter = Counter(chunk.paper_id for chunk in members)
        source_paths = {chunk.paper_id: chunk.metadata.get("source_path") for chunk in members}
        sources = [
            SourceStats(paper_id=pid, chunk_count=count, source_path=source_paths.get(pid))
            for pid, count in sorted(counter.items())
        ]

        aggregations.append(
            ClusterAggregation(
                cluster_id=topic.cluster_id,
                paper_ids=paper_ids,
                centroid=topic.centroid,
                combined_text=combined_text,
                chunks=members,
                sources=sources,
            )
        )

    return aggregations


__all__ = [
    "ChunkRecord",
    "ClusterAggregation",
    "SourceStats",
    "build_chunk_records",
    "aggregate_clusters",
]
