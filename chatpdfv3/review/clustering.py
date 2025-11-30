from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sklearn.cluster import KMeans


@dataclass(frozen=True)
class EmbeddingItem:
    chunk_id: str
    paper_id: str
    embedding: List[float]


@dataclass
class TopicCluster:
    cluster_id: int
    paper_ids: list[str]
    centroid: List[float]
    chunk_ids: list[str]


def cluster_embeddings(
    embs: Sequence[EmbeddingItem | Sequence[float] | dict],
    n_clusters: int,
) -> list[TopicCluster]:
    """Cluster chunk embeddings via KMeans and emit topic descriptors."""

    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")

    vectors: list[list[float]] = []
    chunk_ids: list[str] = []
    paper_ids: list[str] = []

    for idx, item in enumerate(embs):
        vector: list[float] | None = None
        chunk_id = f"chunk_{idx}"
        paper_id = chunk_id

        if isinstance(item, EmbeddingItem):
            vector = item.embedding
            chunk_id = item.chunk_id or chunk_id
            paper_id = item.paper_id or paper_id
        elif isinstance(item, dict):
            vector = item.get("embedding")
            chunk_id = (
                item.get("chunk_id")
                or item.get("id")
                or item.get("chunkId")
                or chunk_id
            )
            paper_id = (
                item.get("paper_id")
                or item.get("paperId")
                or item.get("source_name")
                or item.get("source_path")
                or chunk_id
            )
        else:
            vector = [float(x) for x in item]  # type: ignore[arg-type]

        if not vector:
            continue

        vectors.append([float(x) for x in vector])
        chunk_ids.append(chunk_id)
        paper_ids.append(paper_id)

    if not vectors:
        return []

    n_clusters = min(n_clusters, len(vectors))
    if n_clusters == 0:
        return []

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(vectors)
    centroids = model.cluster_centers_.tolist()

    clusters: list[TopicCluster] = []
    for cluster_id, centroid in enumerate(centroids):
        member_indices = [idx for idx, label in enumerate(labels) if label == cluster_id]
        members_chunk_ids = [chunk_ids[i] for i in member_indices]
        members_paper_ids = sorted({paper_ids[i] for i in member_indices})

        clusters.append(
            TopicCluster(
                cluster_id=cluster_id,
                paper_ids=members_paper_ids,
                centroid=[float(x) for x in centroid],
                chunk_ids=members_chunk_ids,
            )
        )

    return clusters


__all__ = ["EmbeddingItem", "TopicCluster", "cluster_embeddings"]
