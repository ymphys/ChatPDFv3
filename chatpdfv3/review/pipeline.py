from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from ..knowledge_base.builder import load_kb_config
from ..knowledge_base.embedder import Embedder
from ..knowledge_base.vectorstore import VectorStore
from ..services.deepseek_client import create_deepseek_client, post_with_retries_deepseek
from .aggregator import ClusterAggregation, ChunkRecord, aggregate_clusters, build_chunk_records
from .clustering import EmbeddingItem, TopicCluster, cluster_embeddings
from .draft import generate_review_draft
from .outline import generate_outline as llm_generate_outline
from .prompts import TOPIC_SUMMARY_SYSTEM_PROMPT, TOPIC_SUMMARY_USER_PROMPT

logger = logging.getLogger("chatpdf")


class ReviewPipeline:
    """High-level pipeline that orchestrates topic discovery and review writing."""

    def __init__(
        self,
        kb_path: str | Path = "./kb_store",
        *,
        llm_model: str = "deepseek-chat",
        summary_temperature: float = 0.2,
    ) -> None:
        self.kb_path = Path(kb_path)
        self.llm_model = llm_model
        self.summary_temperature = summary_temperature

        self._vector_store = VectorStore(self.kb_path)
        self._kb_config = load_kb_config(self.kb_path)

        embed_model = self._kb_config.get("model_name", "text-embedding-3-large")
        provider = self._kb_config.get("provider")
        self._embedder = Embedder(model_name=embed_model, provider=provider)

        self._client = create_deepseek_client()

        self._chunks: list[ChunkRecord] | None = None
        self._topics: list[TopicCluster] | None = None
        self._aggregations: list[ClusterAggregation] | None = None
        self._topic_summaries: list[dict[str, Any]] | None = None
        self._outline: str | None = None
        self._draft: str | None = None

    # ------------------------------------------------------------------
    # Stage 1: Topic discovery & aggregation
    # ------------------------------------------------------------------
    def discover_topics(self, n_clusters: int) -> list[TopicCluster]:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")

        chunk_records = self._load_chunk_records()
        embedding_items: list[EmbeddingItem] = []
        for chunk in chunk_records:
            if chunk.embedding is None:
                continue
            embedding_vector = (
                chunk.embedding.tolist()
                if hasattr(chunk.embedding, "tolist")
                else [float(x) for x in chunk.embedding]
            )
            if not embedding_vector:
                continue
            embedding_items.append(
                EmbeddingItem(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    embedding=embedding_vector,
                )
            )

        if not embedding_items:
            raise RuntimeError("知识库尚未生成可聚类的向量，请先构建 KB。")

        topics = cluster_embeddings(embedding_items, n_clusters)
        if not topics:
            raise RuntimeError("无法根据现有文献生成主题，请确认知识库内容。")

        self._topics = topics
        self._aggregations = aggregate_clusters(topics, chunk_records)
        logger.info("发现 %s 个候选主题", len(topics))
        return topics

    def summarize_topics(
        self,
        topics: Sequence[TopicCluster] | None = None,
    ) -> list[dict[str, Any]]:
        topics = list(topics or self._topics or [])
        if not topics:
            raise RuntimeError("请先调用 discover_topics 获取主题簇。")

        if not self._aggregations:
            chunk_records = self._load_chunk_records()
            self._aggregations = aggregate_clusters(topics, chunk_records)

        summaries: list[dict[str, Any]] = []
        for aggregation in self._aggregations:
            try:
                summary_text = self._summarize_single_cluster(aggregation)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("总结主题 %s 失败：%s", aggregation.cluster_id, exc)
                summary_text = "未能生成摘要，详见日志。"

            summaries.append(
                {
                    "cluster_id": aggregation.cluster_id,
                    "summary": summary_text,
                    "papers": aggregation.paper_ids,
                    "sources": [
                        {
                            "paper_id": src.paper_id,
                            "chunk_count": src.chunk_count,
                            "source_path": src.source_path,
                        }
                        for src in aggregation.sources
                    ],
                }
            )

        self._topic_summaries = summaries
        return summaries

    # ------------------------------------------------------------------
    # Stage 2: Outline & drafting
    # ------------------------------------------------------------------
    def generate_outline(
        self,
        topic_summaries: Sequence[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.3,
    ) -> str:
        summaries = list(topic_summaries or self._topic_summaries or [])
        if not summaries:
            raise RuntimeError("请先完成主题摘要，然后再生成大纲。")

        outline = llm_generate_outline(
            summaries,
            client=self._client,
            model=self.llm_model,
            temperature=temperature,
        )
        self._outline = outline
        return outline

    def generate_review(
        self,
        outline: str | None = None,
        *,
        temperature: float = 0.4,
    ) -> str:
        outline_text = (outline or self._outline or "").strip()
        if not outline_text:
            raise RuntimeError("请先生成综述大纲，再撰写正文。")

        if not self._aggregations:
            raise RuntimeError("缺少主题聚合结果，请先执行 discover_topics。")

        draft = generate_review_draft(
            outline_text,
            self._aggregations,
            client=self._client,
            model=self.llm_model,
            temperature=temperature,
        )
        self._draft = draft
        return draft

    # ------------------------------------------------------------------
    # Stage 3: Optional refinement via retrieval
    # ------------------------------------------------------------------
    def refine_with_rag(self, draft: str | None = None, *, top_k: int = 3) -> str:
        draft_text = (draft or self._draft or "").strip()
        if not draft_text:
            raise RuntimeError("无可用草稿可供优化。")

        paragraphs = [para.strip() for para in draft_text.split("\n\n") if para.strip()]
        refined_sections: list[str] = []

        for para in paragraphs:
            embedding = self._embedder.embed_text(para)
            chunks = self._vector_store.query(embedding, top_k=top_k)
            if not chunks:
                refined_sections.append(para)
                continue

            references = self._format_references(chunks)
            refined_sections.append(f"{para}\n\n证据支持:\n{references}")

        return "\n\n".join(refined_sections)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_chunk_records(self) -> list[ChunkRecord]:
        if self._chunks is None:
            raw_records = self._vector_store.get_all_documents(include_embeddings=True)
            if not raw_records:
                raise RuntimeError("知识库尚未构建或为空。")
            self._chunks = build_chunk_records(raw_records)
        return self._chunks

    def _summarize_single_cluster(self, aggregation: ClusterAggregation) -> str:
        context = aggregation.combined_text.strip()
        max_chars = 3000
        if len(context) > max_chars:
            context = context[: max_chars - 3] + "..."

        messages = [
            {"role": "system", "content": TOPIC_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": TOPIC_SUMMARY_USER_PROMPT.format(
                    cluster_id=aggregation.cluster_id,
                    paper_ids=", ".join(aggregation.paper_ids),
                    context=context,
                ),
            },
        ]

        response = post_with_retries_deepseek(
            client=self._client,
            model=self.llm_model,
            messages=messages,
            temperature=self.summary_temperature,
        )
        if not response:
            raise RuntimeError(f"LLM 调用失败，无法总结主题 {aggregation.cluster_id}")

        return response.choices[0].message.content.strip()

    @staticmethod
    def _format_references(chunks: Sequence[dict]) -> str:
        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata") or {}
            source = metadata.get("source_name") or metadata.get("source_path") or "unknown"
            preview = " ".join((chunk.get("text") or "").split())
            if len(preview) > 160:
                preview = preview[:157] + "..."
            lines.append(f"- [{idx}] {source}: {preview}")
        return "\n".join(lines)


__all__ = ["ReviewPipeline"]
