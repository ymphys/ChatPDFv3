from __future__ import annotations

from typing import Sequence

from ..services.deepseek_client import post_with_retries_deepseek
from .aggregator import ClusterAggregation
from .prompts import DRAFT_SYSTEM_PROMPT, DRAFT_USER_PROMPT


def generate_review_draft(
    outline: str,
    cluster_contexts: Sequence[ClusterAggregation],
    *,
    client,
    model: str,
    temperature: float = 0.4,
) -> str:
    """Create a review draft that strictly relies on KB-derived contexts."""

    if not outline.strip():
        raise ValueError("outline cannot be empty")
    if not cluster_contexts:
        raise ValueError("cluster_contexts must not be empty")

    cluster_notes = _format_cluster_notes(cluster_contexts)
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DRAFT_USER_PROMPT.format(
                outline=outline.strip(),
                cluster_notes=cluster_notes,
            ),
        },
    ]

    response = post_with_retries_deepseek(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
    )
    if not response:
        raise RuntimeError("LLM call failed while generating draft")

    return response.choices[0].message.content.strip()


def _format_cluster_notes(cluster_contexts: Sequence[ClusterAggregation]) -> str:
    blocks: list[str] = []
    for cluster in cluster_contexts:
        sources = ", ".join(f"{src.paper_id}×{src.chunk_count}" for src in cluster.sources)
        context = cluster.combined_text.strip()
        max_chars = 1800
        if len(context) > max_chars:
            context = context[: max_chars - 3] + "..."
        blocks.append(
            f"[主题 {cluster.cluster_id}] 来源：{sources or '未知'}\n{context}"
        )
    return "\n\n".join(blocks)


__all__ = ["generate_review_draft"]
