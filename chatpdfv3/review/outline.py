from __future__ import annotations

from typing import Any, Sequence

from ..services.deepseek_client import post_with_retries_deepseek
from .prompts import OUTLINE_SYSTEM_PROMPT, OUTLINE_USER_PROMPT


def generate_outline(
    topic_summaries: Sequence[dict[str, Any]],
    *,
    client,
    model: str,
    temperature: float = 0.3,
) -> str:
    """Call the LLM to create a structured outline from topic summaries."""

    if not topic_summaries:
        raise ValueError("topic_summaries must not be empty")

    formatted = _format_topic_summaries(topic_summaries)
    messages = [
        {"role": "system", "content": OUTLINE_SYSTEM_PROMPT},
        {"role": "user", "content": OUTLINE_USER_PROMPT.format(topic_summaries=formatted)},
    ]

    response = post_with_retries_deepseek(client=client, model=model, messages=messages, temperature=temperature)
    if not response:
        raise RuntimeError("LLM call failed while generating outline")

    return response.choices[0].message.content.strip()


def _format_topic_summaries(topic_summaries: Sequence[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for summary in topic_summaries:
        cluster_id = summary.get("cluster_id")
        summary_text = summary.get("summary", "")
        papers = ", ".join(summary.get("papers", []))
        blocks.append(f"主题 {cluster_id} ({papers})\n{summary_text}")
    return "\n\n".join(blocks)


__all__ = ["generate_outline"]
