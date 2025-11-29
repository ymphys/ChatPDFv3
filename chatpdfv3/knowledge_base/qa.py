from __future__ import annotations

import logging
from pathlib import Path

from ..services.deepseek_client import create_deepseek_client, post_with_retries_deepseek
from .retriever import retrieve

logger = logging.getLogger("chatpdf")

DEFAULT_QA_MODEL = "deepseek-chat"


def answer_query(query: str, kb_path: str | Path = "./kb_store", top_k: int = 5) -> str:
    """
    Use retrieved context from the knowledge base to answer a user query via the
    existing DeepSeek LLM client.
    """
    chunks = retrieve(query, kb_path=str(kb_path), top_k=top_k)
    if not chunks:
        return "知识库为空或未检索到相关内容，请先构建知识库。"

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source_name") or metadata.get("source_path", "unknown")
        tag = f"[{idx}] 来源：{source} (片段 {metadata.get('chunk_index', 0) + 1}/{metadata.get('chunk_total', '?')})"
        context_blocks.append(f"{tag}\n{chunk.get('text', '')}".strip())

    payload = "\n\n".join(context_blocks)
    client = create_deepseek_client()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名学术助手，请基于检索到的片段回答用户问题；"
                "若片段未提供答案，请明确说明。引用信息时注明片段编号。"
            ),
        },
        {
            "role": "user",
            "content": f"已检索到以下内容：\n\n{payload}\n\n问题：{query}",
        },
    ]

    response = post_with_retries_deepseek(
        client=client,
        model=DEFAULT_QA_MODEL,
        messages=messages,
        temperature=0.3,
    )
    if not response:
        return "LLM 调用失败，无法生成回答。"

    answer = response.choices[0].message.content.strip()
    references = _format_reference_section(chunks)
    if references:
        answer = f"{answer}\n\n参考片段：\n{references}"
    return answer


def _format_reference_section(chunks: list[dict]) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source_path = metadata.get("source_path")
        source = metadata.get("source_name") or (Path(source_path).name if source_path else "unknown")
        chunk_index = metadata.get("chunk_index", 0) + 1
        chunk_total = metadata.get("chunk_total", "?")
        preview = _first_sentence(chunk.get("text", ""))
        location = source_path or source
        lines.append(
            f"[{idx}] {source} 第 {chunk_index}/{chunk_total} 段 "
            f"({location})：{preview}"
        )
    return "\n".join(lines)


def _first_sentence(text: str, *, limit: int = 80) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


__all__ = ["answer_query"]
