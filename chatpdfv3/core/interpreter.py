from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from ..services.deepseek_client import create_deepseek_client, post_with_retries_deepseek
from ..utils import load_existing_answers, split_into_chunks

logger = logging.getLogger("chatpdf")



def deepseek_interpretation(
    md_content: Optional[dict],
    questions: Sequence[str],
    output_path: Path,
    *,
    model: str = "deepseek-chat",
    chunk_pause_seconds: int = 1,
    temperature: float = 1.0,
) -> str:
    """
    Use the DeepSeek API to interpret markdown content.
    """
    if not md_content:
        logger.info("No content to interpret")
        return ""

    # 创建 DeepSeek 客户端
    client = create_deepseek_client()

    existing_answers = load_existing_answers(output_path)
    new_sections: list[str] = []

    for question in questions:
        if question in existing_answers:
            logger.info(
                "Skipping interpretation for question (already present): %s", question
            )
            continue

        try:
            context = _format_existing_context(existing_answers)
            chunk_answers = _interpret_chunks_deepseek(
                md_content["content"],
                question=question,
                client=client,
                model=model,
                pause_seconds=chunk_pause_seconds,
                context=context,
                temperature=temperature,
            )

            if len(chunk_answers) == 1:
                final_answer = chunk_answers[0]
            else:
                final_answer = _synthesise_answer_deepseek(
                    chunk_answers,
                    question=question,
                    client=client,
                    model=model,
                    context=context,
                    temperature=0.0,  # 合成答案时使用更低的 temperature
                )

            new_sections.append(f"## {question}\n\n{final_answer}\n\n")
            existing_answers[question] = final_answer
        except Exception as exc:
            logger.exception("Error processing question '%s': %s", question, exc)
            new_sections.append(f"## {question}\n\n处理此问题时发生错误。\n\n")

    result = "".join(new_sections)
    if result:
        _append_sections(output_path, result)
    else:
        logger.info("No new interpretation sections to write (all questions handled).")

    return result




def _interpret_chunks_deepseek(
    content: str,
    *,
    question: str,
    client: Any,
    model: str,
    pause_seconds: int,
    context: str,
    temperature: float = 1.0,
) -> list[str]:
    """
    Ask the DeepSeek model the same question across chunked document segments.
    """
    chunk_answers: list[str] = []
    chunks = split_into_chunks(content)
    for idx, chunk in enumerate(chunks, start=1):
        user_sections = []
        if context:
            user_sections.append(
                "以下是之前的问题与回答，可作为上下文：\n\n" + context
            )
        user_sections.append(
            f"文档片段 {idx}/{len(chunks)}：\n\n{chunk}"
        )
        user_sections.append(f"问题：{question}")

        messages = [
            {
                "role": "system",
                "content": "你是一个学术文献分析专家，请基于提供的文档内容回答问题，请注意对专业名词做出解释。",
            },
            {
                "role": "user",
                "content": "\n\n".join(user_sections),
            },
        ]

        response = post_with_retries_deepseek(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
        )

        if response is None:
            chunk_answers.append("[请求失败，未获得该片段回答]")
            logger.warning(
                "No response for chunk %s/%s for question: %s",
                idx,
                len(chunks),
                question,
            )
        else:
            text = response.choices[0].message.content.strip()
            chunk_answers.append(text)
            logger.info("Chunk %s/%s answered for question: %s", idx, len(chunks), question)
            logger.debug("Chunk %s preview: %s", idx, text[:120].replace("\n", " "))

        time.sleep(pause_seconds)
    return chunk_answers


def _synthesise_answer_deepseek(
    chunk_answers: Iterable[str],
    *,
    question: str,
    client: Any,
    model: str,
    context: str,
    temperature: float = 0.0,
) -> str:
    """
    Reconcile multiple chunk answers into a single, coherent response using DeepSeek.
    """
    context_block = (
        "以下是之前的问题与回答，可作为上下文：\n\n"
        + context
        + "\n\n"
        if context
        else ""
    )

    synth_prompt = (
        context_block
        + "请基于下面各片段回答，综合出一个简洁、连贯且基于文档的最终回答；"
        + "若文档未提供信息请明确说明。\n\n"
        + "\n\n---\n\n".join(chunk_answers)
    )
    
    messages = [
        {"role": "system", "content": "你负责把分片回答合并成最终答案。"},
        {"role": "user", "content": f"{synth_prompt}\n\n问题：{question}"},
    ]

    response = post_with_retries_deepseek(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
    )
    
    if response:
        final_answer = response.choices[0].message.content.strip()
        logger.info("Synthesized final answer for question: %s", question)
        return final_answer

    logger.error("Failed to synthesize answer for question '%s'", question)
    return "无法获取答案，API调用失败。"


def _append_sections(output_path: Path, new_sections: str) -> None:
    """
    Append newly generated sections to the target markdown file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            output_path.write_text("# 文档解读\n\n" + new_sections, encoding="utf-8")
        else:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(new_sections)
        logger.info("Interpretation answers appended to %s", output_path)
    except Exception as exc:
        logger.error("Error saving interpretation: %s", exc)


def _format_existing_context(existing_answers: Dict[str, str]) -> str:
    """
    Turn existing question-answer pairs into a markdown block used as context.
    """
    if not existing_answers:
        return ""

    sections: list[str] = []
    for question, answer in existing_answers.items():
        if not question or not answer:
            continue
        sections.append(f"### {question}\n{answer}")
    return "\n\n".join(sections)


__all__ = ["deepseek_interpretation"]
