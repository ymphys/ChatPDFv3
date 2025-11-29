from __future__ import annotations

import re
from typing import List


def split_markdown(text: str, max_len: int = 800) -> List[str]:
    """
    Split markdown text into chunks whose length does not exceed max_len.
    The splitter keeps paragraph boundaries when possible and falls back to
    fixed-size slicing for very long paragraphs.
    """
    if max_len <= 0:
        raise ValueError("max_len must be a positive integer")
    if not text:
        return []

    # Normalize consecutive blank lines to simplify paragraph logic.
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    paragraphs = re.split(r"\n{2,}", normalized)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            chunk = "\n\n".join(buffer).strip()
            if chunk:
                chunks.append(chunk)
        buffer = []
        buffer_len = 0

    for paragraph in paragraphs:
        para = paragraph.strip()
        if not para:
            continue

        if len(para) > max_len:
            flush_buffer()
            for start in range(0, len(para), max_len):
                piece = para[start : start + max_len].strip()
                if piece:
                    chunks.append(piece)
            continue

        additional_len = len(para) if not buffer else len(para) + 2  # account for separator
        if buffer_len + additional_len <= max_len:
            buffer.append(para)
            buffer_len += additional_len
        else:
            flush_buffer()
            buffer.append(para)
            buffer_len = len(para)

    flush_buffer()
    return chunks


__all__ = ["split_markdown"]
