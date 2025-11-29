from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from ..config import Settings, get_settings
from ..services import process_local_files_via_mineru, process_urls_via_mineru
from ..utils import read_md_content
from .chunker import split_markdown
from .embedder import Embedder
from .vectorstore import VectorStore

logger = logging.getLogger("chatpdf")

CONFIG_FILENAME = "kb_config.json"


def build_knowledge_base(
    pdf_folder: str,
    kb_path: str | Path = "./kb_store",
    *,
    chunk_size: int = 800,
    model_name: str = "text-embedding-3-large",
    provider: str | None = None,
    mineru_model: str = "vlm",
    mineru_timeout: int = 900,
) -> dict:
    """
    Build or update the knowledge base by parsing PDFs, chunking markdown, and
    persisting embeddings in ChromaDB.
    """
    settings = get_settings()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    kb_dir = Path(kb_path)
    kb_dir.mkdir(parents=True, exist_ok=True)
    sources_root = kb_dir / "sources"
    sources_root.mkdir(parents=True, exist_ok=True)

    existing_config = load_kb_config(kb_dir)
    if existing_config and existing_config.get("model_name") != model_name:
        raise ValueError(
            "知识库已存在并使用不同的 embedding 模型。"
            " 为避免混用向量，请提供相同的模型名称，或清空 kb_path 后重新构建。"
        )

    embedder = Embedder(model_name=model_name, provider=provider)
    vector_store = VectorStore(kb_dir)

    markdown_paths = list(
        _collect_markdown_sources(
            pdf_folder,
            settings,
            mineru_model,
            mineru_timeout,
            output_root=sources_root,
        )
    )
    unique_paths: list[Path] = []
    seen_paths: set[str] = set()
    for md_path in markdown_paths:
        resolved = str(md_path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        unique_paths.append(md_path)

    markdown_paths = unique_paths
    if not markdown_paths:
        raise FileNotFoundError("未找到可用的 PDF 或 markdown 文件用于构建知识库")

    stats = {
        "documents_processed": 0,
        "chunks_added": 0,
        "kb_path": str(kb_dir),
    }
    processed_sources: set[str] = set(existing_config.get("sources", []))

    for md_path in markdown_paths:
        md_data = read_md_content(md_path)
        if not md_data:
            logger.warning("跳过无法读取的 markdown：%s", md_path)
            continue
        chunks = split_markdown(md_data["content"], max_len=chunk_size)
        if not chunks:
            continue

        metadata_base = {
            "source_path": str(md_path),
            "source_name": md_path.name,
            "chunk_size": chunk_size,
            "model_name": embedder.model_name,
        }
        _persist_chunks(chunks, metadata_base, vector_store, embedder)

        stats["documents_processed"] += 1
        stats["chunks_added"] += len(chunks)
        processed_sources.add(str(md_path))
        logger.info("知识库已收录 %s 的 %s 个切片", md_path.name, len(chunks))

    _save_kb_config(
        kb_dir,
        {
            "model_name": embedder.model_name,
            "provider": embedder.provider,
            "chunk_size": chunk_size,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "sources": sorted(processed_sources),
            "documents_processed": len(processed_sources),
        },
    )

    return stats


def _collect_markdown_sources(
    pdf_folder: str,
    settings: Settings,
    mineru_model: str,
    mineru_timeout: int,
    *,
    output_root: Path,
) -> Iterable[Path]:
    """
    Gather markdown paths by converting PDFs (local/remote) through MinerU or by
    reusing existing markdown files under the provided folder.
    """
    path = Path(pdf_folder)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".md":
            yield path
            return
        if suffix in {".txt", ".urls"}:
            urls = _read_urls_file(path)
            yield from _process_urls(
                urls,
                settings,
                mineru_model,
                mineru_timeout,
                output_root=output_root,
            )
            return
        if suffix == ".pdf":
            yield from _process_local_pdfs(
                [path],
                settings,
                mineru_model,
                mineru_timeout,
                output_root=output_root,
            )
            return

    if path.is_dir():
        md_files = sorted(path.rglob("*.md"))
        for md in md_files:
            yield md

        pdf_files = [
            pdf
            for pdf in sorted(path.rglob("*.pdf"))
            if not _is_within(pdf, output_root)
        ]
        if pdf_files:
            yield from _process_local_pdfs(
                pdf_files,
                settings,
                mineru_model,
                mineru_timeout,
                output_root=output_root,
            )

        urls_files = [
            candidate
            for candidate in (list(path.rglob("*.txt")) + list(path.rglob("*.urls")))
            if not _is_within(candidate, output_root)
        ]
        for urls_file in urls_files:
            urls = _read_urls_file(urls_file)
            if urls:
                yield from _process_urls(
                    urls,
                    settings,
                    mineru_model,
                    mineru_timeout,
                    output_root=output_root,
                )
        return

    if str(pdf_folder).startswith(("http://", "https://")):
        yield from _process_urls(
            [pdf_folder],
            settings,
            mineru_model,
            mineru_timeout,
            output_root=output_root,
        )
        return

    raise FileNotFoundError(f"无法识别的输入源：{pdf_folder}")


def _process_local_pdfs(
    pdf_files: List[Path],
    settings: Settings,
    mineru_model: str,
    mineru_timeout: int,
    *,
    output_root: Path,
) -> Iterable[Path]:
    if not settings.mineru_api_key:
        raise ValueError("构建知识库需要 MINERU_API_KEY 用于解析本地 PDF")
    logger.info("MinerU 开始解析 %s 个本地 PDF", len(pdf_files))
    md_paths = process_local_files_via_mineru(
        file_paths=pdf_files,
        output_root=output_root,
        api_key=settings.mineru_api_key,
        model_version=mineru_model,
        timeout_seconds=mineru_timeout,
    )
    yield from md_paths


def _process_urls(
    urls: List[str],
    settings: Settings,
    mineru_model: str,
    mineru_timeout: int,
    *,
    output_root: Path,
) -> Iterable[Path]:
    if not urls:
        return []
    if not settings.mineru_api_key:
        raise ValueError("构建知识库需要 MINERU_API_KEY 用于解析 PDF URL")
    logger.info("MinerU 开始解析 %s 个 PDF URL", len(urls))
    md_paths = process_urls_via_mineru(
        urls=urls,
        output_root=output_root,
        api_key=settings.mineru_api_key,
        model_version=mineru_model,
        timeout_seconds=mineru_timeout,
    )
    yield from md_paths


def _read_urls_file(path: Path) -> List[str]:
    urls: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                urls.append(stripped)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("读取 URL 文件 %s 失败：%s", path, exc)
    return urls


def _persist_chunks(
    chunks: List[str],
    metadata_base: dict,
    vector_store: VectorStore,
    embedder: Embedder,
    *,
    batch_size: int = 16,
) -> None:
    metadata_records: list[dict] = []
    for idx, chunk in enumerate(chunks):
        metadata_records.append(
            {
                **metadata_base,
                "chunk_index": idx,
                "chunk_total": len(chunks),
            }
        )

    for start in range(0, len(chunks), batch_size):
        chunk_batch = chunks[start : start + batch_size]
        metadata_batch = metadata_records[start : start + batch_size]
        embeddings = embedder.embed_batch(chunk_batch)
        vector_store.add_documents(chunk_batch, embeddings, metadata_batch)


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def load_kb_config(kb_path: str | Path) -> dict:
    path = Path(kb_path) / CONFIG_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - config corruption handling
        logger.warning("无法解析知识库配置 %s：%s", path, exc)
        return {}


def _save_kb_config(kb_dir: Path, payload: dict) -> None:
    config_path = kb_dir / CONFIG_FILENAME
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("知识库配置已更新：%s", config_path)


__all__ = ["build_knowledge_base", "load_kb_config"]
