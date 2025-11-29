from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from ..config import get_settings
from ..core import deepseek_interpretation
from ..logging import configure_logging
from ..services import process_pdf_via_mineru, process_local_files_via_mineru, process_urls_via_mineru, get_batch_results
from ..utils import read_md_content

QUESTIONS = [
    (
        "请用以下模板概括该文档，并将其中的占位符填入具体信息；若文中未提及某项，请写‘未说明’；"
        "若涉及到专业词汇，请在结尾处统一进行解释：[xxxx年]，[xx大学/研究机构]的[xx作者等]"
        "针对[研究问题]，采用[研究手段/方法]，对[研究对象或范围]进行了研究，并发现/得出[主要结论]。"
    )
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process markdown documents or remote PDF files."
    )
    
    # Input source group
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--pdf-url",
        help="URL of the PDF to process with MinerU before analysis.",
    )
    input_group.add_argument(
        "--md-path",
        help="Path to an existing markdown file to process directly.",
    )
    input_group.add_argument(
        "--batch-dir",
        help="Directory containing PDF files to process in batch with MinerU.",
    )
    input_group.add_argument(
        "--batch-id",
        help="Batch ID to check status of previously submitted batch processing.",
    )
    input_group.add_argument(
        "--batch-urls-file",
        nargs="?",
        const="files/batch_urls.txt",
        help="Path to a text file containing URLs of PDF files to process in batch with MinerU (default: files/batch_urls.txt).",
    )
    
    parser.add_argument(
        "--mineru-timeout",
        type=int,
        default=600,
        help="Maximum seconds to wait for MinerU extraction to finish.",
    )
    parser.add_argument(
        "--model-version",
        default="vlm",
        help="MinerU model version to use (default: vlm).",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for DeepSeek model (default: 1.0).",
    )

    subparsers = parser.add_subparsers(dest="command")

    kb_parser = subparsers.add_parser("kb", help="Manage the local knowledge base")
    kb_subparsers = kb_parser.add_subparsers(dest="kb_command")
    kb_subparsers.required = True

    kb_build = kb_subparsers.add_parser("build", help="Parse PDFs and build the knowledge base")
    kb_build.add_argument("pdf_folder", help="Folder, markdown file, or URLs file containing documents")
    kb_build.add_argument(
        "--kb-path",
        help="Directory used to persist the knowledge base (default: <files_root>/kb_store).",
    )
    kb_build.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum characters per chunk (default: 800).",
    )
    kb_build.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="Embedding model name (default: text-embedding-3-large).",
    )
    kb_build.add_argument(
        "--provider",
        choices=["openai", "huggingface"],
        help="Embedding backend provider override.",
    )
    kb_build.add_argument(
        "--mineru-model",
        help="Override MinerU model version for KB build (default: value of --model-version).",
    )

    kb_ask = kb_subparsers.add_parser("ask", help="Query the knowledge base")
    kb_ask.add_argument("question", help="Question to ask across the knowledge base")
    kb_ask.add_argument(
        "--kb-path",
        help="Directory used to persist the knowledge base (default: <files_root>/kb_store).",
    )
    kb_ask.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for answering (default: 5).",
    )
    kb_ask.add_argument(
        "--save",
        help="Optional path to save the answer as markdown.",
    )

    kb_info = kb_subparsers.add_parser("info", help="Show knowledge base statistics")
    kb_info.add_argument(
        "--kb-path",
        help="Directory used to persist the knowledge base (default: <files_root>/kb_store).",
    )

    return parser.parse_args(args=argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    logger = logging.getLogger("chatpdf")
    logger.info("Starting ChatPDFv3 CLI process")

    args = parse_args(argv)
    settings = get_settings()
    files_root = settings.files_root

    files_root.mkdir(parents=True, exist_ok=True)

    if args.command == "kb":
        return _handle_kb_command(args, settings)

    # Handle batch ID query
    if args.batch_id:
        if not settings.mineru_api_key:
            raise ValueError("MINERU_API_KEY environment variable is not set")
        logger.info("Querying batch results for batch_id: %s", args.batch_id)
        batch_results = get_batch_results(
            batch_id=args.batch_id,
            api_key=settings.mineru_api_key,
        )
        logger.info("Batch results: %s", batch_results)
        print(f"Batch {args.batch_id} status: {batch_results.get('status')}")
        print(f"Tasks: {len(batch_results.get('tasks', []))}")
        for task in batch_results.get('tasks', []):
            print(f"  - {task.get('file_name')}: {task.get('state')}")
        return 0

    # Handle batch file processing
    file_paths = []
    urls = []
    if args.batch_dir:
        if not settings.mineru_api_key:
            raise ValueError("MINERU_API_KEY environment variable is not set")
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
        file_paths = list(batch_dir.glob("*.pdf"))
        if not file_paths:
            raise FileNotFoundError(f"No PDF files found in directory: {batch_dir}")
        logger.info("Processing %d PDF files from directory: %s", len(file_paths), batch_dir)
    
    elif args.batch_urls_file is not None:
        if not settings.mineru_api_key:
            raise ValueError("MINERU_API_KEY environment variable is not set")
        
        # Use the provided file path or default
        urls_file = Path(args.batch_urls_file)
        
        if not urls_file.exists():
            raise FileNotFoundError(f"URLs file not found: {urls_file}")
        
        # Read URLs from file
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not urls:
            raise ValueError(f"No valid URLs found in file: {urls_file}")
        
        logger.info("Processing %d URLs from file: %s", len(urls), urls_file)
    
    if file_paths:
        md_paths = process_local_files_via_mineru(
            file_paths=file_paths,
            output_root=files_root,
            api_key=settings.mineru_api_key,
            timeout_seconds=args.mineru_timeout,
            model_version=args.model_version,
        )
        logger.info("Batch processing completed. Generated %d markdown files", len(md_paths))
        for md_path in md_paths:
            logger.info("Processed: %s", md_path)
        
        # Process all markdown files for interpretation
        for md_path in md_paths:
            md_content = read_md_content(md_path)
            interpretation_output = md_path.parent / "interpretation_results.md"
            
            logger.info("Using DeepSeek for interpretation of %s", md_path.name)
            deepseek_interpretation(
                md_content,
                QUESTIONS,
                interpretation_output,
                temperature=args.temperature,
            )
        
        logger.info("ChatPDFv2 CLI process finished")
        return 0
    
    elif urls:
        md_paths = process_urls_via_mineru(
            urls=urls,
            output_root=files_root,
            api_key=settings.mineru_api_key,
            timeout_seconds=args.mineru_timeout,
            model_version=args.model_version,
        )
        logger.info("URL batch processing completed. Generated %d markdown files", len(md_paths))
        for md_path in md_paths:
            logger.info("Processed: %s", md_path)
        
        # Process all markdown files for interpretation
        for md_path in md_paths:
            md_content = read_md_content(md_path)
            interpretation_output = md_path.parent / "interpretation_results.md"
            
            logger.info("Using DeepSeek for interpretation of %s", md_path.name)
            deepseek_interpretation(
                md_content,
                QUESTIONS,
                interpretation_output,
                temperature=args.temperature,
            )
        
        logger.info("ChatPDFv2 CLI process finished")
        return 0
    
    # Handle single file processing
    elif args.pdf_url:
        if not settings.mineru_api_key:
            raise ValueError("MINERU_API_KEY environment variable is not set")
        md_path = process_pdf_via_mineru(
            args.pdf_url,
            output_root=files_root,
            api_key=settings.mineru_api_key,
            timeout_seconds=args.mineru_timeout,
        )
    elif args.md_path:
        md_path = Path(args.md_path)
    else:
        md_path = settings.default_md_path

    # Process markdown content with DeepSeek interpretation
    md_content = read_md_content(md_path)
    interpretation_output = md_path.parent / "interpretation_results.md"

    logger.info("Using DeepSeek for interpretation")
    deepseek_interpretation(
        md_content,
        QUESTIONS,
        interpretation_output,
        temperature=args.temperature,
    )
    logger.info("ChatPDFv2 CLI process finished")
    return 0


def _handle_kb_command(args: argparse.Namespace, settings) -> int:
    try:
        from ..knowledge_base import (  # type: ignore import-not-found
            VectorStore,
            answer_query,
            build_knowledge_base,
            load_kb_config,
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        print(
            "知识库功能需要额外依赖 (chromadb / sentence-transformers)。"
            " 请先运行 `pip install chromadb sentence-transformers`。\n"
            f"详细错误：{exc}"
        )
        return 1

    kb_path = _resolve_kb_path(args, settings)

    if args.kb_command == "build":
        mineru_model = getattr(args, "mineru_model", None) or args.model_version
        stats = build_knowledge_base(
            pdf_folder=args.pdf_folder,
            kb_path=kb_path,
            chunk_size=args.chunk_size,
            model_name=args.embedding_model,
            provider=args.provider,
            mineru_model=mineru_model,
            mineru_timeout=args.mineru_timeout,
        )
        print(f"Knowledge base stored at: {stats['kb_path']}")
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Chunks added: {stats['chunks_added']}")
        return 0

    if args.kb_command == "ask":
        answer = answer_query(
            args.question,
            kb_path=kb_path,
            top_k=args.top_k,
        )
        print(answer)
        if args.save:
            _save_answer_markdown(args.save, args.question, answer)
        return 0

    if args.kb_command == "info":
        store = VectorStore(kb_path)
        stats = store.stats()
        config = load_kb_config(kb_path)
        print(f"KB path: {stats['persist_dir']}")
        print(f"Collection: {stats['collection']}")
        print(f"Chunks: {stats['chunks']}")
        if config:
            print(f"Embedding model: {config.get('model_name')}")
            print(f"Provider: {config.get('provider')}")
            print(f"Chunk size: {config.get('chunk_size')}")
            print(f"Documents tracked: {config.get('documents_processed')}")
            print(f"Last updated: {config.get('last_updated')}")
        else:
            print("No configuration metadata found for this knowledge base.")
        return 0

    raise ValueError(f"Unknown knowledge base command: {args.kb_command}")


def _resolve_kb_path(args: argparse.Namespace, settings) -> Path:
    kb_root = getattr(args, "kb_path", None)
    if kb_root:
        return Path(kb_root).expanduser()
    return (settings.files_root / "kb_store").expanduser()


def _save_answer_markdown(destination: str, question: str, answer: str) -> None:
    output_path = Path(destination).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    block = f"## 问题\n\n{question}\n\n### 回答\n\n{answer.strip()}\n"
    if output_path.exists():
        with output_path.open("a", encoding="utf-8") as fh:
            if output_path.stat().st_size > 0:
                fh.write("\n---\n\n")
            fh.write(block)
    else:
        header = "# 知识库问答记录\n\n"
        output_path.write_text(header + block, encoding="utf-8")
    logger = logging.getLogger("chatpdf")
    logger.info("Answer saved to %s", output_path)


__all__ = ["main", "parse_args"]
